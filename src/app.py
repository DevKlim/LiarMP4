import os
import sys
import asyncio
import subprocess
from pathlib import Path
import logging
import csv
import io
import datetime
import json
import hashlib
import re
from fastapi import FastAPI, Request, Form, UploadFile, File, Body, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, Response, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
import inference_logic
import factuality_logic
import transcription
import user_analysis_logic 
from factuality_logic import parse_vtt
from toon_parser import parse_veracity_toon
from labeling_logic import PROMPT_VARIANTS, LABELING_PROMPT_TEMPLATE, FCOT_MACRO_PROMPT
import benchmarking

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
LITE_MODE = os.getenv("LITE_MODE", "false").lower() == "true"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = "static"
if os.path.isdir("/usr/share/vchat/static"):
    STATIC_DIR = "/usr/share/vchat/static"
elif os.path.isdir("frontend/dist"):
    STATIC_DIR = "frontend/dist"
elif not os.path.isdir(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
    
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if not os.path.isdir("data/videos"):
    os.makedirs("data/videos", exist_ok=True)
app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")

templates = Jinja2Templates(directory=STATIC_DIR)

# Ensure all data directories exist
for d in ["data", "data/videos", "data/labels", "data/prompts", "data/responses", "metadata", "data/profiles", "data/comments", "data/mnl_labeled", "models/sandbox_autogluon"]:
    os.makedirs(d, exist_ok=True)

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)

STOP_QUEUE_SIGNAL = False

# --- CONSTANTS ---
GROUND_TRUTH_FIELDS = [
    "id", "link", "timestamp", "caption", 
    "visual_integrity_score", "audio_integrity_score", "source_credibility_score", 
    "logical_consistency_score", "emotional_manipulation_score",
    "video_audio_score", "video_caption_score", "audio_caption_score",
    "final_veracity_score", "final_reasoning",
    "stats_likes", "stats_shares", "stats_comments", "stats_platform",
    "tags", "classification", "source"
]

# --- Helper: Robust CSV Reader ---
def robust_read_csv(file_path: Path):
    if not file_path.exists(): 
        return

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Skip NULL bytes
            clean_lines = (line.replace('\0', '') for line in f)
            reader = csv.DictReader(clean_lines)
            for row in reader:
                if row:
                    yield row
    except Exception as e:
        logger.error(f"Error reading CSV {file_path}: {e}")
        return

def extract_tweet_id(url: str) -> str | None:
    if not url: return None
    match = re.search(r"(?:twitter|x)\.com/[^/]+/status/(\d+)", url)
    if match: return match.group(1)
    return None

def normalize_link(link: str) -> str:
    if not link: return ""
    return link.split('?')[0].strip().rstrip('/').replace('http://', '').replace('https://', '').replace('www.', '')

def get_processed_indices():
    processed_ids = set()
    processed_links = set()
    
    for filename in ["data/dataset.csv", "data/manual_dataset.csv"]:
        path = Path(filename)
        for row in robust_read_csv(path):
            if row.get('id'): processed_ids.add(row.get('id'))
            if row.get('link'): processed_links.add(normalize_link(row.get('link')))
            
    return processed_ids, processed_links

def check_if_processed(link: str, processed_ids=None, processed_links=None) -> bool:
    target_id = extract_tweet_id(link)
    link_clean = normalize_link(link)
    
    if processed_ids is None or processed_links is None:
        p_ids, p_links = get_processed_indices()
    else:
        p_ids, p_links = processed_ids, processed_links

    if target_id and target_id in p_ids: return True
    if link_clean and link_clean in p_links: return True
    return False

# --- Video Download Helper ---
async def prepare_video_assets(link: str, output_id: str) -> dict:
    video_dir = Path("data/videos")
    video_path = video_dir / f"{output_id}.mp4"
    audio_path = video_dir / f"{output_id}.wav"
    transcript_path = video_dir / f"{output_id}.vtt"
    
    caption = ""
    if not video_path.exists():
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': str(video_path),
            'quiet': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(link, download=True)
                caption = info.get('description', '') or info.get('title', '')
        except Exception as e:
            logger.error(f"Download failed for {link}: {e}")
            return None
    
    if video_path.exists() and not audio_path.exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path), 
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
            str(audio_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    if audio_path.exists() and not transcript_path.exists():
        transcription.load_model()
        transcription.generate_transcript(str(audio_path))
        
    return {
        "video": str(video_path),
        "transcript": str(transcript_path) if transcript_path.exists() else None,
        "caption": caption
    }

@app.on_event("startup")
async def startup_event():
    logging.info("Application starting up...")
    if not LITE_MODE:
        try:
            inference_logic.load_models()
        except Exception: pass

# ============================================================================
# Benchmarking & Predictive Routes
# ============================================================================

@app.get("/benchmarks/stats")
async def get_benchmark_stats():
    return benchmarking.calculate_benchmarks()

@app.post("/benchmarks/train_predictive")
async def run_predictive_training(config: dict = Body(...)):
    return benchmarking.train_predictive_sandbox(config)

@app.get("/config/prompts")
async def list_prompts():
    return [{"id": k, "name": v['description']} for k, v in PROMPT_VARIANTS.items()]

# ============================================================================
# Manual & Ground Truth Management
# ============================================================================

@app.post("/manual/promote")
async def promote_to_ground_truth(request: Request):
    """
    Promotes selected AI-labeled items to the Manual Ground Truth dataset.
    Accepts { "ids": ["id1", "id2"] } or { "id": "id1" }
    """
    try:
        data = await request.json()
        target_ids = data.get("ids", [])
        if not target_ids and data.get("id"):
            target_ids = [data.get("id")]
        
        if not target_ids: 
            return JSONResponse({"status": "error", "message": "No IDs provided"}, status_code=400)

        # Load AI dataset (Source of items)
        ai_path = Path("data/dataset.csv")
        ai_rows = {}
        if ai_path.exists():
            for row in robust_read_csv(ai_path):
                if row.get('id'):
                    ai_rows[str(row['id'])] = row
        
        # Load Manual dataset (Destination, check for duplicates)
        manual_path = Path("data/manual_dataset.csv")
        manual_exists = manual_path.exists()
        
        existing_ids = set()
        if manual_exists:
            for row in robust_read_csv(manual_path):
                if row.get('id'): existing_ids.add(str(row['id']))

        new_rows = []
        promoted_count = 0
        
        for tid in target_ids:
            tid_str = str(tid)
            if tid_str in existing_ids:
                logger.info(f"ID {tid_str} already in Ground Truth. Skipping.")
                continue 
                
            found_row = ai_rows.get(tid_str)
            if found_row:
                # Map AI fields to Strict Ground Truth Schema
                mapped_row = {
                    "id": found_row.get("id"),
                    "link": found_row.get("link"),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "caption": found_row.get("caption"),
                    
                    "visual_integrity_score": found_row.get("visual_score", 0),
                    "audio_integrity_score": found_row.get("audio_score", 0),
                    "source_credibility_score": 5, 
                    "logical_consistency_score": found_row.get("logic_score", 0),
                    "emotional_manipulation_score": 5,

                    "video_audio_score": 5, 
                    "video_caption_score": found_row.get("align_video_caption", 0),
                    "audio_caption_score": 5,

                    "final_veracity_score": found_row.get("final_veracity_score", 0),
                    "final_reasoning": found_row.get("reasoning", ""),
                    
                    "stats_likes": 0, "stats_shares": 0, "stats_comments": 0, "stats_platform": "twitter",
                    "tags": found_row.get("tags", ""),
                    "classification": found_row.get("classification", "None"),
                    "source": "Manual"
                }
                new_rows.append(mapped_row)
                promoted_count += 1
                existing_ids.add(tid_str) # Prevent dupes within same batch

        if not new_rows:
            return {"status": "success", "promoted_count": 0, "message": "No new items to promote (duplicates or not found)."}

        # Append to CSV
        mode = 'a' if manual_exists else 'w'
        with open(manual_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS, extrasaction='ignore')
            if not manual_exists or manual_path.stat().st_size == 0: 
                writer.writeheader()
            
            for r in new_rows:
                # Fill missing schema fields with empty strings
                clean_r = {k: r.get(k, "") for k in GROUND_TRUTH_FIELDS}
                writer.writerow(clean_r)

        return {"status": "success", "promoted_count": promoted_count}
    except Exception as e:
        logger.error(f"Promote error: {e}", exc_info=True)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/manual/delete")
async def delete_ground_truth(request: Request):
    try:
        data = await request.json()
        target_ids = data.get("ids", [])
        if not target_ids and data.get("id"): target_ids = [data.get("id")]
        
        if not target_ids: raise HTTPException(status_code=400)
        target_ids = [str(t) for t in target_ids]

        manual_path = Path("data/manual_dataset.csv")
        if not manual_path.exists(): return {"status": "error", "message": "File not found"}

        rows = []
        deleted_count = 0
        
        # Read existing
        with open(manual_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get('id')) in target_ids:
                    deleted_count += 1
                    continue
                rows.append(row)
        
        # Write back
        with open(manual_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            
        return {"status": "success", "deleted_count": deleted_count}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# ============================================================================
# Profiles & Community Routes
# ============================================================================

@app.get("/profiles/list")
async def list_profiles():
    profiles_dir = Path("data/profiles")
    profiles = []
    if not profiles_dir.exists(): return profiles
    try:
        for d in profiles_dir.iterdir():
            if d.is_dir():
                hist = d / "history.csv"
                count = 0
                if hist.exists():
                    with open(hist, 'r', encoding='utf-8', errors='ignore') as f:
                        count = sum(1 for _ in f) - 1
                profiles.append({"username": d.name, "posts_count": max(0, count)})
    except Exception: pass
    return sorted(profiles, key=lambda x: x['username'])

@app.get("/profiles/{username}/posts")
async def get_profile_posts(username: str):
    csv_path = Path(f"data/profiles/{username}/history.csv")
    posts = []
    if not csv_path.exists(): return posts
    p_ids, p_links = get_processed_indices()
    try:
        for row in robust_read_csv(csv_path):
            link = row.get('link', '')
            is_labeled = False
            t_id = extract_tweet_id(link)
            if t_id and t_id in p_ids: is_labeled = True
            elif normalize_link(link) in p_links: is_labeled = True
            row['is_labeled'] = is_labeled
            posts.append(row)
    except Exception: pass
    return posts

@app.post("/extension/ingest_user_history")
async def ingest_user_history(request: Request):
    try:
        data = await request.json()
        username = data.get("username")
        posts = data.get("posts", [])
        if not username or not posts: raise HTTPException(status_code=400)
        profile_dir = Path(f"data/profiles/{username}")
        profile_dir.mkdir(parents=True, exist_ok=True)
        csv_path = profile_dir / "history.csv"
        file_exists = csv_path.exists()
        existing = set()
        if file_exists:
            for row in robust_read_csv(csv_path): existing.add(row.get('link'))
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ["link", "timestamp", "text", "is_reply", "metric_replies", "metric_reposts", "metric_likes", "metric_views", "ingested_at"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists: writer.writeheader()
            ts = datetime.datetime.now().isoformat()
            count = 0
            for p in posts:
                if p['link'] not in existing:
                    m = p.get('metrics', {})
                    writer.writerow({
                        "link": p.get('link'), "timestamp": p.get('timestamp'),
                        "text": p.get('text', '').replace('\n', ' '), "is_reply": p.get('is_reply', False),
                        "metric_replies": m.get('replies', 0), "metric_reposts": m.get('reposts', 0),
                        "metric_likes": m.get('likes', 0), "metric_views": m.get('views', 0),
                        "ingested_at": ts
                    })
                    count += 1
        return {"status": "success", "new_posts": count}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/extension/save_comments")
async def extension_save_comments(request: Request):
    try:
        data = await request.json()
        link = data.get("link")
        comments = data.get("comments", [])
        if not link: raise HTTPException(status_code=400)
        tweet_id = extract_tweet_id(link) or hashlib.md5(link.encode()).hexdigest()[:10]
        csv_path = Path(f"data/comments/{tweet_id}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["author", "text", "timestamp"])
            writer.writeheader()
            ts = datetime.datetime.now().isoformat()
            for c in comments:
                writer.writerow({"author": c.get("author", "Unknown"), "text": c.get("text", "").replace("\n", " "), "timestamp": ts})
        return {"status": "success", "count": len(comments)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/extension/save_manual")
@app.post("/manual/save")
async def save_manual_label(request: Request):
    """
    Saves manually labeled data.
    1. Saves detailed JSON to data/mnl_labeled/{id}.json
    2. Appends to data/manual_dataset.csv (Ground Truth) if not already present.
    """
    try:
        data = await request.json()
        link = data.get("link")
        if not link: 
            return JSONResponse({"status": "error", "message": "Link required"}, status_code=400)
        
        tweet_id = extract_tweet_id(link) or hashlib.md5(link.encode()).hexdigest()[:10]
        labels = data.get("labels", data) # Handle nested or flat

        # Construct Row matching GROUND_TRUTH_FIELDS
        row = {
            "id": tweet_id,
            "link": link,
            "timestamp": datetime.datetime.now().isoformat(),
            "caption": data.get("caption", ""),
            
            "visual_integrity_score": labels.get("visual_integrity_score", 0),
            "audio_integrity_score": labels.get("audio_integrity_score", 0),
            "source_credibility_score": labels.get("source_credibility_score", 0),
            "logical_consistency_score": labels.get("logical_consistency_score", 0),
            "emotional_manipulation_score": labels.get("emotional_manipulation_score", 5),
            
            "video_audio_score": labels.get("video_audio_score", 0),
            "video_caption_score": labels.get("video_caption_score", 0),
            "audio_caption_score": labels.get("audio_caption_score", 0),
            
            "final_veracity_score": labels.get("final_veracity_score", 0),
            "final_reasoning": labels.get("reasoning", labels.get("final_reasoning", "")),
            
            "stats_likes": 0, "stats_shares": 0, "stats_comments": 0, "stats_platform": "twitter",
            "tags": data.get("tags", labels.get("tags", "")),
            "classification": labels.get("classification", "None"),
            "source": "Manual"
        }

        # 1. Save Detailed JSON
        json_path = Path(f"data/mnl_labeled/{tweet_id}.json")
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(row, jf, indent=2, ensure_ascii=False)
        
        # 2. Append to CSV (Ground Truth)
        manual_path = Path("data/manual_dataset.csv")
        exists = manual_path.exists()
        
        # Check for existing entry to avoid duplicates
        existing_ids = set()
        if exists:
            for r in robust_read_csv(manual_path):
                if r.get('id'): existing_ids.add(str(r['id']))
        
        if str(tweet_id) not in existing_ids:
            with open(manual_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS, extrasaction='ignore')
                if not exists or manual_path.stat().st_size == 0: 
                    writer.writeheader()
                
                clean_row = {k: row.get(k, "") for k in GROUND_TRUTH_FIELDS}
                writer.writerow(clean_row)
            logger.info(f"Saved manual label for {tweet_id} to CSV and JSON.")
        else:
            logger.info(f"ID {tweet_id} already in Ground Truth CSV. Updated JSON only.")
            
        return {"status": "success", "id": tweet_id}
        
    except Exception as e:
        logger.error(f"Error saving manual label: {e}", exc_info=True)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/community/list_datasets")
async def list_community_datasets():
    path = Path("data/comments")
    files = []
    if path.exists():
        for f in path.glob("*.csv"):
            files.append({"id": f.stem, "count": sum(1 for _ in open(f, encoding='utf-8'))-1})
    return files

@app.post("/community/analyze")
async def analyze_community(dataset_id: str = Body(..., embed=True)):
    path = Path(f"data/comments/{dataset_id}.csv")
    if not path.exists(): raise HTTPException(status_code=404)
    comments = list(robust_read_csv(path))
    if not comments: return {"score": 0, "verdict": "No Data"}
    
    s_keys = ["fake", "lie", "staged", "bs", "propaganda", "ai", "deepfake"]
    t_keys = ["true", "real", "confirmed", "fact", "source", "proof"]
    s_count = sum(1 for c in comments if any(k in c['text'].lower() for k in s_keys))
    t_count = sum(1 for c in comments if any(k in c['text'].lower() for k in t_keys))
    
    score = max(0, min(100, 50 + (t_count * 2) - (s_count * 5)))
    verdict = "Community Skepticism" if score < 30 else "Community Verification" if score > 70 else "Neutral/Mixed"
    
    return {"dataset_id": dataset_id, "trust_score": score, "verdict": verdict, "details": {"skeptical_comments": s_count, "trusting_comments": t_count}}

@app.get("/dataset/list")
async def get_dataset_list():
    """
    Returns a unified list of data items from both AI output and Manual Ground Truth.
    """
    dataset = []
    
    # 1. Load Manual Ground Truth first (Highest Priority)
    m_path = Path("data/manual_dataset.csv")
    manual_ids = set()
    if m_path.exists():
         for row in robust_read_csv(m_path):
             row['source'] = 'Manual'
             if row.get('id'): manual_ids.add(str(row['id']))
             dataset.append(row)
             
    # 2. Load AI Data (avoid duplicates if ID already in Manual)
    path = Path("data/dataset.csv")
    if path.exists():
         for row in robust_read_csv(path):
             tid = str(row.get('id', ''))
             if tid not in manual_ids:
                 row['source'] = 'AI'
                 dataset.append(row)
                 
    return sorted(dataset, key=lambda x: x.get('timestamp', ''), reverse=True)

@app.get("/analytics/account_integrity")
async def get_account_integrity():
    id_map = {}
    prof_dir = Path("data/profiles")
    if prof_dir.exists():
        for d in prof_dir.iterdir():
            for row in robust_read_csv(d/"history.csv"):
                tid = extract_tweet_id(row.get('link',''))
                if tid: id_map[tid] = d.name

    scores_map = {}
    for fname in ["data/dataset.csv", "data/manual_dataset.csv"]:
        for row in robust_read_csv(Path(fname)):
            tid = row.get('id')
            sc = row.get('final_veracity_score', '0')
            try: val = float(re.sub(r'[^\d.]', '', str(sc)))
            except: val = 0
            
            auth = id_map.get(tid, "Unknown")
            if auth != "Unknown":
                if auth not in scores_map: scores_map[auth] = []
                scores_map[auth].append(val)
    
    return sorted([{"username": k, "avg_veracity": round(sum(v)/len(v),1), "posts_labeled": len(v)} for k,v in scores_map.items()], key=lambda x: x['avg_veracity'], reverse=True)

@app.post("/queue/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    lines = contents.decode('utf-8').splitlines()
    q_path = Path("data/batch_queue.csv")
    existing = set()
    if q_path.exists():
        for r in robust_read_csv(q_path): existing.add(normalize_link(r.get('link')))
    
    added = 0
    with open(q_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not q_path.exists() or q_path.stat().st_size == 0: writer.writerow(["link", "ingest_timestamp"])
        for line in lines:
            if 'http' in line:
                raw = line.split(',')[0].strip()
                if normalize_link(raw) not in existing:
                    writer.writerow([raw, datetime.datetime.now().isoformat()])
                    added += 1
    return {"status": "success", "added_count": added}

@app.post("/analyze/user_context")
async def analyze_user_context(request: Request):
    try:
        data = await request.json()
        rep = await user_analysis_logic.generate_user_profile_report(data.get("username"))
        return {"status": "success", "report": rep}
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/queue/list")
async def get_queue_list():
    q_path = Path("data/batch_queue.csv")
    items = []
    p_ids, p_links = get_processed_indices()
    for row in robust_read_csv(q_path):
        if row:
            l = row.get("link")
            st = "Processed" if check_if_processed(l, p_ids, p_links) else "Pending"
            items.append({"link": l, "timestamp": row.get("ingest_timestamp",""), "status": st})
    return items

@app.post("/queue/run")
async def run_queue_processing(
    model_selection: str = Form(...),
    gemini_api_key: str = Form(""), gemini_model_name: str = Form(""),
    vertex_project_id: str = Form(""), vertex_location: str = Form(""), vertex_model_name: str = Form(""), vertex_api_key: str = Form(""),
    include_comments: bool = Form(False), reasoning_method: str = Form("cot"), prompt_template: str = Form("standard")
):
    global STOP_QUEUE_SIGNAL
    STOP_QUEUE_SIGNAL = False
    
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name}
    vertex_config = {"project_id": vertex_project_id, "location": vertex_location, "model_name": vertex_model_name, "api_key": vertex_api_key}

    import labeling_logic
    sel_p = PROMPT_VARIANTS.get(prompt_template, PROMPT_VARIANTS['standard'])
    labeling_logic.LABELING_PROMPT_TEMPLATE = LABELING_PROMPT_TEMPLATE.replace("{system_persona}", sel_p['instruction'])
    labeling_logic.FCOT_MACRO_PROMPT = FCOT_MACRO_PROMPT.replace("{system_persona}", sel_p['instruction'])
    
    async def queue_stream():
        q_path = Path("data/batch_queue.csv")
        items = [r.get("link") for r in robust_read_csv(q_path) if r.get("link")]
        p_ids, p_links = get_processed_indices()
        
        yield f"data: [SYSTEM] Persona: {sel_p['description']}\n\n"
        
        for link in items:
            if STOP_QUEUE_SIGNAL: break
            if check_if_processed(link, p_ids, p_links): continue
            
            yield f"data: [START] {link}\n\n"
            tid = extract_tweet_id(link) or hashlib.md5(link.encode()).hexdigest()[:10]
            assets = await prepare_video_assets(link, tid)
            if not assets: 
                yield f"data:   - Download Error\n\n"
                continue

            trans = parse_vtt(assets['transcript']) if assets.get('transcript') else "No transcript."
            yield f"data:   - Inferencing...\n\n"
            
            res_data = None
            if model_selection == 'gemini':
                async for chunk in inference_logic.run_gemini_labeling_pipeline(assets['video'], assets['caption'], trans, gemini_config, include_comments, reasoning_method):
                    if isinstance(chunk, str): yield f"data:   - {chunk}\n\n"
                    else: res_data = chunk
            elif model_selection == 'vertex':
                async for chunk in inference_logic.run_vertex_labeling_pipeline(assets['video'], assets['caption'], trans, vertex_config, include_comments, reasoning_method):
                    if isinstance(chunk, str): yield f"data:   - {chunk}\n\n"
                    else: res_data = chunk

            if res_data and "parsed_data" in res_data:
                parsed = res_data["parsed_data"]
                d_path = Path("data/dataset.csv")
                exists = d_path.exists()
                with open(d_path, 'a', newline='', encoding='utf-8') as f:
                    row = {
                        "id": tid, "link": link, "timestamp": datetime.datetime.now().isoformat(),
                        "caption": assets['caption'],
                        "final_veracity_score": parsed['final_assessment'].get('veracity_score_total', 0),
                        "visual_score": parsed['veracity_vectors'].get('visual_integrity_score', 0),
                        "audio_score": parsed['veracity_vectors'].get('audio_integrity_score', 0),
                        "logic_score": parsed['veracity_vectors'].get('logical_consistency_score', 0),
                        "align_video_caption": parsed['modalities'].get('video_caption_score', 0),
                        "classification": parsed['disinformation_analysis'].get('classification', 'None'),
                        "reasoning": parsed['final_assessment'].get('reasoning', ''),
                        "tags": ",".join(parsed.get('tags', [])),
                        "raw_toon": res_data.get("raw_toon", "")
                    }
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if not exists: writer.writeheader()
                    writer.writerow(row)
                    p_ids.add(tid)
                    p_links.add(normalize_link(link))
                yield f"data: [SUCCESS] Saved.\n\n"
            else: yield f"data: [FAIL] Inference failed.\n\n"
            await asyncio.sleep(0.5)
        yield "event: close\ndata: Done\n\n"

    return StreamingResponse(queue_stream(), media_type="text/event-stream")
