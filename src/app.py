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

# Standard columns for AI Dataset
DATASET_COLUMNS = [
    "id", "link", "timestamp", "caption", 
    "final_veracity_score", "visual_score", "audio_score", "logic_score", 
    "align_video_caption", "classification", "reasoning", "tags", "raw_toon"
]

# --- Helper: Schema Migration ---
def ensure_csv_schema(file_path: Path, fieldnames: list):
    if not file_path.exists():
        return

    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            try:
                start_pos = f.tell()
                line = f.readline()
                if not line: return
                existing_header = [h.strip() for h in line.split(',')]
                missing = [col for col in fieldnames if col not in existing_header]
                if not missing: return
                logger.info(f"Schema mismatch detected in {file_path}. Missing columns: {missing}. Migrating...")
                f.seek(start_pos)
                dict_reader = csv.DictReader(f)
                rows = list(dict_reader)
            except Exception as e:
                logger.warning(f"Could not parse CSV for schema check: {e}")
                return

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        
        logger.info(f"Schema migration complete for {file_path}.")

    except Exception as e:
        logger.error(f"Error during schema migration for {file_path}: {e}")

def robust_read_csv(file_path: Path):
    if not file_path.exists(): 
        return

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
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

def update_queue_status(link: str, status: str):
    q_path = Path("data/batch_queue.csv")
    if not q_path.exists(): return
    
    rows = []
    updated = False
    norm_target = normalize_link(link)
    
    with open(q_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["link", "ingest_timestamp", "status"]
        if "status" not in fieldnames: fieldnames.append("status")
        
        for row in reader:
            if normalize_link(row.get("link", "")) == norm_target:
                row["status"] = status
                updated = True
            rows.append(row)
            
    if updated:
        with open(q_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

def log_queue_error(link: str, error_msg: str):
    p = Path("data/queue_errors.csv")
    with open(p, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not p.exists() or p.stat().st_size == 0:
            writer.writerow(["link", "timestamp", "error"])
        writer.writerow([link, datetime.datetime.now().isoformat(), error_msg])
    update_queue_status(link, "Error")

async def prepare_video_assets(link: str, output_id: str) -> dict:
    video_dir = Path("data/videos")
    video_path = video_dir / f"{output_id}.mp4"
    audio_path = video_dir / f"{output_id}.wav"
    transcript_path = video_dir / f"{output_id}.vtt"
    
    caption = ""
    video_downloaded = False
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': str(video_path),
        'quiet': True, 'ignoreerrors': True, 'no_warnings': True, 'skip_download': False 
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=False)
            if info:
                caption = info.get('description', '') or info.get('title', '')
                formats = info.get('formats', [])
                if not formats and not info.get('url'):
                     logger.info(f"No video formats found for {link}. Treating as text-only.")
                else:
                    if not video_path.exists(): ydl.download([link])
    except Exception as e:
        logger.error(f"Download error for {link}: {e}")
    
    if video_path.exists() and video_path.stat().st_size > 0:
        video_downloaded = True
        if not audio_path.exists():
            subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if audio_path.exists() and not transcript_path.exists():
            transcription.load_model()
            transcription.generate_transcript(str(audio_path))

    return {
        "video": str(video_path) if video_downloaded else None,
        "transcript": str(transcript_path) if video_downloaded and transcript_path.exists() else None,
        "caption": caption
    }

@app.on_event("startup")
async def startup_event():
    logging.info("Application starting up...")
    ensure_csv_schema(Path("data/dataset.csv"), DATASET_COLUMNS)
    ensure_csv_schema(Path("data/manual_dataset.csv"), GROUND_TRUTH_FIELDS)
    if not LITE_MODE:
        try: inference_logic.load_models()
        except Exception: pass

@app.get("/benchmarks/stats")
async def get_benchmark_stats():
    return benchmarking.calculate_benchmarks()

@app.post("/benchmarks/train_predictive")
async def run_predictive_training(config: dict = Body(...)):
    return benchmarking.train_predictive_sandbox(config)

@app.get("/config/prompts")
async def list_prompts():
    return [{"id": k, "name": v['description']} for k, v in PROMPT_VARIANTS.items()]

@app.get("/config/tags")
async def list_configured_tags():
    path = Path("data/tags.json")
    if path.exists():
        with open(path, 'r') as f: return json.load(f)
    return {}

@app.post("/config/tags")
async def save_configured_tags(tags: dict = Body(...)):
    path = Path("data/tags.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(tags, f, indent=2)
    return {"status": "success"}

@app.get("/tags/list")
async def list_all_tags():
    tags_count = {}
    path = Path("data/dataset.csv")
    if path.exists():
        for row in robust_read_csv(path):
            t_str = row.get("tags", "")
            if t_str:
                for t in t_str.split(','):
                    t = t.strip()
                    if t: tags_count[t] = tags_count.get(t, 0) + 1
    sorted_tags = sorted(tags_count.items(), key=lambda x: x[1], reverse=True)
    return [{"name": k, "count": v} for k, v in sorted_tags]

@app.post("/extension/ingest")
async def extension_ingest_link(request: Request):
    try:
        data = await request.json()
        link = data.get("link")
        comments = data.get("comments", [])
        if not link:
            raise HTTPException(status_code=400, detail="Link required")
        
        q_path = Path("data/batch_queue.csv")
        existing = set()
        if q_path.exists():
            for r in robust_read_csv(q_path): existing.add(normalize_link(r.get('link')))
            
        normalized = normalize_link(link)
        if normalized not in existing:
            with open(q_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not q_path.exists() or q_path.stat().st_size == 0: writer.writerow(["link", "ingest_timestamp", "status"])
                writer.writerow([link.strip(), datetime.datetime.now().isoformat(), "Pending"])
        
        if comments:
            tid = extract_tweet_id(link) or hashlib.md5(link.encode()).hexdigest()[:10]
            context_path = Path(f"data/comments/{tid}_ingest.json")
            with open(context_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "link": link,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "comments": comments
                }, f, indent=2)
            logger.info(f"Saved {len(comments)} comments for ingestion context: {tid}")

        return {"status": "success", "link": link, "comments_saved": len(comments)}
    except Exception as e:
        logger.error(f"Ingest Error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/manual/promote")
async def promote_to_ground_truth(request: Request):
    try:
        data = await request.json()
        target_ids = data.get("ids", [])
        if not target_ids and data.get("id"): target_ids = [data.get("id")]
        
        if not target_ids: return JSONResponse({"status": "error", "message": "No IDs provided"}, status_code=400)

        ai_path = Path("data/dataset.csv")
        ai_rows = {}
        if ai_path.exists():
            for row in robust_read_csv(ai_path):
                if row.get('id'): ai_rows[str(row['id'])] = row
        
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
            if tid_str in existing_ids: continue 
            found_row = ai_rows.get(tid_str)
            if found_row:
                mapped_row = {
                    "id": found_row.get("id"), "link": found_row.get("link"),
                    "timestamp": datetime.datetime.now().isoformat(), "caption": found_row.get("caption"),
                    "visual_integrity_score": found_row.get("visual_score", 0),
                    "audio_integrity_score": found_row.get("audio_score", 0),
                    "source_credibility_score": 5, "logical_consistency_score": found_row.get("logic_score", 0),
                    "emotional_manipulation_score": 5, "video_audio_score": 5, 
                    "video_caption_score": found_row.get("align_video_caption", 0), "audio_caption_score": 5,
                    "final_veracity_score": found_row.get("final_veracity_score", 0),
                    "final_reasoning": found_row.get("reasoning", ""),
                    "stats_likes": 0, "stats_shares": 0, "stats_comments": 0, "stats_platform": "twitter",
                    "tags": found_row.get("tags", ""), "classification": found_row.get("classification", "None"),
                    "source": "manual_promoted"
                }
                new_rows.append(mapped_row)
                promoted_count += 1
                existing_ids.add(tid_str)

        if not new_rows: return {"status": "success", "promoted_count": 0}

        mode = 'a' if manual_exists else 'w'
        with open(manual_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS, extrasaction='ignore')
            if not manual_exists or manual_path.stat().st_size == 0: writer.writeheader()
            for r in new_rows: writer.writerow(r)

        return {"status": "success", "promoted_count": promoted_count}
    except Exception as e: return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

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
        with open(manual_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get('id')) in target_ids:
                    deleted_count += 1
                    continue
                rows.append(row)
        with open(manual_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            
        return {"status": "success", "deleted_count": deleted_count}
    except Exception as e: return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/dataset/delete")
async def delete_dataset_items(request: Request):
    try:
        data = await request.json()
        target_ids = data.get("ids", [])
        if not target_ids: raise HTTPException(status_code=400)
        target_ids = set(str(t) for t in target_ids)

        path = Path("data/dataset.csv")
        if not path.exists(): return {"status": "success", "count": 0}

        rows = []
        deleted_count = 0
        for row in robust_read_csv(path):
            if str(row.get('id')) in target_ids:
                deleted_count += 1
            else:
                rows.append(row)

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
            
        return {"status": "success", "deleted_count": deleted_count}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

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
                    with open(hist, 'r', encoding='utf-8', errors='ignore') as f: count = sum(1 for _ in f) - 1
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
    try:
        data = await request.json()
        link = data.get("link")
        if not link: 
            return JSONResponse({"status": "error", "message": "Link required"}, status_code=400)
        
        tweet_id = extract_tweet_id(link) or hashlib.md5(link.encode()).hexdigest()[:10]
        labels = data.get("labels", data)

        row = {
            "id": tweet_id, "link": link, "timestamp": datetime.datetime.now().isoformat(),
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

        tag_str = str(row["tags"])
        tag_list = [t.strip() for t in tag_str.split(',') if t.strip()]

        deep_json = {
            "veracity_vectors": {
                "visual_integrity_score": str(row["visual_integrity_score"]),
                "audio_integrity_score": str(row["audio_integrity_score"]),
                "source_credibility_score": str(row["source_credibility_score"]),
                "logical_consistency_score": str(row["logical_consistency_score"]),
                "emotional_manipulation_score": str(row["emotional_manipulation_score"])
            },
            "modalities": {
                "video_audio_score": str(row["video_audio_score"]),
                "video_caption_score": str(row["video_caption_score"]),
                "audio_caption_score": str(row["audio_caption_score"])
            },
            "video_context_summary": row["caption"], 
            "tags": tag_list,
            "factuality_factors": {
                "claim_accuracy": "Manual",
                "evidence_gap": "Manual Verification",
                "grounding_check": "Manual Verification"
            },
            "disinformation_analysis": {
                "classification": row["classification"],
                "intent": "Manual Labeling",
                "threat_vector": "Manual Labeling"
            },
            "final_assessment": {
                "veracity_score_total": str(row["final_veracity_score"]),
                "reasoning": row["final_reasoning"]
            },
            "raw_parsed_structure": {
                "summary": {"text": row["caption"]},
                "tags": {"keywords": row["tags"]},
                "final": {"score": str(row["final_veracity_score"]), "reasoning": row["final_reasoning"]}
            },
            "meta_info": {
                "id": tweet_id,
                "timestamp": row["timestamp"],
                "link": link,
                "model_selection": "Manual"
            }
        }

        json_path_direct = Path(f"data/labels/{tweet_id}.json")
        with open(json_path_direct, 'w', encoding='utf-8') as jf:
            json.dump(deep_json, jf, indent=2, ensure_ascii=False)
        
        with open(Path(f"data/mnl_labeled/{tweet_id}.json"), 'w', encoding='utf-8') as jf:
            json.dump(row, jf, indent=2, ensure_ascii=False)

        manual_path = Path("data/manual_dataset.csv")
        exists = manual_path.exists()
        ensure_csv_schema(manual_path, GROUND_TRUTH_FIELDS)

        rows = []
        found = False
        if exists:
            for r in robust_read_csv(manual_path):
                if str(r.get('id')) == str(tweet_id):
                    clean_row = {k: row.get(k, "") for k in GROUND_TRUTH_FIELDS}
                    rows.append(clean_row)
                    found = True
                else:
                    rows.append(r)
        
        if not found:
            clean_row = {k: row.get(k, "") for k in GROUND_TRUTH_FIELDS}
            rows.append(clean_row)
            
        with open(manual_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        
        update_queue_status(link, "Processed")
        return {"status": "success", "id": tweet_id}
    except Exception as e:
        logger.error(f"Save Manual Error: {e}")
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
    dataset = []
    m_path = Path("data/manual_dataset.csv")
    manual_ids = set()
    if m_path.exists():
         for row in robust_read_csv(m_path):
             row['source'] = 'Manual'
             if row.get('id'): manual_ids.add(str(row['id']))
             dataset.append(row)
    
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

@app.post("/queue/add")
async def add_queue_item(link: str = Body(..., embed=True)):
    q_path = Path("data/batch_queue.csv")
    existing = set()
    if q_path.exists():
        for r in robust_read_csv(q_path): existing.add(normalize_link(r.get('link')))
        
    normalized = normalize_link(link)
    if not normalized: raise HTTPException(status_code=400, detail="Invalid link")
    if normalized in existing: return {"status": "ignored", "message": "Link already in queue"}
        
    with open(q_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not q_path.exists() or q_path.stat().st_size == 0: writer.writerow(["link", "ingest_timestamp", "status"])
        writer.writerow([link.strip(), datetime.datetime.now().isoformat(), "Pending"])
    return {"status": "success", "link": link}

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
        if not q_path.exists() or q_path.stat().st_size == 0: writer.writerow(["link", "ingest_timestamp", "status"])
        for line in lines:
            if 'http' in line:
                raw = line.split(',')[0].strip()
                if normalize_link(raw) not in existing:
                    writer.writerow([raw, datetime.datetime.now().isoformat(), "Pending"])
                    added += 1
    return {"status": "success", "added_count": added}

@app.post("/queue/stop")
async def stop_processing():
    global STOP_QUEUE_SIGNAL
    STOP_QUEUE_SIGNAL = True
    return {"status": "success", "message": "Stopping queue processing..."}

@app.post("/queue/clear_processed")
async def clear_processed_queue():
    q_path = Path("data/batch_queue.csv")
    if not q_path.exists(): return {"status": "success", "removed_count": 0}
    p_ids, p_links = get_processed_indices()
    kept_rows = []
    removed_count = 0
    for row in robust_read_csv(q_path):
        link = row.get("link")
        status = row.get("status", "Pending")
        if status == "Processed" or check_if_processed(link, p_ids, p_links): removed_count += 1
        else: kept_rows.append(row)
    with open(q_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["link", "ingest_timestamp", "status"])
        writer.writeheader()
        writer.writerows(kept_rows)
    return {"status": "success", "removed_count": removed_count}

@app.post("/queue/delete")
async def delete_queue_items(request: Request):
    try:
        data = await request.json()
        target_links = set(normalize_link(l) for l in data.get("links", []))
        q_path = Path("data/batch_queue.csv")
        if not q_path.exists(): return {"status": "success", "count": 0}
        kept_rows = []
        deleted_count = 0
        for row in robust_read_csv(q_path):
            if normalize_link(row.get('link')) in target_links: deleted_count += 1
            else: kept_rows.append(row)
        with open(q_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["link", "ingest_timestamp", "status"])
            writer.writeheader()
            writer.writerows(kept_rows)
        return {"status": "success", "count": deleted_count}
    except Exception as e: return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

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
            status = row.get("status", "Pending")
            if status == "Pending" and check_if_processed(l, p_ids, p_links): status = "Processed"
            items.append({"link": l, "timestamp": row.get("ingest_timestamp",""), "status": status})
    return items

@app.post("/queue/run")
async def run_queue_processing(
    model_selection: str = Form(...),
    gemini_api_key: str = Form(""), gemini_model_name: str = Form(""),
    vertex_project_id: str = Form(""), vertex_location: str = Form(""), vertex_model_name: str = Form(""), vertex_api_key: str = Form(""),
    include_comments: bool = Form(False), reasoning_method: str = Form("cot"), prompt_template: str = Form("standard"),
    custom_query: str = Form(""), max_reprompts: int = Form(1)
):
    global STOP_QUEUE_SIGNAL
    STOP_QUEUE_SIGNAL = False
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name, "max_retries": max_reprompts}
    vertex_config = {"project_id": vertex_project_id, "location": vertex_location, "model_name": vertex_model_name, "api_key": vertex_api_key, "max_retries": max_reprompts}
    sel_p = PROMPT_VARIANTS.get(prompt_template, PROMPT_VARIANTS['standard'])
    system_persona_txt = sel_p['instruction']
    if custom_query.strip(): system_persona_txt += f"\n\nSPECIAL INSTRUCTION FOR THIS BATCH: {custom_query}"
    
    active_config = vertex_config if model_selection == 'vertex' else gemini_config

    async def queue_stream():
        q_path = Path("data/batch_queue.csv")
        items = [r for r in robust_read_csv(q_path) if r.get("link") and r.get("status", "Pending") == "Pending"]
        p_ids, p_links = get_processed_indices()
        yield f"data: [SYSTEM] Persona: {sel_p['description']}\n\n"
        
        for item in items:
            link = item.get("link")
            if STOP_QUEUE_SIGNAL: 
                yield f"data: [SYSTEM] Stopping by user request.\n\n"
                break
            if check_if_processed(link, p_ids, p_links): 
                update_queue_status(link, "Processed")
                continue
            
            yield f"data: [START] {link}\n\n"
            tid = extract_tweet_id(link) or hashlib.md5(link.encode()).hexdigest()[:10]
            assets = await prepare_video_assets(link, tid)
            if not assets or (not assets.get('video') and not assets.get('caption')):
                log_queue_error(link, "Download/Fetch Error (No Content)")
                yield f"data:   - Download Error.\n\n"
                continue

            trans = parse_vtt(assets['transcript']) if assets.get('transcript') else "No transcript (Audio/Video missing)."
            video_file = assets.get('video')
            if not video_file: 
                yield f"data:   - No video found. Text-only analysis.\n\n"
                video_file = None 
            else: yield f"data:   - Video found. Inferencing...\n\n"
            
            comments_path = Path(f"data/comments/{tid}_ingest.json")
            current_system_persona = system_persona_txt
            if comments_path.exists():
                try:
                    with open(comments_path, 'r') as f:
                        c_data = json.load(f)
                        comments = c_data.get('comments', [])
                        if comments:
                            yield f"data:   - Found {len(comments)} comments. Generating Community Context...\n\n"
                            community_summary = await inference_logic.generate_community_summary(comments, model_selection, active_config)
                            current_system_persona += f"\n\n### COMMUNITY NOTES / CONTEXT (from Comments):\n{community_summary}\n\nUse this community context to cross-reference claims but remain objective."
                            yield f"data:   - Context Generated.\n\n"
                except Exception as e:
                    logger.error(f"Error processing comments for context: {e}")

            res_data = None
            if model_selection == 'gemini':
                async for chunk in inference_logic.run_gemini_labeling_pipeline(video_file, assets['caption'], trans, gemini_config, include_comments, reasoning_method, current_system_persona, request_id=tid):
                    if isinstance(chunk, str): yield f"data:   - {chunk}\n\n"
                    else: res_data = chunk
            elif model_selection == 'vertex':
                async for chunk in inference_logic.run_vertex_labeling_pipeline(video_file, assets['caption'], trans, vertex_config, include_comments, reasoning_method, current_system_persona, request_id=tid):
                    if isinstance(chunk, str): yield f"data:   - {chunk}\n\n"
                    else: res_data = chunk

            if res_data and "parsed_data" in res_data:
                parsed = res_data["parsed_data"]
                d_path = Path("data/dataset.csv")
                ensure_csv_schema(d_path, DATASET_COLUMNS)
                exists = d_path.exists()
                try:
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
                        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS, extrasaction='ignore')
                        if not exists: writer.writeheader()
                        writer.writerow(row)
                except Exception as csv_err: logger.error(f"CSV Write Failed: {csv_err}")

                try:
                    ts = datetime.datetime.now().isoformat()
                    ts_clean = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    flat_parsed = parsed.copy()
                    flat_parsed["meta_info"] = { "id": tid, "timestamp": ts, "link": link, "prompt_used": res_data.get("prompt_used", ""), "model_selection": model_selection }
                    with open(Path(f"data/labels/{tid}_{ts_clean}.json"), 'w', encoding='utf-8') as f: json.dump(flat_parsed, f, indent=2, ensure_ascii=False)
                except Exception as e: logger.error(f"Sidecar Error: {e}")

                p_ids.add(tid)
                p_links.add(normalize_link(link))
                update_queue_status(link, "Processed")
                yield f"data: [SUCCESS] Saved.\n\n"
            else: 
                err_msg = res_data.get('error') if isinstance(res_data, dict) else "Inference failed"
                log_queue_error(link, err_msg)
                yield f"data: [FAIL] {err_msg}.\n\n"
            await asyncio.sleep(0.5)
        yield "event: close\ndata: Done\n\n"

    return StreamingResponse(queue_stream(), media_type="text/event-stream")
