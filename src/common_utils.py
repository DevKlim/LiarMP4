import os
import re
import csv
import logging
import datetime
import subprocess
import hashlib
from pathlib import Path
import yt_dlp
import transcription

logger = logging.getLogger(__name__)

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

def parse_vtt(file_path: str) -> str:
    """Parses a .vtt subtitle file and returns the clean text content."""
    try:
        if not os.path.exists(file_path):
            return "Transcript file not found."
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        text_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('WEBVTT') and not '-->' in line and not line.isdigit():
                clean_line = re.sub(r'<[^>]+>', '', line)
                if clean_line and (not text_lines or clean_line != text_lines[-1]):
                     text_lines.append(clean_line)
        
        return "\n".join(text_lines) if text_lines else "No speech found in transcript."
    except Exception as e:
        logger.error(f"Error parsing VTT file {file_path}: {e}")
        return f"Error reading transcript: {e}"

async def prepare_video_assets(link: str, output_id: str) -> dict:
    video_dir = Path("data/videos")
    if not video_dir.exists():
        video_dir.mkdir(parents=True, exist_ok=True)

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