# factuality_logic.py
import os
import re
import json
import logging
import asyncio
from pathlib import Path
import inference_logic
from toon_parser import parse_toon_line
# Reuse parsing logic from common_utils if desired, or keep as duplicate if standalone
from common_utils import parse_vtt

logger = logging.getLogger(__name__)

# --- Enhanced TOON Prompts for Individual Checks ---
# Using TOON reduces output tokens significantly compared to JSON.

PROMPT_VISUAL_ARTIFACTS = (
    "Analyze the video for visual manipulation (Deepfakes, editing anomalies).\n"
    "Steps inside <thinking>: 1. Scan for artifacts. 2. Check cuts.\n"
    "Output TOON format:\n"
    "visual_analysis: result[2]{score,justification}:\n"
    "Score(1-10),\"Justification text\""
)

PROMPT_CONTENT_ANALYSIS = (
    "Analyze the content for accuracy and logic.\n"
    "Steps inside <thinking>: 1. Identify claims. 2. Check fallacies. 3. Assess emotion.\n"
    "**Transcript:**\n{transcript}\n"
    "Output TOON format:\n"
    "content_analysis: result[2]{score,justification}:\n"
    "Score(1-10),\"Justification text\""
)

PROMPT_AUDIO_ANALYSIS = (
    "Analyze audio for synthesis or manipulation.\n"
    "Steps inside <thinking>: 1. Listen for robotic inflections. 2. Check lip-sync.\n"
    "**Transcript:**\n{transcript}\n"
    "Output TOON format:\n"
    "audio_analysis: result[2]{score,justification}:\n"
    "Score(1-10),\"Justification text\""
)

async def run_factuality_pipeline(paths: dict, checks: dict, generation_config: dict):
    """
    Asynchronously runs a pipeline of factuality checks, parses TOON scores, and yields results.
    """
    video_path = paths.get("video")
    transcript_path = paths.get("transcript")

    if not video_path:
        yield "ERROR: Video path not found. Cannot start analysis.\n\n"
        return

    yield "Step 1: Processing Transcript...\n"
    await asyncio.sleep(0.1)
    transcript = "No transcript was downloaded for this video."
    if transcript_path and os.path.exists(transcript_path):
        transcript = parse_vtt(transcript_path)
        yield f"  - Transcript file found and processed.\n"
    else:
        yield f"  - No transcript file was found.\n"
    
    yield f"\n--- Extracted Transcript ---\n{transcript}\n--------------------------\n\n"
    await asyncio.sleep(0.1)

    analysis_steps = []
    if checks.get("visuals"):
        analysis_steps.append(("Visual Integrity", PROMPT_VISUAL_ARTIFACTS))
    if checks.get("content"):
        analysis_steps.append(("Content Veracity", PROMPT_CONTENT_ANALYSIS.format(transcript=transcript)))
    if checks.get("audio"):
        analysis_steps.append(("Audio Forensics", PROMPT_AUDIO_ANALYSIS.format(transcript=transcript)))

    for i, (title, prompt) in enumerate(analysis_steps):
        yield f"--- Step {i + 2}: Running '{title}' Analysis ---\n"
        yield "(Model is generating TOON analysis with scores...)\n\n"
        await asyncio.sleep(0.1)

        try:
            current_gen_config = generation_config.copy()
            sampling_fps = current_gen_config.pop("sampling_fps", 2.0)
            current_gen_config.pop("num_perceptions", None)
            
            # FORCE LOW TEMP for structured TOON analysis
            current_gen_config["temperature"] = 0.1 
            current_gen_config["do_sample"] = True
                
            ans = inference_logic.inference_step(
                video_path=video_path,
                prompt=prompt,
                generation_kwargs=current_gen_config,
                sampling_fps=sampling_fps,
                pred_glue=None
            )

            yield f"  - Analysis Complete for '{title}'. Parsing TOON...\n\n"
            
            # --- Attempt to parse TOON from the model's response ---
            parsed_result = {}
            # Regex to find the TOON data line: key: type[count]{headers}:\nVALUE
            match = re.search(r'(\w+_analysis): result\[2\]\{score,justification\}:\s*\n(.+)', ans, re.MULTILINE)
            
            thinking = "No thinking block found."
            think_match = re.search(r'<thinking>(.*?)</thinking>', ans, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()

            if match:
                key, value_line = match.groups()
                parsed_result = parse_toon_line({'key': key, 'headers': ['score', 'justification']}, value_line.strip())
            else:
                logger.warning(f"Could not parse TOON for '{title}'. Raw: {ans}")
                yield f"Warning: Model did not return valid TOON. Raw output:\n{ans}\n"
                continue

            # --- Display the parsed, structured result ---
            score = parsed_result.get('score', 'N/A')
            justification = parsed_result.get('justification', 'No justification provided.')
            
            yield f"===== ANALYSIS RESULT: {title.upper()} =====\n"
            yield f"SCORE: {score}/10\n"
            yield f"Reasoning (Step-by-Step): {thinking}\n"
            yield f"Final Justification: {justification}\n\n"
            yield f"========================================\n\n"

        except Exception as e:
            error_message = f"An error occurred during the '{title}' analysis step: {e}"
            logger.error(error_message, exc_info=True)
            yield f"ERROR: {error_message}\n\n"
            break
    
    yield "Factuality Analysis Pipeline Finished.\n"