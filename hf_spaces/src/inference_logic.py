import torch
import re
import ast
import sys
import os
import time
import logging
import asyncio
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from labeling_logic import (
    LABELING_PROMPT_TEMPLATE, SCORE_INSTRUCTIONS_SIMPLE, SCORE_INSTRUCTIONS_REASONING,
    SCHEMA_SIMPLE, SCHEMA_REASONING,
    FCOT_MACRO_PROMPT, FCOT_MESO_PROMPT, FCOT_SYNTHESIS_PROMPT
)
from toon_parser import parse_veracity_toon

# Optional local imports
try:
    from my_vision_process import process_vision_info, client
except ImportError:
    process_vision_info = None
    client = None

# Google GenAI Imports
try:
    import google.generativeai as genai_legacy
    from google.generativeai.types import generation_types, HarmCategory, HarmBlockThreshold
except ImportError:
    genai_legacy = None

try:
    # Modern Google GenAI SDK (v1)
    from google import genai
    from google.genai.types import (
        GenerateContentConfig,
        HttpOptions,
        Retrieval,
        Tool,
        VertexAISearch,
        GoogleSearch,
        Part,
        SafetySetting
    )
    import vertexai
except ImportError:
    genai = None
    vertexai = None

LITE_MODE = os.getenv("LITE_MODE", "false").lower() == "true"
processor = None
base_model = None
peft_model = None
active_model = None
logger = logging.getLogger(__name__)

def load_models():
    pass 

async def attempt_toon_repair(original_text: str, schema: str, client, model_type: str, config: dict):
    logger.info("Attempting TOON Repair...")
    repair_prompt = f"SYSTEM: Reformat the following text into strict TOON schema. Infer missing scores as 0.\n\nSCHEMA:\n{schema}\n\nINPUT:\n{original_text}\n"
    try:
        loop = asyncio.get_event_loop()
        repaired_text = ""
        if model_type == 'gemini':
            model = genai_legacy.GenerativeModel("models/gemini-2.0-flash-exp")
            response = await loop.run_in_executor(None, lambda: model.generate_content(repair_prompt))
            repaired_text = response.text
        elif model_type == 'vertex':
            cl = client if client else genai.Client(vertexai=True, project=config['project_id'], location=config['location'])
            response = await loop.run_in_executor(None, lambda: cl.models.generate_content(model=config['model_name'], contents=repair_prompt))
            repaired_text = response.text
        return repaired_text
    except Exception as e:
        logger.error(f"Repair failed: {e}")
        return original_text

async def run_gemini_labeling_pipeline(video_path: str, caption: str, transcript: str, gemini_config: dict, include_comments: bool, reasoning_method: str = "cot"):
    if genai_legacy is None:
        yield "ERROR: Legacy SDK missing.\n"
        return
    
    api_key = gemini_config.get("api_key")
    if not api_key: 
        yield "ERROR: No Gemini API Key provided."
        return
    
    logger.info(f"[Gemini] Initializing with model {gemini_config.get('model_name')}")
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    try:
        genai_legacy.configure(api_key=api_key)
        loop = asyncio.get_event_loop()
        
        # 1. Upload File
        logger.info(f"[Gemini] Uploading video file: {video_path}...")
        yield f"Uploading video to Gemini..."
        
        uploaded_file = await loop.run_in_executor(None, lambda: genai_legacy.upload_file(path=video_path, mime_type="video/mp4"))
        logger.info(f"[Gemini] Upload complete. URI: {uploaded_file.uri} | State: {uploaded_file.state.name}")

        # 2. Wait for Processing (Fix: Refresh state in loop)
        wait_start = time.time()
        while True:
            # Refresh file status
            uploaded_file = await loop.run_in_executor(None, lambda: genai_legacy.get_file(uploaded_file.name))
            state_name = uploaded_file.state.name
            
            if state_name == "ACTIVE":
                logger.info("[Gemini] Video processing complete. Ready for inference.")
                break
            elif state_name == "FAILED":
                logger.error(f"[Gemini] Video processing failed on server side.")
                yield "ERROR: Google failed to process video."
                return
            
            if time.time() - wait_start > 300: # 5 minute timeout
                logger.error("[Gemini] Video processing timed out.")
                yield "ERROR: Video processing timed out."
                return
                
            logger.info(f"[Gemini] Processing video... (State: {state_name})")
            yield "Processing video on Google servers..."
            await asyncio.sleep(5)
        
        # 3. Prepare Inference
        model_name = gemini_config.get("model_name") or "models/gemini-2.0-flash-exp"
        model = genai_legacy.GenerativeModel(model_name)
        toon_schema = SCHEMA_REASONING if include_comments else SCHEMA_SIMPLE
        score_instructions = SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE
        
        raw_text = ""
        prompt_used = ""
        gen_config = {"temperature": 0.1}

        logger.info(f"[Gemini] Starting inference with method: {reasoning_method}")

        if reasoning_method == "fcot":
            yield "Starting FCoT (Gemini)..."
            chat = model.start_chat(history=[])
            
            macro_prompt = FCOT_MACRO_PROMPT.format(caption=caption, transcript=transcript)
            logger.info("[Gemini] Sending Macro Prompt...")
            res1 = await loop.run_in_executor(None, lambda: chat.send_message([uploaded_file, macro_prompt], safety_settings=safety_settings))
            macro_hypothesis = res1.text
            yield f"Hypothesis: {macro_hypothesis[:100]}...\n"

            meso_prompt = FCOT_MESO_PROMPT.format(macro_hypothesis=macro_hypothesis)
            logger.info("[Gemini] Sending Meso Prompt...")
            res2 = await loop.run_in_executor(None, lambda: chat.send_message(meso_prompt, safety_settings=safety_settings))
            
            synthesis_prompt = FCOT_SYNTHESIS_PROMPT.format(toon_schema=toon_schema, score_instructions=score_instructions)
            logger.info("[Gemini] Sending Synthesis Prompt...")
            res3 = await loop.run_in_executor(None, lambda: chat.send_message(synthesis_prompt, safety_settings=safety_settings))
            
            raw_text = res3.text
            prompt_used = f"FCoT:\n{macro_prompt}\n..."
        else:
            prompt_text = LABELING_PROMPT_TEMPLATE.format(caption=caption, transcript=transcript, toon_schema=toon_schema, score_instructions=score_instructions)
            prompt_used = prompt_text
            yield f"Generating Labels ({model_name})..."
            logger.info("[Gemini] Sending standard generation request...")
            response = await loop.run_in_executor(
                None, 
                lambda: model.generate_content([prompt_text, uploaded_file], generation_config=gen_config, safety_settings=safety_settings)
            )
            raw_text = response.text
        
        # Log response info
        logger.info(f"[Gemini] Response received. Length: {len(raw_text)}")
        if not raw_text:
             yield "Model returned empty response (Check API quota or safety)."
             yield {"error": "Empty Response - likely safety block"}
             return
        
        parsed_data = parse_veracity_toon(raw_text)
        if parsed_data['veracity_vectors']['visual_integrity_score'] == '0':
             yield "Auto-Repairing output..."
             raw_text = await attempt_toon_repair(raw_text, toon_schema, None, 'gemini', gemini_config)
             parsed_data = parse_veracity_toon(raw_text)

        yield {"raw_toon": raw_text, "parsed_data": parsed_data, "prompt_used": prompt_used}
        
        # Cleanup
        try:
            logger.info(f"[Gemini] Deleting remote file {uploaded_file.name}")
            await loop.run_in_executor(None, lambda: genai_legacy.delete_file(name=uploaded_file.name))
        except Exception as cleanup_err:
            logger.warning(f"Failed to cleanup file: {cleanup_err}")

    except Exception as e:
        logger.error(f"Gemini Pipeline Error: {e}", exc_info=True)
        yield f"ERROR (Gemini): {e}"

async def run_vertex_labeling_pipeline(video_path: str, caption: str, transcript: str, vertex_config: dict, include_comments: bool, reasoning_method: str = "cot"):
    if genai is None:
        yield "ERROR: 'google-genai' not installed.\n"
        return

    project_id = vertex_config.get("project_id")
    if not project_id:
        yield "ERROR: No Vertex Project ID."
        return

    logger.info(f"[Vertex] Initializing for project {project_id}")

    safety_settings = [
        SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
        SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
        SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
        SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
    ]

    try:
        client = genai.Client(vertexai=True, project=project_id, location=vertex_config.get("location", "us-central1"))
        
        # For Vertex, we send bytes directly (up to a limit) or use Cloud Storage. 
        # v1 SDK Part.from_bytes is easiest for small/medium videos (< 20MB approx, but allows more in some versions).
        # For larger videos in HF Spaces, this might time out if not using GCS.
        # Assuming direct upload for now.
        logger.info(f"[Vertex] Reading local video file: {video_path}")
        with open(video_path, 'rb') as f: video_bytes = f.read()
        video_part = Part.from_bytes(data=video_bytes, mime_type="video/mp4")

        toon_schema = SCHEMA_REASONING if include_comments else SCHEMA_SIMPLE
        score_instructions = SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE
        model_name = vertex_config.get("model_name", "gemini-2.5-flash-lite")
        
        raw_text = ""
        prompt_used = ""
        loop = asyncio.get_event_loop()
        config = GenerateContentConfig(
            temperature=0.1, 
            response_mime_type="text/plain", 
            tools=[Tool(google_search=GoogleSearch())],
            safety_settings=safety_settings
        )

        logger.info(f"[Vertex] Starting inference with {model_name}")

        if reasoning_method == "fcot":
            yield "Starting FCoT (Vertex)..."
            chat = client.chats.create(model=model_name, config=config)
            
            macro_prompt = FCOT_MACRO_PROMPT.format(caption=caption, transcript=transcript)
            logger.info("[Vertex] Sending Macro Prompt...")
            res1 = await loop.run_in_executor(None, lambda: chat.send_message([video_part, macro_prompt]))
            macro_hypothesis = res1.text
            yield f"Hypothesis: {macro_hypothesis[:80]}...\n"

            meso_prompt = FCOT_MESO_PROMPT.format(macro_hypothesis=macro_hypothesis)
            logger.info("[Vertex] Sending Meso Prompt...")
            res2 = await loop.run_in_executor(None, lambda: chat.send_message(meso_prompt))
            
            synthesis_prompt = FCOT_SYNTHESIS_PROMPT.format(toon_schema=toon_schema, score_instructions=score_instructions)
            logger.info("[Vertex] Sending Synthesis Prompt...")
            res3 = await loop.run_in_executor(None, lambda: chat.send_message(synthesis_prompt))
            
            raw_text = res3.text
            prompt_used = f"FCoT (Vertex):\n{macro_prompt}..."

        else:
            prompt_text = LABELING_PROMPT_TEMPLATE.format(caption=caption, transcript=transcript, toon_schema=toon_schema, score_instructions=score_instructions)
            prompt_used = prompt_text
            yield f"Generating Labels ({model_name})..."
            logger.info("[Vertex] Sending standard generation request...")
            response = await loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(model=model_name, contents=[video_part, prompt_text], config=config)
            )
            raw_text = response.text
        
        logger.info(f"[Vertex] Response Length: {len(raw_text)}")
        if not raw_text:
             yield "Model returned empty response."
             yield {"error": "Empty Response"}
             return

        parsed_data = parse_veracity_toon(raw_text)
        if parsed_data['veracity_vectors']['visual_integrity_score'] == '0':
            yield "Auto-Repairing output..."
            raw_text = await attempt_toon_repair(raw_text, toon_schema, client, 'vertex', vertex_config)
            parsed_data = parse_veracity_toon(raw_text)

        yield {"raw_toon": raw_text, "parsed_data": parsed_data, "prompt_used": prompt_used}
            
    except Exception as e:
        yield f"ERROR (Vertex): {e}"
        logger.error("Vertex Labeling Error", exc_info=True)

async def run_gemini_pipeline(video_path, question, checks, gemini_config, generation_config=None):
    yield "Legacy pipeline not fully supported in HF Space."

async def run_vertex_pipeline(video_path, question, checks, vertex_config, generation_config=None):
    yield "Legacy pipeline not fully supported in HF Space."