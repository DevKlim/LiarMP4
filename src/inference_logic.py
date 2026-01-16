import torch
import re
import ast
import sys
import os
import logging
import asyncio
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from my_vision_process import process_vision_info, client
from labeling_logic import (
    LABELING_PROMPT_TEMPLATE, SCORE_INSTRUCTIONS_SIMPLE, SCORE_INSTRUCTIONS_REASONING,
    SCHEMA_SIMPLE, SCHEMA_REASONING,
    FCOT_MACRO_PROMPT, FCOT_MESO_PROMPT, FCOT_SYNTHESIS_PROMPT
)
from toon_parser import parse_veracity_toon
from agents import get_video_analysis_system

# Google GenAI Imports
try:
    import google.generativeai as genai_legacy
    from google.generativeai.types import generation_types
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
        Part
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
    # FIX: Declare globals at the start
    global LITE_MODE, processor, base_model, peft_model, active_model
    
    if LITE_MODE:
        logger.info("LITE_MODE is enabled. Skipping local model loading.")
        return
    
    if base_model is not None: return
    
    if not torch.cuda.is_available():
        # In the root container, assume GPU is needed unless forcing LITE mode logic was desired.
        # However, to match the fix:
        logger.warning("CUDA is not available. This application requires a GPU for local models. Switching to LITE_MODE.")
        LITE_MODE = True
        return
    
    device = torch.device("cuda")
    logger.info(f"CUDA is available. Initializing models on {device}...")
    local_model_path = "/app/local_model"
    
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
    except ImportError:
        attn_implementation = "sdpa"

    logger.info(f"Loading base model from {local_model_path}...")
    try:
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            local_model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation=attn_implementation
        ).eval()
        processor = AutoProcessor.from_pretrained(local_model_path)
        active_model = base_model
    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        LITE_MODE = True

def switch_active_model(model_name: str):
    global active_model, base_model, peft_model
    if model_name == "custom" and peft_model is not None:
        active_model = peft_model
    else:
        active_model = base_model

def inference_step(video_path, prompt, generation_kwargs, sampling_fps, pred_glue=None):
    global processor, active_model
    if active_model is None: raise RuntimeError("Models not loaded.")

    messages = [
        {"role": "user", "content": [
                {"type": "video", "video": video_path, 'key_time': pred_glue, 'fps': sampling_fps,
                 "total_pixels": 128*12 * 28 * 28, "min_pixels": 128 * 28 * 28},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, client=client)
    fps_inputs = video_kwargs['fps'][0]
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(active_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = active_model.generate(**inputs, **generation_kwargs, use_cache=True)
    
    generated_ids = [output_ids[i][len(inputs['input_ids'][i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return output_text[0]

async def run_inference_pipeline(video_path, question, generation_config, prompts):
    num_perceptions = int(generation_config.pop("num_perceptions", 3))
    sampling_fps = float(generation_config.pop("sampling_fps", 2.0))
    if generation_config.get("temperature", 0.0) > 0.0: generation_config["do_sample"] = True
    pred_glue = None
    final_answer = "No answer."

    for percption in range(num_perceptions):
        yield f" Perception {percption + 1}/{num_perceptions} \n"
        current_prompt = prompts["glue"].replace("[QUESTION]", question) if percption < num_perceptions - 1 else prompts["final"].replace("[QUESTION]", question)
        ans = inference_step(video_path, current_prompt, generation_kwargs=generation_config, pred_glue=pred_glue, sampling_fps=sampling_fps)
        yield f"Model Output: {ans}\n"
        final_answer = ans
        match_glue = re.search(r'<glue>(.*?)</glue>', ans, re.DOTALL)
        if match_glue:
            pred_glue = ast.literal_eval(match_glue.group(1).strip())
        else:
            pred_glue = None
    yield f"\n Final Answer \n{final_answer}\n"

async def attempt_toon_repair(original_text: str, schema: str, client, model_type: str, config: dict):
    logger.info("Attempting TOON Repair via separate AI call...")
    repair_prompt = (
        f"SYSTEM: You are a data formatting expert. The following output from a previous model "
        f"failed to parse correctly. \n"
        f"YOUR TASK: Rewrite the data strictly into the following TOON schema. Do not add new content, "
        f"just format it. If scores are missing, infer them from the text or default to 0.\n\n"
        f"TARGET SCHEMA:\n{schema}\n\n"
        f"BAD OUTPUT:\n{original_text}\n"
    )
    
    try:
        loop = asyncio.get_event_loop()
        repaired_text = ""
        
        if model_type == 'gemini':
            model = genai_legacy.GenerativeModel("models/gemini-2.0-flash-exp")
            response = await loop.run_in_executor(
                None, 
                lambda: model.generate_content(repair_prompt, generation_config={"temperature": 0.0})
            )
            repaired_text = response.text
            
        elif model_type == 'vertex':
            # Use separate client instance if needed or passed client
            cl = client if client else genai.Client(vertexai=True, project=config['project_id'], location=config['location'])
            response = await loop.run_in_executor(
                None,
                lambda: cl.models.generate_content(
                    model=config['model_name'],
                    contents=repair_prompt,
                    config=GenerateContentConfig(temperature=0.0)
                )
            )
            repaired_text = response.text
            
        logger.info(f"Repair successful. New text length: {len(repaired_text)}")
        return repaired_text
    except Exception as e:
        logger.error(f"Repair failed: {e}")
        return original_text

async def run_gemini_labeling_pipeline(video_path: str, caption: str, transcript: str, gemini_config: dict, include_comments: bool, reasoning_method: str = "cot"):
    if genai_legacy is None:
        yield "ERROR: Legacy SDK missing.\n"
        return
    
    api_key = gemini_config.get("api_key")
    if not api_key: return
    
    try:
        genai_legacy.configure(api_key=api_key)
        loop = asyncio.get_event_loop()
        uploaded_file = await loop.run_in_executor(None, lambda: genai_legacy.upload_file(path=video_path))
        while uploaded_file.state.name == "PROCESSING": await asyncio.sleep(2)
        
        model = genai_legacy.GenerativeModel("models/gemini-2.0-flash-exp")
        toon_schema = SCHEMA_REASONING if include_comments else SCHEMA_SIMPLE
        score_instructions = SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE
        
        raw_text = ""
        prompt_used = ""

        if reasoning_method == "fcot":
            # ... existing fcot logic ...
            pass
        elif reasoning_method == "agentic":
            yield "Starting Multi-Agent Analysis..."
            # Create a clean system instance with the provided API key
            model_name = gemini_config.get("model_name", "models/gemini-2.1-flash-lite")
            system = get_video_analysis_system(api_key=api_key, model_name=model_name)
            
            results = await loop.run_in_executor(None, lambda: system.run(
                video_path=video_path,
                caption=caption,
                transcript=transcript
            ))
            # Concatenate agent outputs for parsing
            raw_text = "\n\n".join([str(res) for res in results])
            prompt_used = "Multi-Agent System Execution"
        else:
            # --- Standard Chain of Thought (Single Turn) ---
            prompt_text = LABELING_PROMPT_TEMPLATE.format(
                caption=caption, 
                transcript=transcript,
                toon_schema=toon_schema,
                score_instructions=score_instructions
            )
            prompt_used = prompt_text
            yield "Generating Labels (Standard CoT)..."
            response = await loop.run_in_executor(None, lambda: model.generate_content([prompt_text, uploaded_file], generation_config={"temperature": 0.1}))
            raw_text = response.text
        
        if not raw_text:
             yield "Model returned empty response (possibly triggered safety filter)."
             yield {"error": "Empty Response"}
             return

        parsed_data = parse_veracity_toon(raw_text)
        
        is_zero = parsed_data['veracity_vectors']['visual_integrity_score'] == '0'
        if is_zero:
             yield "Parsing incomplete (score 0). Initiating Auto-Repair..."
             raw_text = await attempt_toon_repair(raw_text, toon_schema, None, 'gemini', gemini_config)
             parsed_data = parse_veracity_toon(raw_text)

        yield {"raw_toon": raw_text, "parsed_data": parsed_data, "prompt_used": prompt_used}
        await loop.run_in_executor(None, lambda: genai_legacy.delete_file(name=uploaded_file.name))

    except Exception as e:
        yield f"ERROR: {e}"

async def run_vertex_labeling_pipeline(video_path: str, caption: str, transcript: str, vertex_config: dict, include_comments: bool, reasoning_method: str = "cot"):
    if genai is None:
        yield "ERROR: 'google-genai' not installed.\n"
        return

    project_id = vertex_config.get("project_id")
    location = vertex_config.get("location", "us-central1")
    model_name = vertex_config.get("model_name", "gemini-1.5-pro-preview-0409")
    
    if not project_id: return

    try:
        client = genai.Client(vertexai=True, project=project_id, location=location)
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        video_part = Part.from_bytes(data=video_bytes, mime_type="video/mp4")

        toon_schema = SCHEMA_REASONING if include_comments else SCHEMA_SIMPLE
        score_instructions = SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE
        
        raw_text = ""
        prompt_used = ""
        
        loop = asyncio.get_event_loop()
        config = GenerateContentConfig(
            temperature=0.1,
            response_mime_type="text/plain",
            max_output_tokens=8192,
            tools=[Tool(google_search=GoogleSearch())]
        )

        if reasoning_method == "fcot":
            # --- Fractal Chain of Thought (Vertex) ---
            yield "Starting Fractal Chain of Thought (Vertex FCoT)..."
            chat = client.chats.create(model=model_name, config=config)
            
            # 1. Macro
            yield "FCoT Step 1: Macro-Scale..."
            macro_prompt = FCOT_MACRO_PROMPT.format(caption=caption, transcript=transcript)
            # First message must contain the video
            res1 = await loop.run_in_executor(None, lambda: chat.send_message([video_part, macro_prompt]))
            macro_hypothesis = res1.text
            yield f"Hypothesis: {macro_hypothesis[:80]}...\n"

            # 2. Meso
            yield "FCoT Step 2: Meso-Scale..."
            meso_prompt = FCOT_MESO_PROMPT.format(macro_hypothesis=macro_hypothesis)
            res2 = await loop.run_in_executor(None, lambda: chat.send_message(meso_prompt))
            
            # 3. Synthesis
            yield "FCoT Step 3: Synthesis..."
            synthesis_prompt = FCOT_SYNTHESIS_PROMPT.format(toon_schema=toon_schema, score_instructions=score_instructions)
            res3 = await loop.run_in_executor(None, lambda: chat.send_message(synthesis_prompt))
            
            raw_text = res3.text
            prompt_used = f"FCoT (Vertex):\nMacro: {macro_prompt}\nMeso: {meso_prompt}\nSynthesis: {synthesis_prompt}"

        else:
            # --- Standard CoT ---
            prompt_text = LABELING_PROMPT_TEMPLATE.format(
                caption=caption, 
                transcript=transcript,
                toon_schema=toon_schema,
                score_instructions=score_instructions
            )
            prompt_used = prompt_text
            yield "Generating Labels (Vertex CoT)..."
            
            response = await loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[video_part, prompt_text],
                    config=config
                )
            )
            raw_text = response.text
        
        if not raw_text:
             yield "Model returned empty response."
             yield {"error": "Empty Response"}
             return

        parsed_data = parse_veracity_toon(raw_text)
        
        is_zero = parsed_data['veracity_vectors']['visual_integrity_score'] == '0'
        if is_zero:
            yield "Parsing incomplete (score 0). Initiating Auto-Repair..."
            raw_text = await attempt_toon_repair(raw_text, toon_schema, client, 'vertex', vertex_config)
            parsed_data = parse_veracity_toon(raw_text)

        yield {"raw_toon": raw_text, "parsed_data": parsed_data, "prompt_used": prompt_used}
            
    except Exception as e:
        yield f"ERROR: {e}"
        logger.error("Vertex Labeling Error", exc_info=True)

# Keep legacy pipeline functions for general Q&A compatibility
async def run_gemini_pipeline(video_path, question, checks, gemini_config, generation_config=None):
    if genai_legacy is None: return
    api_key = gemini_config.get("api_key")
    if not api_key: return
    genai_legacy.configure(api_key=api_key)
    loop = asyncio.get_event_loop()
    uploaded_file = await loop.run_in_executor(None, lambda: genai_legacy.upload_file(path=video_path))
    while uploaded_file.state.name == "PROCESSING": await asyncio.sleep(2)
    model = genai_legacy.GenerativeModel(gemini_config.get("model_name", "models/gemini-1.5-pro-latest"))
    response = await loop.run_in_executor(None, lambda: model.generate_content([question, uploaded_file]))
    yield response.text
    await loop.run_in_executor(None, lambda: genai_legacy.delete_file(name=uploaded_file.name))

async def run_vertex_pipeline(video_path, question, checks, vertex_config, generation_config=None):
    if genai is None: return
    client = genai.Client(vertexai=True, project=vertex_config['project_id'], location=vertex_config['location'])
    with open(video_path, 'rb') as f: video_bytes = f.read()
    video_part = Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=vertex_config.get("model_name", "gemini-2.5-flash-lite"),
            contents=[video_part, question]
        )
    )
    yield response.text
