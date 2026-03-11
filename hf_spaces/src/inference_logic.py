import re
import sys
import os
import time
import logging
import asyncio
import json
import requests
import datetime

# Safe imports for Lite Mode (API only)
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
except ImportError:
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None
    PeftModel = None

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

LITE_MODE = os.getenv("LITE_MODE", "true").lower() == "true"
processor = None
base_model = None
peft_model = None
active_model = None
logger = logging.getLogger(__name__)

TEXT_ONLY_INSTRUCTIONS = """
NOTE: You are operating in TEXT-ONLY mode. The video file could not be analyzed directly.
You must rely entirely on the provided Context (Caption and Transcript) to deduce the veracity.
If the text lacks sufficient detail to score visual or audio integrity, score them as 5 (Neutral/Unknown).
"""

def get_formatted_tag_list():
    return "Suggested tags: politics, satire, deepfake, misleading, true, news"

def load_models():
    pass 
    
def extract_json_from_text(text):
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return {}
    
def smart_merge(base, new_data):
    if not isinstance(new_data, dict): return new_data if new_data else base
    if not isinstance(base, dict): return new_data
    for k, v in new_data.items():
        if k not in base: base[k] = v
        else:
            if isinstance(base[k], dict) and isinstance(v, dict): smart_merge(base[k], v)
            else:
                base_val = base[k]
                new_val = v
                is_base_valid = base_val and str(base_val) != "0" and str(base_val).lower() != "n/a"
                is_new_valid = new_val and str(new_val) != "0" and str(new_val).lower() != "n/a"
                if not is_base_valid and is_new_valid: base[k] = new_val
    return base

def validate_parsed_data(data, is_text_only):
    missing =[]
    if not data.get('video_context_summary'): missing.append("summary")
    final = data.get('final_assessment', {})
    if not final.get('reasoning') or len(str(final.get('reasoning', ''))) < 5: missing.append("final:reasoning")
    vectors = data.get('veracity_vectors', {})
    for k in['visual_integrity_score', 'audio_integrity_score', 'source_credibility_score', 'logical_consistency_score', 'emotional_manipulation_score']:
        if k in['visual_integrity_score', 'audio_integrity_score'] and is_text_only: continue
        v = vectors.get(k)
        if not v or str(v) == '0' or str(v).lower() == 'n/a': missing.append(f"vector:{k}")
    mod = data.get('modalities', {})
    for k in ['video_audio_score', 'video_caption_score', 'audio_caption_score']:
        if k in ['video_audio_score', 'video_caption_score'] and is_text_only: continue
        v = mod.get(k)
        if not v or str(v) == '0' or str(v).lower() == 'n/a': missing.append(f"modality:{k}")
    return missing

def save_debug_log(request_id, kind, content, attempt, label=""):
    if not request_id: return
    try:
        dir_map = {'prompt': 'data/prompts', 'response': 'data/responses'}
        directory = dir_map.get(kind, 'data')
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = f"_{label}" if label else ""
        filename = f"{directory}/{request_id}_{ts}_att{attempt}{safe_label}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(str(content))
    except Exception as e:
        logger.error(f"Failed to save debug log: {e}")

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

async def run_gemini_labeling_pipeline(video_path: str, caption: str, transcript: str, gemini_config: dict, include_comments: bool, reasoning_method: str = "cot", system_persona: str = "", request_id: str = None):
    if genai_legacy is None:
        yield "ERROR: Legacy SDK missing.\n"
        return
    
    api_key = gemini_config.get("api_key")
    if not api_key: 
        yield "ERROR: No Gemini API Key provided."
        return
    
    max_retries = int(gemini_config.get("max_retries", 1))

    safety_settings =[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    try:
        genai_legacy.configure(api_key=api_key)
        loop = asyncio.get_event_loop()
        uploaded_file = None
        is_text_only = False
        
        if video_path and os.path.exists(video_path):
            yield f"Uploading video to Gemini..."
            uploaded_file = await loop.run_in_executor(None, lambda: genai_legacy.upload_file(path=video_path, mime_type="video/mp4"))
            wait_start = time.time()
            while True:
                uploaded_file = await loop.run_in_executor(None, lambda: genai_legacy.get_file(uploaded_file.name))
                state_name = uploaded_file.state.name
                if state_name == "ACTIVE": break
                elif state_name == "FAILED":
                    yield "ERROR: Google failed to process video."
                    return
                if time.time() - wait_start > 300:
                    yield "ERROR: Video processing timed out."
                    return
                yield "Processing video on Google servers..."
                await asyncio.sleep(5)
        else:
            is_text_only = True
        
        model_name = gemini_config.get("model_name") or "models/gemini-2.0-flash-exp"
        model = genai_legacy.GenerativeModel(model_name)
        toon_schema = SCHEMA_REASONING if include_comments else SCHEMA_SIMPLE
        score_instructions = SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE
        
        if is_text_only: system_persona += "\n" + TEXT_ONLY_INSTRUCTIONS
        
        raw_text = ""
        prompt_used = ""
        gen_config = {"temperature": 0.1}
        accumulated_data = {}
        fcot_trace = {}
        full_raw_text = ""

        for attempt in range(max_retries + 1):
            raw_text = ""
            if attempt > 0:
                missing = validate_parsed_data(accumulated_data, is_text_only)
                yield f"Validation failed. Missing or incomplete fields: {missing}. Initiating Iterative Reprompt (Attempt {attempt}/{max_retries}) to acquire remaining factuality components...\n"
                prompt_text = (
                     f"SYSTEM: Review the previous attempt which failed validation.\n"
                     f"CONTEXT: Caption: \"{caption}\"\nTranscript: \"{transcript}\"\n"
                     f"PREVIOUS (PARTIAL) DATA: {json.dumps(accumulated_data, indent=2)}\n"
                     f"MISSING FIELDS: {missing}\n"
                     f"INSTRUCTION: Analyze the provided Video and Context again. "
                     f"Generate the missing fields to complete the schema. You MUST provide the missing scores for {missing}.\n"
                     f"Output the FULL VALID TOON OBJECT containing all required fields.\n"
                     f"{toon_schema}"
                )
                save_debug_log(request_id, 'prompt', prompt_text, attempt, 'reprompt')
                inputs = [prompt_text]
                if uploaded_file: inputs.append(uploaded_file)
                response = await loop.run_in_executor(None, lambda: model.generate_content(inputs, generation_config={"temperature": 0.2}))
                raw_text = response.text
                save_debug_log(request_id, 'response', raw_text, attempt, 'reprompt')
            else:
                if reasoning_method == "fcot":
                    yield "Starting FCoT (Gemini)..."
                    chat = model.start_chat(history=[])
                    
                    macro_prompt = FCOT_MACRO_PROMPT.format(system_persona=system_persona, caption=caption, transcript=transcript)
                    if is_text_only: macro_prompt = "NOTE: Text Only Analysis.\n" + macro_prompt
                    save_debug_log(request_id, 'prompt', macro_prompt, attempt, 'fcot_macro')
                    
                    inputs1 = [macro_prompt]
                    if uploaded_file: inputs1.insert(0, uploaded_file)
                    
                    res1 = await loop.run_in_executor(None, lambda: chat.send_message(inputs1, safety_settings=safety_settings))
                    macro_hypothesis = res1.text
                    save_debug_log(request_id, 'response', macro_hypothesis, attempt, 'fcot_macro')
                    fcot_trace['macro'] = macro_hypothesis
                    yield f"Hypothesis: {macro_hypothesis[:100]}...\n"

                    meso_prompt = FCOT_MESO_PROMPT.format(macro_hypothesis=macro_hypothesis)
                    save_debug_log(request_id, 'prompt', meso_prompt, attempt, 'fcot_meso')
                    res2 = await loop.run_in_executor(None, lambda: chat.send_message(meso_prompt, safety_settings=safety_settings))
                    micro_observations = res2.text
                    save_debug_log(request_id, 'response', micro_observations, attempt, 'fcot_meso')
                    fcot_trace['meso'] = micro_observations
                    
                    synthesis_prompt = FCOT_SYNTHESIS_PROMPT.format(toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=get_formatted_tag_list())
                    save_debug_log(request_id, 'prompt', synthesis_prompt, attempt, 'fcot_synthesis')
                    res3 = await loop.run_in_executor(None, lambda: chat.send_message(synthesis_prompt, safety_settings=safety_settings))
                    raw_text = res3.text
                    save_debug_log(request_id, 'response', raw_text, attempt, 'fcot_synthesis')
                    prompt_used = f"FCoT:\n{macro_prompt}\n..."
                else:
                    prompt_text = LABELING_PROMPT_TEMPLATE.format(system_persona=system_persona, caption=caption, transcript=transcript, toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=get_formatted_tag_list())
                    if is_text_only: prompt_text = "NOTE: Text Only Analysis.\n" + prompt_text
                    prompt_used = prompt_text
                    save_debug_log(request_id, 'prompt', prompt_text, attempt, 'standard')
                    yield f"Generating Labels ({model_name})..."
                    
                    inputs = [prompt_text]
                    if uploaded_file: inputs.append(uploaded_file)
                    
                    response = await loop.run_in_executor(
                        None, 
                        lambda: model.generate_content(inputs, generation_config=gen_config, safety_settings=safety_settings)
                    )
                    raw_text = response.text
                    save_debug_log(request_id, 'response', raw_text, attempt, 'standard')
            
            if raw_text:
                full_raw_text += f"\n--- Attempt {attempt} ---\n{raw_text}\n"
                parsed_step = parse_veracity_toon(raw_text)
                json_data = extract_json_from_text(raw_text)
                if json_data:
                    for k in['veracity_vectors', 'modalities', 'video_context_summary', 'final_assessment', 'factuality_factors', 'disinformation_analysis', 'tags']:
                        if k in json_data:
                            if isinstance(parsed_step.get(k), dict) and isinstance(json_data[k], dict):
                                parsed_step[k].update(json_data[k])
                            else:
                                parsed_step[k] = json_data[k]
                accumulated_data = smart_merge(accumulated_data, parsed_step)

            missing_fields = validate_parsed_data(accumulated_data, is_text_only)
            if not missing_fields:
                yield "Validation Passed. All factuality components processed and confidence scores obtained.\n"
                yield {"raw_toon": full_raw_text, "parsed_data": accumulated_data, "prompt_used": prompt_used, "fcot_trace": fcot_trace}
                break 

            if attempt == max_retries:
                 yield f"Max retries reached. Saving incomplete data.\n"
                 yield {"raw_toon": full_raw_text, "parsed_data": accumulated_data, "prompt_used": prompt_used, "fcot_trace": fcot_trace}
                 break

        if uploaded_file:
            try:
                await loop.run_in_executor(None, lambda: genai_legacy.delete_file(name=uploaded_file.name))
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Gemini Pipeline Error: {e}", exc_info=True)
        yield f"ERROR (Gemini): {e}"


async def run_vertex_labeling_pipeline(video_path: str, caption: str, transcript: str, vertex_config: dict, include_comments: bool, reasoning_method: str = "cot", system_persona: str = "", request_id: str = None):
    if genai is None:
        yield "ERROR: 'google-genai' not installed.\n"
        return

    project_id = vertex_config.get("project_id")
    if not project_id:
        yield "ERROR: No Vertex Project ID."
        return

    safety_settings =[
        SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
        SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
        SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
        SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
    ]

    try:
        api_key = vertex_config.get("api_key")
        if api_key:
            client = genai.Client(vertexai=True, project=project_id, location=vertex_config.get("location", "us-central1"), api_key=api_key)
        else:
            client = genai.Client(vertexai=True, project=project_id, location=vertex_config.get("location", "us-central1"))
        
        video_part = None
        is_text_only = False
        if video_path and os.path.exists(video_path):
            with open(video_path, 'rb') as f: video_bytes = f.read()
            video_part = Part.from_bytes(data=video_bytes, mime_type="video/mp4")
        else:
            is_text_only = True

        toon_schema = SCHEMA_REASONING if include_comments else SCHEMA_SIMPLE
        score_instructions = SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE
        model_name = vertex_config.get("model_name", "gemini-2.5-flash-lite")
        max_retries = int(vertex_config.get("max_retries", 1))
        
        raw_text = ""
        prompt_used = ""
        loop = asyncio.get_event_loop()
        config = GenerateContentConfig(
            temperature=0.1, 
            response_mime_type="text/plain", 
            tools=[Tool(google_search=GoogleSearch())] if vertex_config.get("use_search", True) else None,
            safety_settings=safety_settings
        )

        if is_text_only: system_persona += "\n" + TEXT_ONLY_INSTRUCTIONS

        accumulated_data = {}
        fcot_trace = {}
        full_raw_text = ""

        for attempt in range(max_retries + 1):
            raw_text = ""
            if attempt > 0:
                missing = validate_parsed_data(accumulated_data, is_text_only)
                yield f"Validation failed. Missing or incomplete fields: {missing}. Initiating Iterative Reprompt (Attempt {attempt}/{max_retries}) to acquire remaining factuality components...\n"
                
                prompt_text = (
                     f"SYSTEM: Review the previous attempt which failed validation.\n"
                     f"CONTEXT: Caption: \"{caption}\"\nTranscript: \"{transcript}\"\n"
                     f"PREVIOUS (PARTIAL) DATA: {json.dumps(accumulated_data, indent=2)}\n"
                     f"MISSING FIELDS: {missing}\n"
                     f"INSTRUCTION: Analyze the provided Video and Context again. "
                     f"Generate the missing fields to complete the schema. You MUST provide the missing scores for {missing}.\n"
                     f"Output the FULL VALID TOON OBJECT containing all required fields.\n"
                     f"{toon_schema}"
                )
                
                save_debug_log(request_id, 'prompt', prompt_text, attempt, 'reprompt')
                contents =[prompt_text]
                if video_part: contents.insert(0, video_part)
                
                response = await loop.run_in_executor(None, lambda: client.models.generate_content(model=model_name, contents=contents, config=config))
                raw_text = response.text
                save_debug_log(request_id, 'response', raw_text, attempt, 'reprompt')
            else:
                if reasoning_method == "fcot":
                    yield "Starting FCoT (Vertex)..."
                    chat = client.chats.create(model=model_name, config=config)
                    
                    macro_prompt = FCOT_MACRO_PROMPT.format(system_persona=system_persona, caption=caption, transcript=transcript)
                    save_debug_log(request_id, 'prompt', macro_prompt, attempt, 'fcot_macro')
                    inputs1 = [macro_prompt]
                    if video_part: inputs1.insert(0, video_part)
                    else: inputs1[0] = "NOTE: Text Only Analysis.\n" + inputs1[0]

                    res1 = await loop.run_in_executor(None, lambda: chat.send_message(inputs1))
                    macro_hypothesis = res1.text
                    save_debug_log(request_id, 'response', macro_hypothesis, attempt, 'fcot_macro')
                    fcot_trace['macro'] = macro_hypothesis
                    yield f"Hypothesis: {macro_hypothesis[:80]}...\n"

                    meso_prompt = FCOT_MESO_PROMPT.format(macro_hypothesis=macro_hypothesis)
                    save_debug_log(request_id, 'prompt', meso_prompt, attempt, 'fcot_meso')
                    res2 = await loop.run_in_executor(None, lambda: chat.send_message(meso_prompt))
                    micro_observations = res2.text
                    save_debug_log(request_id, 'response', micro_observations, attempt, 'fcot_meso')
                    fcot_trace['meso'] = micro_observations
                    
                    synthesis_prompt = FCOT_SYNTHESIS_PROMPT.format(toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=get_formatted_tag_list())
                    save_debug_log(request_id, 'prompt', synthesis_prompt, attempt, 'fcot_synthesis')
                    res3 = await loop.run_in_executor(None, lambda: chat.send_message(synthesis_prompt))
                    raw_text = res3.text
                    save_debug_log(request_id, 'response', raw_text, attempt, 'fcot_synthesis')
                    prompt_used = f"FCoT (Vertex):\n{macro_prompt}..."

                else:
                    prompt_text = LABELING_PROMPT_TEMPLATE.format(system_persona=system_persona, caption=caption, transcript=transcript, toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=get_formatted_tag_list())
                    contents = []
                    if video_part: contents = [video_part, prompt_text]
                    else: contents = [f"NOTE: Text Only Analysis (No Video).\n{prompt_text}"]
                    prompt_used = prompt_text
                    save_debug_log(request_id, 'prompt', prompt_text, attempt, 'standard')
                    yield f"Generating Labels ({model_name})..."
                    response = await loop.run_in_executor(
                        None, 
                        lambda: client.models.generate_content(model=model_name, contents=contents, config=config)
                    )
                    raw_text = response.text
                    save_debug_log(request_id, 'response', raw_text, attempt, 'standard')
            
            if not raw_text:
                 yield {"error": "Empty Response"}
                 return

            if raw_text:
                full_raw_text += f"\n--- Attempt {attempt} ---\n{raw_text}\n"
                parsed_step = parse_veracity_toon(raw_text)
                json_data = extract_json_from_text(raw_text)
                if json_data:
                    for k in['veracity_vectors', 'modalities', 'video_context_summary', 'final_assessment', 'factuality_factors', 'disinformation_analysis', 'tags']:
                        if k in json_data:
                            if isinstance(parsed_step.get(k), dict) and isinstance(json_data[k], dict):
                                parsed_step[k].update(json_data[k])
                            else:
                                parsed_step[k] = json_data[k]
                accumulated_data = smart_merge(accumulated_data, parsed_step)

            missing_fields = validate_parsed_data(accumulated_data, is_text_only)
            if not missing_fields:
                yield "Validation Passed. All factuality components processed and confidence scores obtained.\n"
                yield {"raw_toon": full_raw_text, "parsed_data": accumulated_data, "prompt_used": prompt_used, "fcot_trace": fcot_trace}
                break

            if attempt == max_retries:
                 yield f"Max retries reached. Saving incomplete data.\n"
                 yield {"raw_toon": full_raw_text, "parsed_data": accumulated_data, "prompt_used": prompt_used, "fcot_trace": fcot_trace}
                 break
                
    except Exception as e:
        yield f"ERROR (Vertex): {e}"
        logger.error("Vertex Labeling Error", exc_info=True)


async def run_nrp_labeling_pipeline(video_path: str, caption: str, transcript: str, nrp_config: dict, include_comments: bool, reasoning_method: str = "cot", system_persona: str = "", request_id: str = None):
    api_key = nrp_config.get("api_key")
    model_name = nrp_config.get("model_name", "gpt-4")
    base_url = nrp_config.get("base_url", "https://api.openai.com/v1").rstrip("/")
    max_retries = int(nrp_config.get("max_retries", 1))

    if not api_key:
        yield "ERROR: NRP API Key missing.\n"
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    is_text_only = True
    system_persona += "\n" + TEXT_ONLY_INSTRUCTIONS

    toon_schema = SCHEMA_REASONING if include_comments else SCHEMA_SIMPLE
    score_instructions = SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE
    tag_list_text = get_formatted_tag_list()

    accumulated_data = {}
    prompt_used = ""
    fcot_trace = {}
    full_raw_text = ""
    loop = asyncio.get_event_loop()

    async def _call_nrp(messages, attempt_label=""):
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.1
        }
        def do_request():
            resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=600)
            if resp.status_code != 200:
                raise Exception(f"API Error {resp.status_code}: {resp.text}")
            return resp.json()["choices"][0]["message"]["content"]
        return await loop.run_in_executor(None, do_request)

    try:
        for attempt in range(max_retries + 1):
            raw_text = ""
            if attempt > 0:
                missing = validate_parsed_data(accumulated_data, is_text_only)
                yield f"Validation failed. Missing fields: {missing}. Initiating Reprompt...\n"
                prompt_text = (
                     f"SYSTEM: Review the previous attempt which failed validation.\n"
                     f"CONTEXT: Caption: \"{caption}\"\nTranscript: \"{transcript}\"\n"
                     f"PREVIOUS (PARTIAL) DATA: {json.dumps(accumulated_data, indent=2)}\n"
                     f"MISSING FIELDS: {missing}\n"
                     f"INSTRUCTION: Generate the missing fields to complete the schema.\n"
                     f"{toon_schema}"
                )
                save_debug_log(request_id, 'prompt', prompt_text, attempt, 'reprompt')
                raw_text = await _call_nrp([
                    {"role": "system", "content": system_persona},
                    {"role": "user", "content": prompt_text}
                ])
                save_debug_log(request_id, 'response', raw_text, attempt, 'reprompt')
            else:
                if reasoning_method == "fcot":
                    yield "Starting Fractal Chain of Thought (NRP FCoT)...\n"
                    macro_prompt = FCOT_MACRO_PROMPT.format(system_persona=system_persona, caption=caption, transcript=transcript)
                    macro_prompt = "NOTE: Text Only Analysis.\n" + macro_prompt
                    save_debug_log(request_id, 'prompt', macro_prompt, attempt, 'fcot_macro')
                    
                    macro_messages =[{"role": "system", "content": system_persona}, {"role": "user", "content": macro_prompt}]
                    macro_hypothesis = await _call_nrp(macro_messages)
                    save_debug_log(request_id, 'response', macro_hypothesis, attempt, 'fcot_macro')
                    fcot_trace['macro'] = macro_hypothesis

                    meso_prompt = FCOT_MESO_PROMPT.format(macro_hypothesis=macro_hypothesis)
                    save_debug_log(request_id, 'prompt', meso_prompt, attempt, 'fcot_meso')
                    
                    meso_messages = macro_messages +[{"role": "assistant", "content": macro_hypothesis}, {"role": "user", "content": meso_prompt}]
                    micro_observations = await _call_nrp(meso_messages)
                    save_debug_log(request_id, 'response', micro_observations, attempt, 'fcot_meso')
                    fcot_trace['meso'] = micro_observations

                    synthesis_prompt = FCOT_SYNTHESIS_PROMPT.format(toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=tag_list_text)
                    save_debug_log(request_id, 'prompt', synthesis_prompt, attempt, 'fcot_synthesis')
                    
                    synthesis_messages = meso_messages +[{"role": "assistant", "content": micro_observations}, {"role": "user", "content": synthesis_prompt}]
                    raw_text = await _call_nrp(synthesis_messages)
                    save_debug_log(request_id, 'response', raw_text, attempt, 'fcot_synthesis')
                    prompt_used = f"FCoT (NRP):\nMacro: {macro_hypothesis}\nMeso: {micro_observations}"
                    
                else:
                    prompt_text = LABELING_PROMPT_TEMPLATE.format(
                        system_persona=system_persona, caption=caption, transcript=transcript,
                        toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=tag_list_text
                    )
                    prompt_text = f"NOTE: Text Only Analysis (No Video).\n{prompt_text}"
                    prompt_used = prompt_text
                    save_debug_log(request_id, 'prompt', prompt_text, attempt, 'standard')
                    yield "Generating Labels (NRP CoT)...\n"
                    raw_text = await _call_nrp([
                        {"role": "system", "content": system_persona},
                        {"role": "user", "content": prompt_text}
                    ])
                    save_debug_log(request_id, 'response', raw_text, attempt, 'standard')

            if raw_text:
                full_raw_text += f"\n--- Attempt {attempt} ---\n{raw_text}\n"
                parsed_step = parse_veracity_toon(raw_text)
                json_data = extract_json_from_text(raw_text)
                if json_data:
                    for k in['veracity_vectors', 'modalities', 'video_context_summary', 'final_assessment', 'factuality_factors', 'disinformation_analysis', 'tags']:
                        if k in json_data:
                            if isinstance(parsed_step.get(k), dict) and isinstance(json_data[k], dict):
                                parsed_step[k].update(json_data[k])
                            else:
                                parsed_step[k] = json_data[k]
                accumulated_data = smart_merge(accumulated_data, parsed_step)

            missing_fields = validate_parsed_data(accumulated_data, is_text_only)
            if not missing_fields:
                yield "Validation Passed.\n"
                yield {"raw_toon": full_raw_text, "parsed_data": accumulated_data, "prompt_used": prompt_used, "fcot_trace": fcot_trace}
                break

            if attempt == max_retries:
                 yield {"raw_toon": full_raw_text, "parsed_data": accumulated_data, "prompt_used": prompt_used, "fcot_trace": fcot_trace}
                 break

    except Exception as e:
        yield f"ERROR: {e}\n\n"
        logger.error("NRP Labeling Error", exc_info=True)


async def generate_community_summary(comments: list, model_selection: str, active_config: dict) -> str:
    prompt = "Summarize the following user comments to identify any context, consensus on factuality, or claims provided by the community:\n\n"
    for i, c in enumerate(comments[:50]):
        prompt += f"Comment {i+1}: {c.get('text', '')}\n"
    
    prompt += "\nProvide a brief summary focusing on community fact-checking efforts or shared context."
    
    if model_selection == 'nrp':
        try:
            url = f"{active_config.get('base_url', 'https://api.openai.com/v1').rstrip('/')}/chat/completions"
            headers = {"Authorization": f"Bearer {active_config.get('api_key')}", "Content-Type": "application/json"}
            payload = {
                "model": active_config.get('model_name', 'gpt-4'),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            loop = asyncio.get_event_loop()
            def do_req():
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code != 200:
                    raise Exception(f"API Error {resp.status_code}: {resp.text}")
                return resp.json()["choices"][0]["message"]["content"]
            return await loop.run_in_executor(None, do_req)
        except Exception as e:
            return f"Error generating summary: {e}"
            
    elif model_selection == 'vertex':
        if genai is None: return "Vertex SDK missing."
        try:
            client_local = genai.Client(vertexai=True, project=active_config.get('project_id'), location=active_config.get('location', 'us-central1'), api_key=active_config.get('api_key') if active_config.get('api_key') else None)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: client_local.models.generate_content(model=active_config.get('model_name', 'gemini-1.5-pro'), contents=prompt))
            return response.text
        except Exception as e:
            return f"Error generating summary: {e}"
    else:
        if genai_legacy is None: return "Gemini SDK missing."
        try:
            genai_legacy.configure(api_key=active_config.get('api_key'))
            model = genai_legacy.GenerativeModel(active_config.get('model_name', 'gemini-1.5-pro'))
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
            return response.text
        except Exception as e:
            return f"Error generating summary: {e}"