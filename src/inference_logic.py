import torch
import re
import ast
import sys
import os
import logging
import asyncio
import json
import datetime
import requests
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from my_vision_process import process_vision_info, client
from labeling_logic import (
    LABELING_PROMPT_TEMPLATE, SCORE_INSTRUCTIONS_SIMPLE, SCORE_INSTRUCTIONS_REASONING,
    SCHEMA_SIMPLE, SCHEMA_REASONING,
    FCOT_MACRO_PROMPT, FCOT_MESO_PROMPT, FCOT_SYNTHESIS_PROMPT, TEXT_ONLY_INSTRUCTIONS,
    get_formatted_tag_list
)
from toon_parser import parse_veracity_toon

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
    global LITE_MODE, processor, base_model, peft_model, active_model
    
    if LITE_MODE:
        logger.info("LITE_MODE is enabled. Skipping local model loading.")
        return
    
    if base_model is not None: return
    
    if not torch.cuda.is_available():
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
        {"role": "user", "content":[
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

async def generate_simple_text(prompt: str, model_type: str, config: dict):
    loop = asyncio.get_event_loop()
    try:
        if model_type == 'gemini':
            if genai_legacy is None: return "Error: Legacy SDK missing."
            genai_legacy.configure(api_key=config.get("api_key"))
            model = genai_legacy.GenerativeModel(config.get("model_name", "models/gemini-2.0-flash-exp"))
            response = await loop.run_in_executor(
                None, 
                lambda: model.generate_content(prompt, generation_config={"temperature": 0.0})
            )
            return response.text
            
        elif model_type == 'vertex':
            if genai is None: return "Error: Vertex SDK missing."
            api_key = config.get("api_key")
            if api_key:
                cl = genai.Client(vertexai=True, project=config['project_id'], location=config['location'], api_key=api_key)
            else:
                cl = genai.Client(vertexai=True, project=config['project_id'], location=config['location'])
            response = await loop.run_in_executor(
                None,
                lambda: cl.models.generate_content(
                    model=config['model_name'],
                    contents=prompt,
                    config=GenerateContentConfig(temperature=0.0)
                )
            )
            return response.text

        elif model_type == 'nrp':
            api_key = config.get("api_key")
            model_name = config.get("model_name", "gpt-4")
            base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
            if not api_key: return "Error: NRP API key missing."
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
            def do_request():
                resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=600)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                return f"Error: {resp.status_code} {resp.text}"
            return await loop.run_in_executor(None, do_request)

    except Exception as e:
        logger.error(f"Text Gen Error: {e}")
        return f"Error generating text: {e}"

async def generate_community_summary(comments: list, model_type: str, config: dict):
    if not comments: return "No comments available."
    c_text = "\n".join([f"- {c.get('author', 'User')}: {c.get('text', '')}" for c in comments[:15]])
    prompt = (
        "You are a Community Context Analyst. Analyze the following user comments regarding a social media post.\n"
        "Your goal is to extract 'Community Notes' - specifically looking for fact-checking, debunking, or additional context provided by users.\n"
        f"COMMENTS:\n{c_text}\n\n"
        "OUTPUT:\n"
        "Provide a concise 1-paragraph summary of the community consensus regarding the veracity of the post."
    )
    return await generate_simple_text(prompt, model_type, config)

def extract_json_from_text(text):
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return {}

def validate_parsed_data(data, is_text_only):
    missing =[]
    
    if not data.get('video_context_summary'): missing.append("summary")
    
    final = data.get('final_assessment', {})
    if not final.get('reasoning') or len(str(final.get('reasoning', ''))) < 5: missing.append("final:reasoning")
    
    vectors = data.get('veracity_vectors', {})
    required_vectors =['visual_integrity_score', 'audio_integrity_score', 'source_credibility_score', 'logical_consistency_score', 'emotional_manipulation_score']
    for k in required_vectors:
        if k in ['visual_integrity_score', 'audio_integrity_score'] and is_text_only: continue
        v = vectors.get(k)
        if not v or str(v) == '0' or str(v).lower() == 'n/a': missing.append(f"vector:{k}")

    mod = data.get('modalities', {})
    for k in['video_audio_score', 'video_caption_score', 'audio_caption_score']:
        if k in['video_audio_score', 'video_caption_score'] and is_text_only: continue
        v = mod.get(k)
        if not v or str(v) == '0' or str(v).lower() == 'n/a': missing.append(f"modality:{k}")

    fact = data.get('factuality_factors', {})
    if not fact.get('claim_accuracy'): missing.append("factuality:claim_accuracy")

    disinfo = data.get('disinformation_analysis', {})
    if not disinfo.get('classification'): missing.append("disinfo:classification")

    return missing

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

async def run_gemini_labeling_pipeline(video_path: str, caption: str, transcript: str, gemini_config: dict, include_comments: bool, reasoning_method: str = "cot", system_persona: str = "", request_id: str = None):
    if genai_legacy is None:
        yield "ERROR: Legacy SDK missing.\n"
        return
    api_key = gemini_config.get("api_key")
    if not api_key: return
    max_retries = int(gemini_config.get("max_retries", 1))
    
    try:
        genai_legacy.configure(api_key=api_key)
        loop = asyncio.get_event_loop()
        uploaded_file = None
        is_text_only = False
        if video_path and os.path.exists(video_path):
             uploaded_file = await loop.run_in_executor(None, lambda: genai_legacy.upload_file(path=video_path))
             while uploaded_file.state.name == "PROCESSING": await asyncio.sleep(2)
        else: is_text_only = True
        
        active_tools =[]
        if gemini_config.get("use_search", False):
            active_tools.append({"google_search_retrieval": {}})
        if gemini_config.get("use_code", False):
            active_tools.append({"code_execution": {}})

        model = genai_legacy.GenerativeModel("models/gemini-2.0-flash-exp", tools=active_tools if active_tools else None)
        toon_schema = SCHEMA_REASONING if include_comments else SCHEMA_SIMPLE
        score_instructions = SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE
        tag_list_text = get_formatted_tag_list()
        
        accumulated_data = {}
        prompt_used = ""
        fcot_trace = {}
        full_raw_text = ""
        if is_text_only: system_persona += "\n" + TEXT_ONLY_INSTRUCTIONS

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
                    yield "Starting Fractal Chain of Thought (Gemini FCoT)..."
                    chat = model.start_chat(history=[])
                    
                    macro_prompt = FCOT_MACRO_PROMPT.format(system_persona=system_persona, caption=caption, transcript=transcript)
                    save_debug_log(request_id, 'prompt', macro_prompt, attempt, 'fcot_macro')
                    inputs1 = [macro_prompt]
                    if uploaded_file: inputs1.insert(0, uploaded_file)
                    res1 = await loop.run_in_executor(None, lambda: chat.send_message(inputs1))
                    macro_hypothesis = res1.text
                    save_debug_log(request_id, 'response', macro_hypothesis, attempt, 'fcot_macro')
                    fcot_trace['macro'] = macro_hypothesis

                    meso_prompt = FCOT_MESO_PROMPT.format(macro_hypothesis=macro_hypothesis)
                    save_debug_log(request_id, 'prompt', meso_prompt, attempt, 'fcot_meso')
                    res2 = await loop.run_in_executor(None, lambda: chat.send_message(meso_prompt))
                    micro_observations = res2.text
                    save_debug_log(request_id, 'response', micro_observations, attempt, 'fcot_meso')
                    fcot_trace['meso'] = micro_observations

                    synthesis_prompt = FCOT_SYNTHESIS_PROMPT.format(toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=tag_list_text)
                    save_debug_log(request_id, 'prompt', synthesis_prompt, attempt, 'fcot_synthesis')
                    res3 = await loop.run_in_executor(None, lambda: chat.send_message(synthesis_prompt))
                    raw_text = res3.text
                    save_debug_log(request_id, 'response', raw_text, attempt, 'fcot_synthesis')
                    prompt_used = f"FCoT Pipeline:\nMacro: {macro_hypothesis}\nMeso: {micro_observations}"
                else:
                    prompt_text = LABELING_PROMPT_TEMPLATE.format(
                        system_persona=system_persona, caption=caption, transcript=transcript,
                        toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=tag_list_text
                    )
                    prompt_used = prompt_text
                    if is_text_only: prompt_text = "NOTE: Text Analysis Only.\n" + prompt_text
                    save_debug_log(request_id, 'prompt', prompt_text, attempt, 'standard')
                    inputs = [prompt_text]
                    if uploaded_file: inputs.append(uploaded_file)
                    response = await loop.run_in_executor(None, lambda: model.generate_content(inputs, generation_config={"temperature": 0.1}))
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

        if uploaded_file: await loop.run_in_executor(None, lambda: genai_legacy.delete_file(name=uploaded_file.name))
    except Exception as e: yield f"ERROR: {e}"

async def run_vertex_labeling_pipeline(video_path: str, caption: str, transcript: str, vertex_config: dict, include_comments: bool, reasoning_method: str = "cot", system_persona: str = "", request_id: str = None):
    if genai is None:
        yield "ERROR: 'google-genai' not installed.\n"
        return

    project_id = vertex_config.get("project_id")
    location = vertex_config.get("location", "us-central1")
    model_name = vertex_config.get("model_name", "gemini-1.5-pro-preview-0409")
    max_retries = int(vertex_config.get("max_retries", 1))
    api_key = vertex_config.get("api_key")

    if not project_id: return

    try:
        # Pass api_key directly if available to use API Keys instead of ADC Service Accounts
        if api_key:
            client = genai.Client(vertexai=True, project=project_id, location=location, api_key=api_key)
        else:
            client = genai.Client(vertexai=True, project=project_id, location=location)

        video_part = None
        is_text_only = False
        if video_path and os.path.exists(video_path):
            with open(video_path, 'rb') as f: video_bytes = f.read()
            video_part = Part.from_bytes(data=video_bytes, mime_type="video/mp4")
        else: is_text_only = True

        active_tools =[]
        if vertex_config.get("use_search", True):
            active_tools.append(Tool(google_search=GoogleSearch()))
        if vertex_config.get("use_code", False):
            try:
                from google.genai.types import CodeExecution
                active_tools.append(Tool(code_execution=CodeExecution()))
            except ImportError:
                pass

        config = GenerateContentConfig(
            temperature=0.1, response_mime_type="text/plain", max_output_tokens=8192,
            tools=active_tools if active_tools else None
        )

        toon_schema = SCHEMA_REASONING if include_comments else SCHEMA_SIMPLE
        score_instructions = SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE
        tag_list_text = get_formatted_tag_list()
        
        accumulated_data = {}
        prompt_used = ""
        fcot_trace = {}
        full_raw_text = ""
        loop = asyncio.get_event_loop()
        
        if is_text_only: system_persona += "\n" + TEXT_ONLY_INSTRUCTIONS

        for attempt in range(max_retries + 1):
            raw_text = ""
            if attempt > 0:
                missing = validate_parsed_data(accumulated_data, is_text_only)
                yield f"Validation failed. Missing or incomplete fields: {missing}. Initiating Iterative Reprompt (Attempt {attempt}/{max_retries}) to acquire remaining factuality components...\n"
                
                # REPROMPT CONSTRUCTION
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
                contents = [prompt_text]
                if video_part: contents.insert(0, video_part)
                
                response = await loop.run_in_executor(None, lambda: client.models.generate_content(model=model_name, contents=contents, config=config))
                raw_text = response.text
                save_debug_log(request_id, 'response', raw_text, attempt, 'reprompt')
            else:
                if reasoning_method == "fcot":
                    yield "Starting Fractal Chain of Thought (Vertex FCoT)..."
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

                    meso_prompt = FCOT_MESO_PROMPT.format(macro_hypothesis=macro_hypothesis)
                    save_debug_log(request_id, 'prompt', meso_prompt, attempt, 'fcot_meso')
                    res2 = await loop.run_in_executor(None, lambda: chat.send_message(meso_prompt))
                    micro_observations = res2.text
                    save_debug_log(request_id, 'response', micro_observations, attempt, 'fcot_meso')
                    fcot_trace['meso'] = micro_observations
                    
                    synthesis_prompt = FCOT_SYNTHESIS_PROMPT.format(toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=tag_list_text)
                    save_debug_log(request_id, 'prompt', synthesis_prompt, attempt, 'fcot_synthesis')
                    res3 = await loop.run_in_executor(None, lambda: chat.send_message(synthesis_prompt))
                    raw_text = res3.text
                    save_debug_log(request_id, 'response', raw_text, attempt, 'fcot_synthesis')
                    prompt_used = f"FCoT (Vertex):\nMacro: {macro_hypothesis}\nMeso: {micro_observations}"
                else:
                    prompt_text = LABELING_PROMPT_TEMPLATE.format(
                        system_persona=system_persona, caption=caption, transcript=transcript,
                        toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=tag_list_text
                    )
                    contents = []
                    if video_part: contents =[video_part, prompt_text]
                    else: contents =[f"NOTE: Text Only Analysis (No Video).\n{prompt_text}"]
                    prompt_used = prompt_text
                    save_debug_log(request_id, 'prompt', prompt_text, attempt, 'standard')
                    yield "Generating Labels (Vertex CoT)..."
                    response = await loop.run_in_executor(None, lambda: client.models.generate_content(model=model_name, contents=contents, config=config))
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

    except Exception as e:
        yield f"ERROR: {e}"
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
        
        logger.info(f"[{request_id}] NRP API Call ({attempt_label}) - URL: {base_url}/chat/completions")
        logger.info(f"[{request_id}] NRP API Call - Model: {model_name}")
        logger.info(f"[{request_id}] NRP API Call - Messages count: {len(messages)}")

        def do_request():
            start_time = datetime.datetime.now()
            logger.info(f"[{request_id}] Dispatching requests.post (timeout=600s)...")
            
            resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=600)
            
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"[{request_id}] NRP API Response received in {elapsed:.2f}s. Status Code: {resp.status_code}")
            
            if resp.status_code != 200:
                logger.error(f"[{request_id}] API Error {resp.status_code}: {resp.text}")
                raise Exception(f"API Error {resp.status_code}: {resp.text}")
                
            resp_json = resp.json()
            usage = resp_json.get("usage", {})
            logger.info(f"[{request_id}] NRP API Usage: {usage}")
            
            return resp_json["choices"][0]["message"]["content"]
            
        return await loop.run_in_executor(None, do_request)

    try:
        for attempt in range(max_retries + 1):
            raw_text = ""
            if attempt > 0:
                missing = validate_parsed_data(accumulated_data, is_text_only)
                yield f"Validation failed. Missing fields: {missing}. Initiating Reprompt (Attempt {attempt}/{max_retries})...\n"
                
                prompt_text = (
                     f"SYSTEM: Review the previous attempt which failed validation.\n"
                     f"CONTEXT: Caption: \"{caption}\"\nTranscript: \"{transcript}\"\n"
                     f"PREVIOUS (PARTIAL) DATA: {json.dumps(accumulated_data, indent=2)}\n"
                     f"MISSING FIELDS: {missing}\n"
                     f"INSTRUCTION: Generate the missing fields to complete the schema. You MUST provide the missing scores for {missing}.\n"
                     f"Output the FULL VALID TOON OBJECT containing all required fields.\n"
                     f"{toon_schema}"
                )
                
                save_debug_log(request_id, 'prompt', prompt_text, attempt, 'reprompt')
                
                yield f"  - Sending Reprompt request to NRP API (Model: {model_name}, Timeout: 600s)...\n"
                raw_text = await _call_nrp([
                    {"role": "system", "content": system_persona},
                    {"role": "user", "content": prompt_text}
                ], attempt_label=f"reprompt_{attempt}")
                yield f"  - Received Reprompt response from NRP API.\n\n"
                
                save_debug_log(request_id, 'response', raw_text, attempt, 'reprompt')
            else:
                if reasoning_method == "fcot":
                    yield "Starting Fractal Chain of Thought (NRP FCoT)...\n"
                    
                    macro_prompt = FCOT_MACRO_PROMPT.format(system_persona=system_persona, caption=caption, transcript=transcript)
                    macro_prompt = "NOTE: Text Only Analysis.\n" + macro_prompt
                    save_debug_log(request_id, 'prompt', macro_prompt, attempt, 'fcot_macro')
                    
                    macro_messages =[{"role": "system", "content": system_persona}, {"role": "user", "content": macro_prompt}]
                    yield f"  - Stage 1: Sending Macro Hypothesis request to NRP API (Timeout: 600s)...\n"
                    macro_hypothesis = await _call_nrp(macro_messages, attempt_label="fcot_macro")
                    yield f"  - Stage 1: Received Macro Hypothesis response.\n"
                    
                    save_debug_log(request_id, 'response', macro_hypothesis, attempt, 'fcot_macro')
                    fcot_trace['macro'] = macro_hypothesis

                    meso_prompt = FCOT_MESO_PROMPT.format(macro_hypothesis=macro_hypothesis)
                    save_debug_log(request_id, 'prompt', meso_prompt, attempt, 'fcot_meso')
                    meso_messages = macro_messages +[{"role": "assistant", "content": macro_hypothesis}, {"role": "user", "content": meso_prompt}]
                    
                    yield f"  - Stage 2: Sending Meso Analysis request to NRP API (Timeout: 600s)...\n"
                    micro_observations = await _call_nrp(meso_messages, attempt_label="fcot_meso")
                    yield f"  - Stage 2: Received Meso Analysis response.\n"
                    
                    save_debug_log(request_id, 'response', micro_observations, attempt, 'fcot_meso')
                    fcot_trace['meso'] = micro_observations

                    synthesis_prompt = FCOT_SYNTHESIS_PROMPT.format(toon_schema=toon_schema, score_instructions=score_instructions, tag_list_text=tag_list_text)
                    save_debug_log(request_id, 'prompt', synthesis_prompt, attempt, 'fcot_synthesis')
                    synthesis_messages = meso_messages +[{"role": "assistant", "content": micro_observations}, {"role": "user", "content": synthesis_prompt}]
                    
                    yield f"  - Stage 3: Sending Synthesis/Formatting request to NRP API (Timeout: 600s)...\n"
                    raw_text = await _call_nrp(synthesis_messages, attempt_label="fcot_synthesis")
                    yield f"  - Stage 3: Received Synthesis response.\n\n"
                    
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
                    yield f"  - Sending Standard request to NRP API (Model: {model_name}, Timeout: 600s)...\n"
                    
                    raw_text = await _call_nrp([
                        {"role": "system", "content": system_persona},
                        {"role": "user", "content": prompt_text}
                    ], attempt_label="standard_cot")
                    
                    yield f"  - Received response from NRP API.\n\n"
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

    except Exception as e:
        yield f"ERROR: {e}\n\n"
        logger.error("NRP Labeling Error", exc_info=True)

