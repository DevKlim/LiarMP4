import logging
import os
import asyncio
import nest_asyncio
import hashlib
import datetime
import json
import re
import csv
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Apply nested asyncio if possible
try:
    nest_asyncio.apply()
except (ValueError, ImportError):
    pass

import common_utils
import inference_logic

# --- Tool Definition & Agent Logic ---

def analyze_video_veracity(video_url: str, specific_question: str = "", agent_config: dict = None) -> dict:
    """Tool to analyze video veracity."""
    if agent_config is None: agent_config = {}
    loop = asyncio.get_event_loop()
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _analyze_video_async(video_url, specific_question, agent_config)).result()
    else:
        return asyncio.run(_analyze_video_async(video_url, specific_question, agent_config))

async def _analyze_video_async(video_url: str, context: str, agent_config: dict) -> dict:
    try:
        use_search = agent_config.get("use_search", False)
        use_code = agent_config.get("use_code", False)
        provider = agent_config.get("provider", "vertex")
        api_key = agent_config.get("api_key", os.getenv("GEMINI_API_KEY", ""))
        project_id = agent_config.get("project_id", os.getenv("VERTEX_PROJECT_ID", ""))
        location = agent_config.get("location", os.getenv("VERTEX_LOCATION", "us-central1"))
        model_name = agent_config.get("model_name", os.getenv("VERTEX_MODEL_NAME", "gemini-1.5-pro-preview-0409"))
        reasoning_method = agent_config.get("reasoning_method", "cot")
        prompt_template = agent_config.get("prompt_template", "standard")
        
        request_id = hashlib.md5(f"{video_url}_{datetime.datetime.now()}".encode()).hexdigest()[:10]
        assets = await common_utils.prepare_video_assets(video_url, request_id)
        
        # We need the prompt instructions
        try:
            from labeling_logic import PROMPT_VARIANTS
            sel_p = PROMPT_VARIANTS.get(prompt_template, PROMPT_VARIANTS['standard'])
            system_persona_txt = sel_p['instruction']
        except Exception:
            system_persona_txt = "You are a Factuality Agent."
            
        system_persona = f"You are the LiarMP4 Verifier. Context: {context}\n\nPersona: {system_persona_txt}"
        
        trans = common_utils.parse_vtt(assets['transcript']) if assets.get('transcript') else "No transcript."

        final_result = None
        raw_toon_text = ""
        pipeline_logs =[]
        
        if provider == "gemini":
            if not api_key:
                return {"error": "Gemini API Key missing. Please provide it in the Inference Config."}
            gemini_config = {"api_key": api_key, "model_name": model_name, "max_retries": 3, "use_search": use_search, "use_code": use_code}
            async for chunk in inference_logic.run_gemini_labeling_pipeline(
                video_path=assets.get('video'),
                caption=assets.get('caption', ''),
                transcript=trans,
                gemini_config=gemini_config,
                include_comments=False,
                reasoning_method=reasoning_method, 
                system_persona=system_persona,
                request_id=request_id
            ):
                if isinstance(chunk, str):
                    pipeline_logs.append(chunk.strip())
                elif isinstance(chunk, dict) and "parsed_data" in chunk:
                    final_result = chunk["parsed_data"]
                    raw_toon_text = chunk.get("raw_toon", "")
        else:
            if not project_id:
                return {"error": "Vertex Project ID missing. Please provide it in the Inference Config."}
            vertex_config = {
                "project_id": project_id,
                "location": location,
                "model_name": model_name,
                "max_retries": 3,
                "use_search": use_search,
                "use_code": use_code,
                "api_key": api_key
            }
            async for chunk in inference_logic.run_vertex_labeling_pipeline(
                video_path=assets.get('video'),
                caption=assets.get('caption', ''),
                transcript=trans,
                vertex_config=vertex_config,
                include_comments=False,
                reasoning_method=reasoning_method, 
                system_persona=system_persona,
                request_id=request_id
            ):
                if isinstance(chunk, str):
                    pipeline_logs.append(chunk.strip())
                elif isinstance(chunk, dict) and "parsed_data" in chunk:
                    final_result = chunk["parsed_data"]
                    raw_toon_text = chunk.get("raw_toon", "")
        
        if final_result:
            # 1. Compare to GT Database
            gt_score = None
            manual_path = Path("data/manual_dataset.csv")
            if manual_path.exists():
                for row in common_utils.robust_read_csv(manual_path):
                    if common_utils.normalize_link(row.get('link', '')) == common_utils.normalize_link(video_url):
                        try: gt_score = float(row.get('final_veracity_score', 0))
                        except: pass
                        break

            # 2. Extract Data
            ai_score_val = final_result.get('final_assessment', {}).get('veracity_score_total', 0)
            try: ai_score = float(ai_score_val)
            except: ai_score = 0
            
            reasoning = final_result.get('final_assessment', {}).get('reasoning', 'No reasoning provided.')

            vec = final_result.get('veracity_vectors', {})
            mod = final_result.get('modalities', {})
            fact = final_result.get('factuality_factors', {})

            reply_text = f"[ANALYSIS COMPLETE]\nVideo: {video_url}\n\n"
            reply_text += "--- AGENT PIPELINE LOGS ---\n"
            reply_text += "\n".join([log for log in pipeline_logs if log]) + "\n\n"

            reply_text += f"Final Veracity Score: {ai_score}/100\n"
            reply_text += f"Reasoning: {reasoning}\n\n"
            
            reply_text += "--- VERACITY VECTORS ---\n"
            reply_text += f"Visual Integrity       : {vec.get('visual_integrity_score', 'N/A')}\n"
            reply_text += f"Audio Integrity        : {vec.get('audio_integrity_score', 'N/A')}\n"
            reply_text += f"Source Credibility     : {vec.get('source_credibility_score', 'N/A')}\n"
            reply_text += f"Logical Consistency    : {vec.get('logical_consistency_score', 'N/A')}\n"
            reply_text += f"Emotional Manipulation : {vec.get('emotional_manipulation_score', 'N/A')}\n\n"
            
            reply_text += "--- MODALITIES ---\n"
            reply_text += f"Video-Audio            : {mod.get('video_audio_score', 'N/A')}\n"
            reply_text += f"Video-Caption          : {mod.get('video_caption_score', 'N/A')}\n"
            reply_text += f"Audio-Caption          : {mod.get('audio_caption_score', 'N/A')}\n"

            reply_text += "\n--- FACTUALITY FACTORS ---\n"
            reply_text += f"Claim Accuracy         : {fact.get('claim_accuracy', 'N/A')}\n"
            reply_text += f"Evidence Gap           : {fact.get('evidence_gap', 'N/A')}\n"
            reply_text += f"Grounding Check        : {fact.get('grounding_check', 'N/A')}\n"

            if gt_score is not None:
                delta = abs(ai_score - gt_score)
                reply_text += f"\n--- GROUND TRUTH COMPARISON ---\n"
                reply_text += f"Verified GT Score      : {gt_score}/100\n"
                reply_text += f"AI Generated Score     : {ai_score}/100\n"
                reply_text += f"Accuracy Delta         : {delta} points\n"
            
            reply_text += "\n--- RAW TOON OUTPUT ---\n"
            reply_text += f"{raw_toon_text}\n\n"

            config_params_str = json.dumps({"agent_active": True, "use_search": use_search, "use_code": use_code})
            
            # 3. Save to Dataset properly to track agent config accuracy
            d_path = Path("data/dataset.csv")
            try:
                with open(d_path, 'a', newline='', encoding='utf-8') as f:
                    row = {
                        "id": request_id, "link": video_url, "timestamp": datetime.datetime.now().isoformat(),
                        "caption": assets.get('caption', ''),
                        "final_veracity_score": ai_score,
                        "visual_score": final_result.get('veracity_vectors', {}).get('visual_integrity_score', 0),
                        "audio_score": final_result.get('veracity_vectors', {}).get('audio_integrity_score', 0),
                        "source_score": final_result.get('veracity_vectors', {}).get('source_credibility_score', 0),
                        "logic_score": final_result.get('veracity_vectors', {}).get('logical_consistency_score', 0),
                        "emotion_score": final_result.get('veracity_vectors', {}).get('emotional_manipulation_score', 0),
                        "align_video_audio": final_result.get('modalities', {}).get('video_audio_score', 0),
                        "align_video_caption": final_result.get('modalities', {}).get('video_caption_score', 0),
                        "align_audio_caption": final_result.get('modalities', {}).get('audio_caption_score', 0),
                        "classification": final_result.get('disinformation_analysis', {}).get('classification', 'None'),
                        "reasoning": reasoning,
                        "tags": ",".join(final_result.get('tags',[])),
                        "raw_toon": raw_toon_text,
                        "config_type": "A2A Agent",
                        "config_model": model_name,
                        "config_prompt": prompt_template,
                        "config_reasoning": reasoning_method,
                        "config_params": config_params_str
                    }
                    writer = csv.DictWriter(f, fieldnames=[
                        "id", "link", "timestamp", "caption", 
                        "final_veracity_score", "visual_score", "audio_score", "source_score", "logic_score", "emotion_score", 
                        "align_video_audio", "align_video_caption", "align_audio_caption",
                        "classification", "reasoning", "tags", "raw_toon",
                        "config_type", "config_model", "config_prompt", "config_reasoning", "config_params"
                    ], extrasaction='ignore')
                    if not d_path.exists() or d_path.stat().st_size == 0: writer.writeheader()
                    writer.writerow(row)
            except Exception as e:
                logger.error(f"Failed writing A2A to dataset: {e}")

            # 4. Save Raw JSON AI-generated file exactly like the ingest queue
            try:
                ts_clean = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                flat_parsed = final_result.copy()
                flat_parsed["raw_toon"] = raw_toon_text
                flat_parsed["meta_info"] = { 
                    "id": request_id, "timestamp": datetime.datetime.now().isoformat(), "link": video_url, 
                    "prompt_used": "A2A Agent Prompt", 
                    "model_selection": provider,
                    "config_type": "GenAI A2A",
                    "config_model": model_name,
                    "config_prompt": prompt_template,
                    "config_reasoning": reasoning_method,
                    "config_params": {"agent_active": True, "use_search": use_search, "use_code": use_code}
                }
                with open(Path(f"data/labels/{request_id}_{ts_clean}.json"), 'w', encoding='utf-8') as f:
                    json.dump(flat_parsed, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed saving A2A raw JSON sidecar: {e}")

            reply_text += f"\n[Pipeline] Successfully parsed context, analyzed factuality, and saved raw AI Label File to Data Manager (Provider: {provider}, Model: {model_name}, Search: {use_search})."

            return {"text": reply_text, "data": final_result}
        
        return {"error": "Inference yielded no data or credentials missing."}

    except Exception as e:
        logger.error(f"[Tool Error] {e}")
        return {"error": str(e)}

# --- Custom A2A App ---
def create_a2a_app():
    """Creates a robust Starlette/FastAPI app that implements core A2A JSON-RPC behavior."""
    from fastapi import FastAPI, Request
    
    a2a_app = FastAPI(title="LiarMP4 A2A Agent")

    @a2a_app.post("/")
    @a2a_app.post("/jsonrpc")
    async def jsonrpc_handler(request: Request):
        try:
            data = await request.json()
            method = data.get("method", "agent.process")
            params = data.get("params", {})
            
            input_text = ""
            agent_config = {}
            if isinstance(params, dict):
                input_text = params.get("input", params.get("text", params.get("query", params.get("prompt", ""))))
                agent_config = params.get("agent_config", {})
                if not input_text and "url" in params:
                    input_text = params["url"]
            elif isinstance(params, list) and len(params) > 0:
                if isinstance(params[0], dict):
                    input_text = params[0].get("text", params[0].get("input", ""))
                else:
                    input_text = str(params[0])
            elif isinstance(params, str):
                input_text = params
            
            # Accept an array of standard agentic invocation methods
            accepted_methods =["agent.process", "agent.generate", "model.generate", "a2a.generate", "a2a.interact", "agent.interact"]
            
            if method in accepted_methods or not method:
                
                # Dynamic Setup & Config Management via Agent Conversation
                update_config = {}
                low_input = str(input_text).lower()
                if "set provider to " in low_input:
                    val = low_input.split("set provider to ")[-1].strip().split()[0]
                    if val in["gemini", "vertex"]: update_config["provider"] = val
                if "set api key to " in low_input:
                    val = input_text.split("set api key to ")[-1].strip().split()[0]
                    update_config["api_key"] = val
                if "set project id to " in low_input:
                    val = input_text.split("set project id to ")[-1].strip().split()[0]
                    update_config["project_id"] = val
                    
                if update_config:
                    return {
                        "jsonrpc": "2.0", "id": data.get("id", 1), 
                        "result": {
                            "text": f"✅ Agent configuration updated automatically ({', '.join(update_config.keys())}). You can now provide a video link or further instructions.",
                            "update_config": update_config
                        }
                    }

                urls = re.findall(r'(https?://[^\s]+)', str(input_text))
                
                if urls:
                    url = urls[0]
                    logger.info(f"Agent Processing Video URL: {url} | Config: {agent_config}")
                    res = await _analyze_video_async(url, str(input_text), agent_config)
                    
                    if "error" in res:
                        reply = f"Error analyzing video: {res['error']}"
                    else:
                        reply = res.get("text", "Processing finished but no reply generated.")
                else:
                    # Agent Setup Guidance Logic
                    provider = agent_config.get("provider", "vertex")
                    api_key = agent_config.get("api_key", "")
                    project_id = agent_config.get("project_id", "")
                    
                    base_capabilities = (
                        "**Agent Capabilities:**\n"
                        "- Process raw video & audio modalities via A2A\n"
                        "- Fetch & analyze comment sentiment and community context\n"
                        "- Run full Factuality pipeline (FCoT) & Generate Veracity Vectors\n"
                        "- Automatically save raw AI Labeled JSON files & sync to Data Manager\n"
                        "- Verify and compare AI outputs against Ground Truth\n"
                        "- Reprompt dynamically for missing scores or incomplete data\n\n"
                        "**Easy Command:**\n"
                        "Use `Run full pipeline on[URL]` to analyze a video, extract all vectors (source, logic, emotion, etc.), and save aligned files."
                    )
                    
                    if provider == 'vertex' and not project_id:
                        reply = f"Welcome to the LiarMP4 Agent Nexus!\n\nIt looks like you haven't configured **Vertex AI** yet. Please enter your Google Cloud Project ID in the 'Inference Config' panel on the left, or tell me directly: *'set project id to [YOUR_PROJECT]'*.\n\n{base_capabilities}"
                    elif provider == 'gemini' and not api_key:
                        reply = f"👋 Welcome to the LiarMP4 Agent Nexus!\n\nIt looks like you haven't configured **Gemini** yet. Please enter your API Key in the 'Inference Config' panel on the left, or tell me directly: *'set api key to[YOUR_KEY]'*.\n\n{base_capabilities}"
                    else:
                        reply = f"✅ I am the LiarMP4 Verifier, fully configured ({provider.capitalize()}) and ready!\n\n{base_capabilities}"

                return {
                    "jsonrpc": "2.0",
                    "id": data.get("id", 1),
                    "result": {
                        "text": reply,
                        "data": {"status": "success", "agent": "LiarMP4_A2A"}
                    }
                }
            else:
                logger.warning(f"A2A Agent rejected unknown method: {method}")
                return {
                    "jsonrpc": "2.0", 
                    "id": data.get("id", 1), 
                    "error": {
                        "code": -32601, 
                        "message": f"Method '{method}' not found. Supported: {', '.join(accepted_methods)}"
                    }
                }
        except Exception as e:
            logger.error(f"A2A Parse Error: {e}")
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            
    logger.info("✅ A2A Custom Agent App created successfully.")
    return a2a_app

