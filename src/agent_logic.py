import logging
import os
import asyncio
import nest_asyncio
import hashlib
import datetime
import json
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Apply nested asyncio if possible
try:
    nest_asyncio.apply()
except (ValueError, ImportError):
    pass

# Try importing ADK, set flag if missing
ADK_AVAILABLE = False
try:
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TransportProtocol
    from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor, A2aAgentExecutorConfig
    from google.adk.agents import Agent
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    ADK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Google ADK/A2A Libraries not found: {e}. Switching to MOCK AGENT.")
    ADK_AVAILABLE = False

import common_utils
import inference_logic

# --- Tool Definition ---

def analyze_video_veracity(video_url: str, specific_question: str = "") -> dict:
    """Tool to analyze video veracity."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _analyze_video_async(video_url, specific_question)).result()
    else:
        return asyncio.run(_analyze_video_async(video_url, specific_question))

async def _analyze_video_async(video_url: str, context: str) -> dict:
    try:
        request_id = hashlib.md5(f"{video_url}_{datetime.datetime.now()}".encode()).hexdigest()[:10]
        assets = await common_utils.prepare_video_assets(video_url, request_id)
        
        # Env Config
        config = {
            "project_id": os.getenv("VERTEX_PROJECT_ID"),
            "location": os.getenv("VERTEX_LOCATION", "us-central1"),
            "model_name": os.getenv("VERTEX_MODEL_NAME", "gemini-1.5-pro-preview-0409"), 
            "max_retries": 1
        }
        
        system_persona = f"You are the LiarMP4 Verifier. Context: {context}"
        trans = common_utils.parse_vtt(assets['transcript']) if assets.get('transcript') else "No transcript."

        final_result = None
        # Use Vertex pipeline if configured, otherwise this tool will fail gracefully in the pipeline
        if config["project_id"]:
            async for chunk in inference_logic.run_vertex_labeling_pipeline(
                video_path=assets.get('video'),
                caption=assets.get('caption', ''),
                transcript=trans,
                vertex_config=config,
                include_comments=False,
                reasoning_method="cot", 
                system_persona=system_persona,
                request_id=request_id
            ):
                if isinstance(chunk, dict) and "parsed_data" in chunk:
                    final_result = chunk["parsed_data"]
        
        if final_result:
            return final_result.get("final_assessment", {})
        
        return {"error": "Inference yielded no data or credentials missing."}

    except Exception as e:
        logger.error(f"[Tool Error] {e}")
        return {"error": str(e)}

# --- Mock Fallback ---

def create_mock_a2a_app():
    """Creates a basic Starlette/FastAPI app that mimics A2A JSON-RPC behavior."""
    from fastapi import FastAPI, Request
    
    mock_app = FastAPI()

    @mock_app.post("/")
    @mock_app.post("/jsonrpc")
    async def mock_rpc(request: Request):
        try:
            data = await request.json()
            # Standard JSON-RPC Response
            return {
                "jsonrpc": "2.0",
                "id": data.get("id", 1),
                "result": {
                    "text": "⚠️ AGENT OFFLINE (MOCK MODE). The Google ADK libraries are missing or failed to load. Check server logs.",
                    "data": {"status": "mock"}
                }
            }
        except Exception as e:
            return {"error": {"code": -32700, "message": "Parse error"}}
            
    return mock_app

# --- Factory ---

def create_a2a_app():
    """
    Builds the A2A Application. Returns Mock App if ADK missing.
    """
    if not ADK_AVAILABLE:
        logger.warning("⚠️ Google ADK not found (or import failed). Returning Mock A2A App to ensure routing works.")
        return create_mock_a2a_app()

    try:
        model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-pro-preview-0409")
        
        liar_agent = Agent(
            name="liarmp4_verifier",
            model=model_name,
            instruction="You are the LiarMP4 Video Verification Agent.",
            tools=[analyze_video_veracity]
        )

        base_url = os.getenv("A2A_BASE_URL", "/a2a")
        
        agent_card = AgentCard(
            name="LiarMP4 Agent",
            url=base_url,
            description="Video Veracity Verifier",
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=True),
            preferred_transport=TransportProtocol.jsonrpc,
            default_input_modes=["text/plain"],
            default_output_modes=["application/json"],
            skills=[AgentSkill(id="analyze", name="Analyze", description="Analyze Video", tags=["video"])]
        )

        runner = Runner(
            app_name="liarmp4",
            agent=liar_agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

        executor = A2aAgentExecutor(runner=runner, config=A2aAgentExecutorConfig())

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        a2a_app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        
        logger.info("✅ A2A ADK Agent created successfully.")
        return a2a_app.build()

    except Exception as e:
        logger.critical(f"❌ Failed to build ADK Agent: {e}. Falling back to Mock.", exc_info=True)
        return create_mock_a2a_app()
