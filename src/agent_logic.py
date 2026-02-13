import logging
import os
import asyncio
import nest_asyncio
import hashlib
import datetime
from typing import Any, Dict
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

# Import from common_utils instead of app to avoid circular import
import common_utils
import inference_logic

logger = logging.getLogger(__name__)

# Apply nested asyncio safely
try:
    nest_asyncio.apply()
except ValueError as e:
    logger.warning(f"Could not apply nest_asyncio (likely using uvloop): {e}. Proceeding without patching.")

# --- Tool Definition ---

def analyze_video_veracity(video_url: str, specific_question: str = "") -> dict:
    """
    Analyzes a video for veracity, deepfakes, and logical consistency.
    
    Args:
        video_url: The URL of the video (Twitter/X, YouTube, etc.).
        specific_question: Optional specific question to answer about the video.
    
    Returns:
        A dictionary containing veracity scores, reasoning, and classification.
    """
    # Helper wrapper to run async logic synchronously for the tool if needed
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If we are already in a loop, we create a task and wait if possible,
        # or rely on the fact that Google ADK might handle coroutines natively in future versions.
        # For now, we return a coroutine if the caller expects it, but ADK tools usually expect sync returns
        # or we use nest_asyncio. Since nest_asyncio failed, we must be careful.
        
        # If the ADK executor is async-aware, we should define this function with `async def`.
        # However, standard ADK Tool signatures are often synchronous.
        # We will attempt to return the coroutine for the executor to await if it supports it.
        # If not, this might raise a runtime warning/error about awaiting inside a sync function.
        
        # NOTE: A2A/ADK integration is evolving. We will try to run it.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _analyze_video_async(video_url, specific_question))
            return future.result()
    else:
        return asyncio.run(_analyze_video_async(video_url, specific_question))

async def _analyze_video_async(video_url: str, context: str) -> dict:
    try:
        # Generate ID
        request_id = hashlib.md5(f"{video_url}_{datetime.datetime.now()}".encode()).hexdigest()[:10]
        logger.info(f"[A2A Tool] Analyzing: {video_url} (ID: {request_id})")

        # 1. Download/Prepare using common_utils
        assets = await common_utils.prepare_video_assets(video_url, request_id)
        if not assets.get("video") and not assets.get("caption"):
            return {"error": "Could not fetch content or caption from URL."}

        # 2. Config
        # In a real scenario, ensure these env vars are set
        config = {
            "project_id": os.getenv("VERTEX_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "")),
            "location": os.getenv("VERTEX_LOCATION", "us-central1"),
            "model_name": "gemini-1.5-pro-preview-0409", 
            "max_retries": 1
        }
        
        # 3. Transcript
        trans = common_utils.parse_vtt(assets['transcript']) if assets.get('transcript') else "No transcript available."
        
        system_persona = (
            "You are the LiarMP4 Agent called via A2A Protocol. "
            "Analyze the content strictly. "
        )
        if context:
            system_persona += f" Context/Focus: {context}"

        # 4. Run Inference
        final_result = None
        # Consume the stream
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
            return {
                "veracity_score": final_result.get("final_assessment", {}).get("veracity_score_total"),
                "classification": final_result.get("disinformation_analysis", {}).get("classification"),
                "summary": final_result.get("video_context_summary"),
                "reasoning": final_result.get("final_assessment", {}).get("reasoning"),
                "vectors": final_result.get("veracity_vectors"),
                "tags": final_result.get("tags")
            }
        return {"error": "Inference pipeline finished but yielded no data."}

    except Exception as e:
        logger.error(f"[A2A Tool Error] {e}")
        return {"error": str(e)}

# --- Agent & App Construction ---

def create_a2a_app():
    """
    Builds and returns the A2A Starlette Application.
    """
    
    # 1. Define the ADK Agent
    model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-pro-preview-0409")
    
    liar_agent = Agent(
        name="liarmp4_verifier",
        model=model_name,
        instruction="""
        You are the LiarMP4 Video Verification Agent.
        Your purpose is to analyze social media videos for misinformation, deepfakes, and logical inconsistencies.
        
        You have a specialized tool `analyze_video_veracity`.
        When asked to analyze a video URL, you MUST use this tool.
        Do not try to hallucinate the video content.
        
        Return the analysis provided by the tool in a structured summary.
        """,
        tools=[analyze_video_veracity]
    )

    # 2. Define the A2A Agent Card (Metadata)
    base_url = os.getenv("A2A_BASE_URL", "http://localhost:8000/a2a")
    
    agent_card = AgentCard(
        name="LiarMP4 Video Verifier",
        url=base_url,
        description="Analyzes social media videos for veracity and deepfakes using FCoT.",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["application/json", "text/plain"],
        preferred_transport=TransportProtocol.jsonrpc,
        skills=[
            AgentSkill(
                id="analyze_video",
                name="Analyze Video Veracity",
                description="Checks a video URL for misinformation scores and deepfake probability.",
                tags=["verification", "deepfake", "fact-check", "video"],
                examples=[
                    "Analyze this video: https://twitter.com/user/status/123",
                    "Is this video real? https://youtube.com/watch?v=xyz",
                    "Check the veracity of this post"
                ]
            )
        ]
    )

    # 3. Create the Server/App
    runner = Runner(
        app_name=liar_agent.name,
        agent=liar_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )

    config = A2aAgentExecutorConfig()
    executor = A2aAgentExecutor(runner=runner, config=config)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )

    return a2a_app.build()
