# audio_agent.py
from __future__ import annotations

import asyncio
import os
from typing import Optional

from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# --- 1) Output schema (matches your screenshot “output_schema”) ---

class AudioFactualityOutput(BaseModel):
    score: int = Field(description="1-10 likelihood of audio manipulation/synthesis. 1=authentic, 10=highly manipulated.")
    justification: str = Field(description="1-3 sentence justification. Mention uncertainty if evidence is insufficient.")


# --- 2) LlmAgent identity + instruction (matches your screenshot) ---

AUDIO_INSTRUCTION = """\
You are an Audio Forensics Agent for social-media videos.

Goal:
Score the likelihood that the AUDIO is synthesized/manipulated/spliced/voice-cloned.

Input you will receive:
- transcript: ASR/VTT text extracted from the video.

Output requirements:
- Respond ONLY with JSON that conforms to the output schema.
- No markdown, no code fences, no extra keys.

Scoring rubric:
- 1-3: likely authentic
- 4-6: uncertain / mixed evidence
- 7-10: likely manipulated (robotic prosody, unnatural phrasing, inconsistent cues, etc.)
If evidence is insufficient, choose 4-6 and explicitly say uncertainty.
"""

audio_agent = LlmAgent(
    model=os.getenv("AUDIO_AGENT_MODEL", "gemini-2.5-flash"),
    name="audio_factuality_agent",
    description="Scores audio manipulation likelihood from transcript and returns structured JSON.",
    instruction=AUDIO_INSTRUCTION,
    # matches screenshot “Fine-Tuning LLM Generation (generate_content_config)”
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=250,
    ),
    # matches screenshot “Managing Context (include_contents)”
    include_contents="none",
    # matches screenshot “Structuring Data (output_schema, output_key)”
    output_schema=AudioFactualityOutput,
    output_key="audio_factuality",   # saved into session.state["audio_factuality"]
)

# --- 3) Minimal Runner wrapper (so you can call it like a function) ---

_APP_NAME = os.getenv("ADK_APP_NAME", "veracity_app")
_USER_ID = os.getenv("ADK_USER_ID", "local_user")
_SESSION_ID = os.getenv("ADK_SESSION_ID", "audio_factuality_session")

_lock = asyncio.Lock()
_runner: Optional[Runner] = None
_sessions: Optional[InMemorySessionService] = None

async def _get_runner() -> Runner:
    global _runner, _sessions
    async with _lock:
        if _runner and _sessions:
            return _runner
        _sessions = InMemorySessionService()
        await _sessions.create_session(app_name=_APP_NAME, user_id=_USER_ID, session_id=_SESSION_ID)
        _runner = Runner(agent=audio_agent, app_name=_APP_NAME, session_service=_sessions)
        return _runner

async def score_audio_factuality(transcript: str) -> AudioFactualityOutput:
    """
    Call the agent and return a validated AudioFactualityOutput (score + justification).
    """
    runner = await _get_runner()

    # Since we set output_schema/output_key, we can just pass transcript and read session.state
    msg = types.Content(
        role="user",
        parts=[types.Part(text=f"transcript:\n{transcript}\n")]
    )

    async for _event in runner.run_async(user_id=_USER_ID, session_id=_SESSION_ID, new_message=msg):
        pass

    # read structured output from session state
    session = await _sessions.get_session(app_name=_APP_NAME, user_id=_USER_ID, session_id=_SESSION_ID)
    data = session.state.get("audio_factuality")

    # ADK stores text; output_schema enforces it is JSON. Convert to model.
    # Depending on ADK version, `data` may already be dict; handle both.
    if isinstance(data, AudioFactualityOutput):
        return data
    return AudioFactualityOutput.model_validate(data)
