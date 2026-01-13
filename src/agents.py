from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent, Gemini
from google.genai import types
import logging

logger = logging.getLogger(__name__)

# --- Retry Configuration ---
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

# --- Individual Agents ---

# Context Summary Agent: Provides a neutral summary of the video content.
context_summary_agent = Agent(
    name="ContextSummaryAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    <thinking>
    1. Scan the video, caption, and transcript for key entities and events.
    2. Identify the main narrative thread or objective of the content.
    3. Synthesize the findings into a neutral, objective summary.
    </thinking>
    Output format: summary: text[1]{text}: "Summary text" """,
)

# Political Bias Agent: Identifies political leaning and quantifies bias.
political_bias_agent = Agent(
    name="PoliticalBiasAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    <thinking>
    1. Examine the language used for loaded terms or rhetorical devices.
    2. Analyze the context of mentioned political figures or institutions.
    3. Determine the leaning (Left/Right/Center) and evaluate the intensity of bias.
    </thinking>
    Output format: political_bias: details[1]{score,reasoning}: (Int),"Reasoning" """,
)

# Criticism Level Agent: Measures the degree of hostility or support in the tone.
criticism_level_agent = Agent(
    name="CriticismLevelAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    <thinking>
    1. Assess the emotional valence of the speaker and visual cues.
    2. Identify instances of direct criticism, sarcasm, or praise.
    3. Quantify the overall hostility level on a neutral-to-supportive scale.
    </thinking>
    Output format: criticism_level: details[1]{score,reasoning}: (Int),"Reasoning" """,
)

# Modalities Agent: Evaluates the consistency between video, audio, and text.
modalities_agent = Agent(
    name="ModalitiesAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    <thinking>
    1. Compare visual events with audio descriptions for sync issues or contradictions.
    2. Verify if the user-provided caption accurately reflects the visual content.
    3. Check the transcript against both the audio and the caption for discrepancies.
    </thinking>
    Output format: 
    video_audio_pairing: details[1]{score,reasoning}: (Int),"Reasoning"
    video_caption_pairing: details[1]{score,reasoning}: (Int),"Reasoning"
    audio_caption_pairing: details[1]{score,reasoning}: (Int),"Reasoning" """,
)

# Disinformation Agent: Analyzes potential manipulation and threat levels.
disinformation_agent = Agent(
    name="DisinformationAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    <thinking>
    1. Search for signs of technical manipulation (deepfakes, AI artifacts).
    2. Analyze the intent behind potential misinformation (Political/Commercial).
    3. Evaluate the risk level and the specific threat vector used.
    </thinking>
    Output format: disinformation_analysis: details[1]{level,intent,threat_vector}: (Int),(Intent),(Vector) """,
)

# Sentiment Bias Agent: Captures the overall emotional tone and inherent bias.
sentiment_bias_agent = Agent(
    name="SentimentBiasAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    <thinking>
    1. Aggregate the emotional signals from the entire video duration.
    2. Identify recurring biased patterns or slanted perspectives.
    3. Synthesize a comprehensive overview of the sentiment and bias.
    </thinking>
    Output format: sentiment_and_bias: text[1]{text}: "Synthesis text" """,
)

# --- Agent Grouping ---

# The ParallelAgent runs all its sub-agents simultaneously.
analysis_team = ParallelAgent(
    name="AnalysisTeam",
    sub_agents=[
        context_summary_agent,
        political_bias_agent,
        criticism_level_agent,
        modalities_agent,
        disinformation_agent,
        sentiment_bias_agent
    ],
)

# This SequentialAgent defines the high-level workflow.
video_analysis_system = SequentialAgent(
    name="VideoAnalysisSystem",
    sub_agents=[analysis_team],
)

if __name__ == "__main__":
    print("✅ Video Analysis Agents created.")
