import os
import csv
import json
import logging
import asyncio
from pathlib import Path
import inference_logic

# Configure Logging
logger = logging.getLogger(__name__)

# --- Prompts for User Analysis ---

PROMPT_USER_PROFILING = """
You are an Expert Intelligence Analyst specializing in Information Integrity and Social Influence Operations.

**TASK:**
Analyze the following timeline of social media posts from a single user: "@{username}".
Your goal is to construct a "Credibility & Bias Profile" based on their historical behavior.

**INPUT DATA (Recent Posts):**
{timeline_text}

**ANALYSIS REQUIREMENTS:**
1.  **Thematic Clusters:** What subjects does this user repeatedly post about? (e.g., "Crypto", "US Politics", "Climate Skepticism").
2.  **Echo Chamber Indicators:** Does the user frequently repost specific domains or engage with specific narratives without adding nuance?
3.  **Emotional Valence:** Analyze the aggregate emotional tone (Alarmist, Neutral, Aggressive, Satirical).
4.  **Bias Detection:** Identify explicit political or ideological biases based on the text.
5.  **Credibility Weighting:** Based on the content, assign a "Historical Credibility Score" (0.0 to 1.0).
    *   0.0 = High frequency of inflammatory/unverified claims.
    *   1.0 = Consistently neutral or verified sourcing.

**OUTPUT FORMAT (Strict JSON):**
{{
  "username": "@{username}",
  "thematic_clusters": ["Topic A", "Topic B"],
  "echo_chamber_detected": boolean,
  "bias_assessment": "Description of bias...",
  "emotional_valence": "Dominant tone...",
  "credibility_score": float,
  "summary_profile": "A concise paragraph summarizing the user's role in the information ecosystem."
}}
"""

async def load_user_history(username: str, limit: int = 50) -> str:
    """
    Reads the user's history.csv and formats it into a text block for the LLM.
    """
    csv_path = Path(f"data/profiles/{username}/history.csv")
    if not csv_path.exists():
        return ""

    timeline_entries = []
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            # Read all, sort by date descending if needed, but scraper usually does desc
            rows = list(reader)
            # Take latest 'limit' posts
            recent_rows = rows[-limit:] 
            
            for row in recent_rows:
                entry = (
                    f"[{row['timestamp']}] "
                    f"{'REPOST' if row.get('is_reply')=='True' else 'POST'}: "
                    f"\"{row['text']}\" "
                    f"(Likes: {row['metric_likes']}, Views: {row['metric_views']})"
                )
                timeline_entries.append(entry)
    except Exception as e:
        logger.error(f"Error reading history for {username}: {e}")
        return ""

    return "\n".join(timeline_entries)

async def generate_user_profile_report(username: str):
    """
    Orchestrates the analysis pipeline:
    1. Load History.
    2. Construct Prompt.
    3. Call LLM (using Vertex/Gemini config from environment or default).
    4. Save JSON Report.
    """
    logger.info(f"Starting analysis for user: {username}")
    
    timeline_text = await load_user_history(username)
    if not timeline_text:
        return {"error": "No history found or empty timeline."}

    # Format Prompt
    prompt = PROMPT_USER_PROFILING.format(username=username, timeline_text=timeline_text)

    # Use Vertex AI by default if configured, else try Gemini Legacy
    # For now, we reuse the pipeline functions in inference_logic if available, 
    # or create a direct call here for simplicity.
    
    # We'll assume Vertex is the primary backend for this advanced analysis
    # This requires valid credentials in the environment or passed config.
    # Fallback to a placeholder if no model is loaded.
    
    report_json = {}
    
    try:
        # Attempt to use the existing Vertex Client in inference_logic if initialized
        # Otherwise, we instantiate a quick one if env vars exist
        project_id = os.getenv("VERTEX_PROJECT_ID")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        
        if inference_logic.genai and project_id:
            from google.genai import Client
            from google.genai.types import GenerateContentConfig
            
            client = Client(vertexai=True, project=project_id, location=location)
            response = client.models.generate_content(
                model="gemini-1.5-pro-preview-0409",
                contents=prompt,
                config=GenerateContentConfig(response_mime_type="application/json")
            )
            report_text = response.text
            report_json = json.loads(report_text)
            
        else:
            # Fallback Mock for Demo/LITE mode
            logger.warning("Vertex AI credentials not found. Generating Mock Analysis.")
            report_json = {
                "username": f"@{username}",
                "thematic_clusters": ["Simulated Topic 1", "Simulated Topic 2"],
                "bias_assessment": "System running in LITE mode. Configure Vertex AI for real analysis.",
                "credibility_score": 0.5,
                "summary_profile": "Mock profile generated because AI backend is not active."
            }

    except Exception as e:
        logger.error(f"LLM Analysis failed: {e}")
        report_json = {"error": str(e)}

    # Save Report
    output_path = Path(f"data/profiles/{username}/analysis_report.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2)

    return report_json