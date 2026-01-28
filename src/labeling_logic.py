# Utilizes TOON (Token-Oriented Object Notation) for token efficiency and structured output.

PROMPT_VARIANTS = {
    "standard": {
        "description": "Balanced Fact-Checker (Default)",
        "instruction": "You are an AI Factuality Assessment Agent operating under the 'Ali Arsanjani Factuality Factors' framework. Your goal is to be neutral and objective."
    },
    "skeptic": {
        "description": "Aggressive Skeptic",
        "instruction": "You are a hardened debunking analyst. Assume content is misleading until proven otherwise. Scrutinize every visual artifact and logical inconsistency harshly."
    },
    "lenient": {
        "description": "Context-Aware/Loose",
        "instruction": "You are a social media context analyst. Prioritize entertainment value and satire. Be lenient on technical imperfections if the narrative intent is clearly humorous."
    }
}

LABELING_PROMPT_TEMPLATE = """
{system_persona}

**INPUT DATA:**
- **User Caption:** "{caption}"
- **Audio Transcript:** "{transcript}"
- **Visuals:** (Provided in video context)

**INSTRUCTIONS:**
1.  **Grounding:** Cross-reference claims in the transcript with your internal knowledge base (and tools if active).
2.  **Chain of Thought (<thinking>):** You MUST think step-by-step inside a `<thinking>` block before generating output.
    *   Analyze *Visual Integrity* (Artifacts, edits).
    *   Analyze *Audio Integrity* (Voice cloning, sync).
    *   Analyze *Modality Alignment* (Does video match audio? Does caption match content? Does audio match caption?).
    *   Analyze *Logic* (Fallacies, gaps).
    *   **Classify Tags:** Identify 3-5 relevant tags (e.g., "political", "celebrity", "targeting", "satire", "news").
    *   Determine *Disinformation* classification.
3.  **Output Format:** Output strictly in **TOON** format (Token-Oriented Object Notation) as defined below.

**CRITICAL CONSTRAINTS:** 
- Do NOT repeat the input data.
- START your response IMMEDIATELY with the `<thinking>` tag.
- **DO NOT use Markdown code blocks.** (Output plain text only).
- Use strict `Key : Type [ Count ] {{ Headers }} :` format followed by data lines.
- Strings containing commas MUST be quoted.
- ALL scores must be filled (use 0 if unsure, do not leave blank).
- **MODALITY SCORING:** You must provide 3 distinct alignment scores: Video-Audio, Video-Caption, and Audio-Caption.

**TOON SCHEMA:**
{toon_schema}

{score_instructions}

**RESPONSE:**
<thinking>
"""

SCORE_INSTRUCTIONS_REASONING = """
**Constraints:** 
1. Provide specific reasoning for EACH score in the `vectors` and `modalities` tables.
2. Ensure strings are properly quoted.
"""

SCORE_INSTRUCTIONS_SIMPLE = """
**Constraint:** Focus on objective measurements. Keep text concise.
"""

# Updated Schema based on user requirements - Ensure explicit newlines
SCHEMA_SIMPLE = """summary: text[1]{text}:
"Brief neutral summary of the video events"

tags: list[1]{keywords}:
"political, celebrity, deepfake, viral"

vectors: scores[1]{visual,audio,source,logic,emotion}:
(Int 1-10),(Int 1-10),(Int 1-10),(Int 1-10),(Int 1-10)
*Scale: 1=Fake/Malicious, 10=Authentic/Neutral*

modalities: scores[1]{video_audio_score,video_caption_score,audio_caption_score}:
(Int 1-10),(Int 1-10),(Int 1-10)
*Scale: 1=Mismatch, 10=Perfect Match*

factuality: factors[1]{accuracy,gap,grounding}:
(Verified/Misleading/False),"Missing evidence description","Grounding check results"

disinfo: analysis[1]{class,intent,threat}:
(None/Misinfo/Disinfo/Satire),(Political/Commercial/None),(Deepfake/Recontextualization/None)

final: assessment[1]{score,reasoning}:
(Int 1-100),"Final synthesis of why this score was given"
"""

SCHEMA_REASONING = """
summary: text[1]{text}:
"Brief neutral summary of the video events"

tags: list[1]{keywords}:
"political, celebrity, deepfake, viral"

vectors: details[5]{category,score,reasoning}:
Visual,(Int 1-10),"Reasoning for visual score"
Audio,(Int 1-10),"Reasoning for audio score"
Source,(Int 1-10),"Reasoning for source credibility"
Logic,(Int 1-10),"Reasoning for logical consistency"
Emotion,(Int 1-10),"Reasoning for emotional manipulation"

modalities: details[3]{category,score,reasoning}:
VideoAudio,(Int 1-10),"Reasoning for video-to-audio alignment"
VideoCaption,(Int 1-10),"Reasoning for video-to-caption alignment"
AudioCaption,(Int 1-10),"Reasoning for audio-to-caption alignment"

factuality: factors[1]{accuracy,gap,grounding}:
(Verified/Misleading/False),"Missing evidence description","Grounding check results"

disinfo: analysis[1]{class,intent,threat}:
(None/Misinfo/Disinfo/Satire),(Political/Commercial/None),(Deepfake/Recontextualization/None)

final: assessment[1]{score,reasoning}:
(Int 1-100),"Final synthesis of why this score was given"
"""

# ==========================================
# Fractal Chain of Thought (FCoT) Prompts
# ==========================================

FCOT_MACRO_PROMPT = """
{system_persona}

**Fractal Chain of Thought - Stage 1: Macro-Scale Hypothesis (Wide Aperture)**

You are analyzing a video for factuality.
**Context:** Caption: "{caption}" | Transcript: "{transcript}"

1. **Global Scan**: Observe the video, audio, and caption as a whole entity.
2. **Context Aperture**: Wide. Assess the overall intent (Humor, Information, Political, Social) and the setting.
3. **Macro Hypothesis**: Formulate a high-level hypothesis about the veracity. (e.g., "The video is likely authentic but the caption misrepresents the location" or "The audio quality suggests synthetic generation").

**Objective**: Maximize **Coverage** (broadly explore potential angles of manipulation).

**Output**: A concise paragraph summarizing the "Macro Hypothesis".
"""

FCOT_MESO_PROMPT = """
**Fractal Chain of Thought - Stage 2: Meso-Scale Expansion (Recursive Verification)**

**Current Macro Hypothesis**: "{macro_hypothesis}"

**Action**: Zoom In. Decompose the hypothesis into specific verification branches.
Perform the following checks recursively:

1. **Visual Branch**: Look for specific artifacts, lighting inconsistencies, cuts, or deepfake signs.
2. **Audio Branch**: Analyze lip-sync, background noise consistency, and voice tonality.
3. **Logical Branch**: Does the visual evidence strictly support the caption's claim? Are there logical fallacies?

**Dual-Objective Self-Correction**:
- **Faithfulness**: Do not hallucinate details not present in the video.
- **Coverage**: Did you miss any subtle cues?

**Output**: Detailed "Micro-Observations" for each branch. If you find contradictions to the Macro Hypothesis, note them explicitly as **"Self-Correction"**.
"""

FCOT_SYNTHESIS_PROMPT = """
**Fractal Chain of Thought - Stage 3: Inter-Scale Consensus & Synthesis**

**Action**: Integrate your Macro Hypothesis and Micro-Observations.
- **Consensus Check**: If Micro-Observations contradict the Macro Hypothesis, prioritize the Micro evidence (Self-Correction).
- **Compression**: Synthesize the findings into the final structured format.
- **Tags**: Assign 3-5 high-level tags (e.g., "political", "fabricated", "humor").

**Output Format**:
Strictly fill out the following TOON schema based on the consensus. Do not include markdown code blocks.

**TOON SCHEMA**:
{toon_schema}

{score_instructions}
"""
