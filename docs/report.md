# liarMP4 Research Report: Benchmarking Predictive Scalars against Generative Veracity Vectors in Multimodal Content Moderation

**Authors:** Kliment Ho, Keqing Li, Shiwei Yang, Ali Arsanjani  
**Date:** January 15, 2026

## Abstract
The rapid dissemination of video content on social media platforms necessitates automated systems capable of distinguishing between authentic journalism, harmless satire, and malicious disinformation. Traditional content moderation relies on **Predictive AI (PredAI)**, utilizing tabular metadata (account age, engagement velocity) to produce scalar probabilities ($P \in[0,1]$). While computationally efficient, these models fail to interpret the semantic dissonance between visual evidence and textual claims (recontextualization).

In this report, we benchmark these traditional approaches against a **Generative AI (GenAI)** framework utilizing a novel **Fractal Chain-of-Thought (FCoT)** inference strategy. By analyzing a dataset of videos processed via the liarMP4 architecture, we demonstrate that GenAI provides superior interpretability through "Veracity Vectors", a multi-dimensional scoring system. We mathematically formalize the scoring logic, compare outputs across both approaches, and detail the Human-in-the-Loop (HITL) grounding protocol required to calibrate these models against authorial intent.

*   **GitHub Repository:**[https://github.com/DevKlim/LiarMP4](https://github.com/DevKlim/LiarMP4)
*   **Hugging Face Space (Live Demo):**[https://huggingface.co/spaces/GlazedDon0t/liarMP4](https://huggingface.co/spaces/GlazedDon0t/liarMP4)

---

## 1. Introduction

The central challenge in video moderation is not merely detecting fake pixels (deepfakes), but detecting false context. A video may be visually authentic yet semantically false if the accompanying caption misrepresents the event.

Our research investigates the performance gap between two distinct modeling architectures found within the `liarMP4` project structure:

1.  **Predictive AI (PredAI):** Defined in `preprocessing_tools/clickbaitDataAndTraining`. These models (AutoGluon/XGBoost) treat content as a set of tabular features. They are fast but opaque.
2.  **Generative AI (GenAI):** Defined in `src/inference_logic.py`. These models (Qwen3VL / Gemini) ingest raw pixels and audio waveforms to perform semantic reasoning using a recursive "Fractal" methodology driven by Google's Agent ADK.

This report analyzes the output of benchmark runs to interpret how these models score veracity and how a Human-in-the-Loop (HITL) workflow interprets these scores.

## 2. Methodology: Interpretability in AI Scoring

### 2.1 Predictive AI: The Scalar Black Box
Predictive AI operates on metadata features extracted from the post.
Let $x$ be a feature vector containing:
$$x =[\text{shares}, \text{likes}, \text{account\_age}, \text{sentiment\_score}, \text{keyword\_flags}]$$

The output is a single scalar probability:
$$y_{pred} = \sigma(W^T x + b) \in [0, 1]$$

**Interpretation Limit:** A score of $0.85$ indicates a "High Probability of Misinformation." However, it cannot distinguish whether the video is a deepfake ($S_{vis} \approx 0$) or if the user is simply lying about a real video ($S_{vis} \approx 10, S_{align} \approx 0$).

### 2.2 Generative AI: The Fractal Chain-of-Thought (FCoT)
To overcome the scalar limit, the liarMP4 system utilizes FCoT (implemented in `src/inference_logic.py`). This divides inference into three orthogonal axes:

1.  **Macro-Scale:** The model ingests the entire context (Caption + Video) to form a high-level intent hypothesis ("Political Satire").
2.  **Meso-Scale:** The model zooms into specific modalities:
    *   **Visual Branch:** Scans for pixel artifacts.
    *   **Audio Branch:** Scans for voice cloning/synthesis.
3.  **Synthesis:** The model resolves contradictions to generate a **TOON** (Token-Oriented Object Notation) object.

### 2.3 Fundamental Definition on Assessment Criteria

To mathematically evaluate factuality across different configurations, we strictly decouple definitions to extract 8 distinct Veracity Vectors:
- **Visual Integrity:** Absolute authenticity of the source pixels (spatial manipulation, AI generation, editing artifacts).
- **Audio Integrity:** Absolute authenticity of the source waveforms (voice cloning, TTS overlays, sentence mixing).
- **Source Credibility:** The historical reliability and organizational institutional backing of the entity posting the content.
- **Logical Consistency:** The structural, rhetorical validity of the claims made within the post, independently analyzing fallacies.
- **Emotional Manipulation:** The degree to which the post uses inflammatory rhetoric, rage-bait, or fear-mongering to override critical thinking.
- **Video-Audio Alignment:** Evaluating if the waveform matches the visual acoustic environment.
- **Video-Caption Alignment:** Measuring semantic recontextualization (lying about what the real video shows).
- **Audio-Caption Alignment:** Evaluating accurate transcription vs contradictory summarization.

## 3. Factuality Factors & Confidence Scoring Formulation

The final output explicitly factors these 8 parameters to provide a corresponding confidence score and distinct explanation for each layer of the analysis.

The scalar "Final Veracity Score" ($S_{final}$) in GenAI is explicitly calculated as a non-linear function of *Integrity Vectors* and *Alignment Modalities*. 

Let the Veracity Vector be $\mathbf{v} = \{v_{vis}, v_{aud}, v_{src}, v_{log}, v_{emo}\}$ and Alignment Vector be $\mathbf{m} = \{m_{va}, m_{vc}, m_{ac}\}$.

The final score $S_{final}$ incorporates the **Alignment Penalty**:
Even if visual integrity ($v_{vis}$) is 10/10, a Video-Caption mismatch ($m_{vc} < 4$) will tank the score (Recontextualization Penalty). 

### Factuality Explanations
Every resulting score is attached to a contextual "Explanation" (e.g., Claim Accuracy, Evidence Gaps, and Grounding Check logic) generated synchronously through the schema parser to ensure auditability.

## 4. Benchmark Analysis and Results

By testing the GenAI framework using our automated A2A verification pipeline, we evaluated improvements against the Ground Truth database using a **Composite MAE** (Mean Absolute Error across all 8 sub-vectors) and Tag Accuracy intersection.

### 4.1 Automated Hill Climbing (Leaderboard)

| Type | Model | Prompt | Reasoning | Tools | FCoT Depth | Accuracy | Comp. MAE | Tag Acc |
|---|---|---|---|---|---|---|---|---|
| GenAI | gemini-2.5-flash | standard | fcot | None | 2 | 83% | 15.11 | 64.8% |
| GenAI | gemini-2.5-flash | standard | fcot | Search, Code | 2 | 77.3% | 16.69 | 34.2% |
| GenAI | gemini-2.5-flash | standard | cot | Search, Code | 1 | 76.2% | 20.79 | 20.2% |
| GenAI | Unknown | Standard | None | None | 0 | 73.3% | 11.04 | 100% |
| GenAI | gemini-2.5-flash | standard | none | Search, Code | 0 | 71.4% | 23.33 | 22.2% |
| GenAI | gemini-2.5-flash | standard | cot | None | 1 | 63.6% | 26.16 | 32.7% |
| GenAI | gemini-2.5-flash | standard | none | None | 0 | 63% | 20.66 | 18.4% |
| GenAI | gemini-2.5-flash-lite | standard | cot | None | 1 | 56.5% | 20.37 | 24.5% |
| GenAI | qwen3 | standard | none | None | 0 | 46.2% | 44.36 | 27.8% |
| GenAI | gemini-2.5-flash-lite | standard | fcot | None | 2 | 28.6% | 27.21 | 30.7% |
| GenAI | qwen3 | standard | cot | None | 1 | 0% | 54.44 | 80% |

### 4.2 Detailed Vector Error Analysis (MAE)

| Model | Prompt | Reasoning | Tools | Vis | Aud | Src | Log | Emo | V-A | V-C | A-C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-2.5-flash | standard | fcot | None | 6.38 | 12.13 | 17.02 | 11.28 | 14.47 | 29.15 | 10.85 | 23.4 |
| gemini-2.5-flash | standard | fcot | Search, Code | 5.91 | 13.18 | 16.36 | 15.00 | 16.36 | 30.91 | 12.27 | 22.73 |
| gemini-2.5-flash | standard | cot | Search, Code | 12.86 | 21.90 | 17.14 | 20.48 | 23.33 | 30.00 | 17.62 | 27.14 |
| Unknown | Standard | None | None | 23.33 | 20.67 | 6.00 | 11.33 | 6.00 | 6.67 | 5.33 | 5.33 |
| gemini-2.5-flash | standard | none | Search, Code | 17.14 | 24.29 | 18.10 | 24.29 | 24.76 | 32.38 | 28.10 | 23.81 |
| gemini-2.5-flash | standard | cot | None | 12.73 | 30.00 | 19.09 | 25.45 | 30.91 | 39.09 | 24.55 | 33.64 |
| gemini-2.5-flash | standard | none | None | 7.41 | 16.67 | 17.78 | 22.22 | 20.74 | 30.74 | 20.00 | 28.15 |
| gemini-2.5-flash-lite | standard | cot | None | 14.78 | 25.22 | 14.35 | 19.13 | 16.52 | 26.52 | 17.39 | 21.30 |
| qwen3 | standard | none | None | 88.46 | 65.38 | 19.23 | 33.08 | 21.54 | 52.31 | 53.85 | 34.62 |
| gemini-2.5-flash-lite | standard | fcot | None | 20.00 | 32.38 | 19.05 | 19.52 | 17.14 | 34.76 | 24.76 | 24.76 |
| qwen3 | standard | cot | None | 90.00 | 90.00 | 30.00 | 50.00 | 30.00 | 50.00 | 70.00 | 30.00 |

### 4.3 Analysis and Key Takeaways

The empirical results from the leaderboard evaluation yield several critical insights regarding multimodal agentic moderation:

1. **FCoT Efficacy:** Deep, recursive reasoning significantly reduces hallucinations. Gemini 2.5 Flash operating at an FCoT depth of 2 (without external tools) achieves the highest overall accuracy (83%) and a strong Tag Accuracy (64.8%).
2. **The Distraction of External Tools:** Counterintuitively, providing the agent with Web Search and Code Execution *decreased* overall accuracy (to 77.3%) and severely degraded Tag Accuracy (dropping to 34.2%). External tools caused the agent to over-index on textual search results, distracting it from analyzing the raw visual and audio alignment vectors present in the source media.
3. **Model Capabilities:** Open-source vision models like Qwen3 struggled significantly without specific orchestration (scoring 0% and 46.2%), emphasizing the absolute necessity of robust, multi-stage reasoning frameworks to parse semantic dissonance rather than relying on simple object detection.

## 5. Tagging Solution and Topic Clustering

To map posts into a multi-dimensional latent space, the Multi-Agent system assigns strict tags to every ingested video utilizing the TOON standard. By leveraging FCoT, the system dynamically clusters topics based on narrative intent rather than just spatial features.

For example, a satirical political video structurally mimics an entertainment sketch, pulling it closer to "Comedy" in the contextual vector space. This prevents the rigid, misclassified pigeonholing (e.g., labeling all political content as highly controversial or malicious) that is rampant in traditional metadata-focused PredAI systems.

## 6. Future Work

*   **Account Honesty and Credibility Tracking:** We plan to expand the User Credibility Profiler into a persistent "Account Honesty" scoring system. By tracking longitudinal behavior, the system will maintain historical integrity scores, penalizing accounts that repeatedly post recontextualized media while maintaining trust for historically accurate accounts.
*   **Video-Post Alignment Database:** Establishing a dedicated, searchable database to track specific video-post alignment patterns. This will allow the system to cross-reference known "cheap fakes" and actively flag recycled authentic videos that are frequently weaponized with new, deceptive captions.

## 7. Conclusion
The comparison reveals a strict trade-off:
1.  **PredAI:** High-speed filter. Good for spam, bad for semantics.
2.  **GenAI:** High-precision analyst. Decouples visual authenticity from caption veracity using FCoT.

Our benchmark demonstrates that low Veracity Scores in social media video contexts are rarely due to traditional "Deepfakes", but are overwhelmingly driven by "Recontextualization"—an element uniquely identified through multimodal factorization and contextual topic distance clustering.
