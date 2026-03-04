# liarMP4 Research Report: Benchmarking Predictive Scalars against Generative Veracity Vectors in Multimodal Content Moderation

**Authors:** Kliment Ho, Keqing Li, Shiwei Yang, Ali Arsanjani  
**Date:** January 15, 2026

## Abstract
The rapid dissemination of video content on social media platforms necessitates automated systems capable of distinguishing between authentic journalism, harmless satire, and malicious disinformation. Traditional content moderation relies on **Predictive AI (PredAI)**, utilizing tabular metadata (account age, engagement velocity) to produce scalar probabilities ($P \in[0,1]$). While computationally efficient, these models fail to interpret the semantic dissonance between visual evidence and textual claims (recontextualization).

In this report, we benchmark these traditional approaches against a **Generative AI (GenAI)** framework utilizing a novel **Fractal Chain-of-Thought (FCoT)** inference strategy. By analyzing a dataset of videos processed via the liarMP4 architecture, we demonstrate that GenAI provides superior interpretability through "Veracity Vectors", a multi-dimensional scoring system. We mathematically formalize the scoring logic, compare outputs across both approaches, and detail the Human-in-the-Loop (HITL) grounding protocol required to calibrate these models against authorial intent.

*   **GitHub Repository:** [https://github.com/DevKlim/LiarMP4](https://github.com/DevKlim/LiarMP4)
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

## 4. Benchmark Analysis: Sample Output Interpretations

We extracted representative samples from the training set to compare how PredAI and GenAI handle complex scenarios.

### 4.1 Table 1: Predictive AI Output (Metadata-Based)

| ID | Keywords Detected | Acct Age | Engagement | PredAI Score |
|---|---|---|---|---|
| 1988414... | H-1B, Trump, Visa, Talent | High | Viral | **Clickbait** |
| 1978859... | Mitch McConnell, Falls, Knees | Med | High | **Clickbait** |
| 1993755... | Katy Perry, Lawsuit, Dying Vet | Low | Med | **Spam/Clickbait** |

### 4.2 Table 2: Generative AI Output (Multimodal FCoT)

| ID | Visual (v) | Align (m) | Score | Reasoning Interpretation |
|---|---|---|---|---|
| 1988414... | 9 (Real) | 9 (Match) | **90** | Authentic interview. PredAI fails here by flagging it as clickbait due to topic; GenAI validates the content is grounded. |
| 1978859... | 9 (Real) | 1 (Mismatch) | **85** | *Anomaly*. Video shows McConnell falling (Real). Audio is unrelated. GenAI penalizes slightly but verifies the *visual* event occurred. |
| 1993755... | 7 (Real) | 1 (Lie) | **15** | **Recontextualization.** Video is real (Perry talking about space), Caption claims she is suing a vet. Huge alignment gap crushes the score. |

## 5. Methods Comparing & Hill Climbing Table

By testing the GenAI framework using our automated A2A verification pipeline, we evaluated improvements against the Ground Truth database using a **Composite MAE** (Mean Absolute Error across all 8 sub-vectors) and Tag Accuracy intersection.

| Iteration | Method | FCoT Depth | Composite MAE | Tag Accuracy | Avg Time/Video |
|---|---|---|---|---|---|
| 1 | Baseline PredAI | 0 | 38.4 | N/A | 1.2s |
| 2 | GenAI (CoT) | 1 | 24.1 | 68.2% | 15.4s |
| 3 | GenAI (FCoT) | 2 | 12.8 | 84.5% | 34.1s |
| 4 | Agent A2A + Search | 3 | 7.4 | 92.1% | 45.8s |

## 6. Impact of FCoT Iterations on Quality

Several iterations on the Fractal Chain-of-Thought (FCoT) were rigorously evaluated to observe its explicit effect on analytical quality.
1. **Shallow CoT (0-1 Iteration):** Captures obvious visual anomalies (e.g., weird hand configurations, glitches) but often misses subtle semantic recontextualization hidden within dense captions. Prompt to hallucinate relationships if they visually *look* related.
2. **Deep FCoT (2-3 Iterations):** Recursively questioning the Macro Hypothesis against Meso-Observations significantly reduces hallucination. The agent accurately self-corrects prior assumptions when nuanced audio cues conflict with broader visual expectations, driving the Composite MAE down by >50%.

## 7. Evaluating Topic Distance & Multimodal Influence

We evaluated the "Distance" between different topic videos (e.g., Political versus Entertainment). By prompting the agent to cross-score between diverse topics, we observed how spatial vector distances shift when the agent incorporates holistic considerations (Sound, Visuals, Captions) versus isolated elements.

- **Text-Only Consideration:** Topic distances are immense; political posts cluster tightly away from entertainment in latent space, often leading to over-categorization bias where anything "Political" defaults to "High Controversy".
- **Multimodal Consideration (Audio + Visual):** The spatial distance between ostensibly different topics converges dynamically. For instance, a satirical political video visually structurally mimics an entertainment sketch (laugh tracks, exaggerated lighting), pulling it closer to entertainment in the contextual vector space. This demonstrates that multimodal generative reasoning actively prevents the rigid, misclassified pigeonholing rampant in traditional metadata-focused PredAI systems.

## 8. Conclusion
The comparison reveals a strict trade-off:
1.  **PredAI:** High-speed filter. Good for spam, bad for semantics.
2.  **GenAI:** High-precision analyst. Decouples visual authenticity from caption veracity using FCoT.

Our benchmark demonstrates that low Veracity Scores are rarely due to "Deepfakes", but overwhelmingly due to "Recontextualization" uniquely identified through multimodal factorization and the evaluation of Topic Distance contexts.
