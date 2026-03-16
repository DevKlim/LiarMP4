# liarMP4: Multimodal Content Moderation via Fractal Chain-of-Thought

## Project Context
This project is an implementation/extension of the **Alternus Vera Research Project** supervised by **Dr. Ali Arsanjani**. It focuses on the **AI4Good** mission of identifying and mitigating digital misinformation and disinformation through advanced techniques such as **Veracity Vectors** and **Truthness Tensors**.

For more details on the core methodology, visit [Alternus Vera](https://alternusvera.com) or the[LiarMP4 Repository](https://github.com/DevKlim/LiarMP4).

## Research Overview

The liarMP4 project investigates the efficacy of Generative AI (GenAI) systems in detecting "contextual malformation" in video content, as opposed to traditional Predictive AI (PredAI) which focuses on metadata and engagement velocity.

While traditional content moderation relies on scalar probabilities derived from tabular data (account age, keyword triggers), this research proposes a **Fractal Chain-of-Thought** methodology. This approach utilizes Multimodal Large Language Models to analyze the semantic dissonance between visual evidence, audio waveforms, and textual claims.

The system generates **Veracity Vectors**, multi-dimensional scores representing Visual Integrity, Audio Integrity, and Cross-Modal Alignment—outputting data in a strict Token-Oriented Object Notation (TOON) schema.

## Multi-Agentic Architecture & Google Agent ADK

The `liarMP4` factuality pipeline is orchestrated using a highly scalable **Multi-Agentic Architecture** designed natively around **Google's Agent Development Kit (ADK)** and the **A2A JSON-RPC Protocol**. 

This allows us to decouple complex multimodal reasoning into specialized agents capable of cross-communicating:
1.  **Orchestrator Agent:** Ingests links, manages queue states, and routes execution payloads.
2.  **Vision & Audio Agents:** Connects directly with foundational models (Gemini 2.0 / Qwen3-VL) to extract spatial pixel anomalies and audio waveform inconsistencies.
3.  **Community Context Agent:** Scrapes and analyzes user comment sentiment dynamically to surface "Community Notes" that ground the AI's contextual understanding.
4.  **Verification Agent (A2A):** A centralized evaluator that manages the recursive Fractal Chain-of-Thought (FCoT) logic. It uses tools like Google Search Retrieval to ground its answers, automatically requesting re-prompts if semantic vectors (like Logic or Emotion) fail confidence checks.

By standardizing on the A2A protocol, the system dynamically shifts configuration parameters (Prompt Personas, Reasoning Methods, Base Models) during live execution. This empowers our automated **"Hill Climbing" Verification** system—where human-verified Ground Truth records are continuously re-queued through the agents to benchmark and mathematically optimize the most accurate agentic configurations.

## Key Features

*   **Fractal Chain-of-Thought (FCoT):** A recursive inference strategy that hypothesizes intent at a macro-scale and verifies pixel/audio artifacts at a meso-scale.
*   **Predictive vs Generative Benchmarking:** Direct comparisons against AutoGluon/Gradient Boosting models evaluating structural feature differences over holistic comprehension.
*   **Comprehensive Scoring:** Calculates sophisticated distance models across 8 distinct verification vectors plus tag accuracy.
*   **Human-in-the-Loop (HITL) Protocol:** A browser-based grounding workflow (via browser extension) to calibrate AI "reasoning" against human authorial intent.

## Project Resources

*   **Live Demonstration (Hugging Face):**[https://huggingface.co/spaces/GlazedDon0t/liarMP4](https://huggingface.co/spaces/GlazedDon0t/liarMP4)
*   **Source Code (GitHub):** [https://github.com/DevKlim/LiarMP4](https://github.com/DevKlim/LiarMP4)
*   **Vision for this project:**[https://alternusvera.com](https://alternusvera.com)

## Installation and Deployment

This project is containerized to ensure reproducibility.

1.  Clone the repository:
    ```bash
    git clone https://github.com/DevKlim/LiarMP4.git
    cd LiarMP4/liarMP4
    ```

2.  Build and run the containerized environment:
    ```bash
    docker-compose up --build
    ```

The system will initialize the backend services, mount the A2A endpoints, and expose the UI for the analysis pipeline.

## Acknowledgments & Citation

**Project Attribution:** This work was developed as part of the Alternus Vera Research Project, focusing on AI4Good: Misinformation & Disinformation Detection, Ranking, and Mitigation. This project was conducted under the supervision of Dr. Ali Arsanjani.

**Project Website:**[alternusvera.com](https://alternusvera.com)
**Core Codebase:**[GitHub: DevKlim/LiarMP4](https://github.com/DevKlim/LiarMP4)

If you use this work academically or integrate it into your systems, please consider citing the core Alternus Vera project:

```bibtex
@misc{arsanjani_alternusvera,
  author = {Arsanjani, Ali and others},
  title = {Alternus Vera: A Research Project for LiarMP4 Detecting Contextual Malformation with Fractal Chain of Thought},
  year = {2024},
  publisher = {Alternus Vera Research Group},
  url = {https://alternusvera.com},
  note = {Core codebase: https://github.com/DevKlim/LiarMP4}
}
```

## License

This research project is open-source. Please refer to the LICENSE file in the repository for specific terms regarding usage and distribution.
