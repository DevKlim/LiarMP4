---
title: VFacts
emoji: 😀
colorFrom: gray
colorTo: gray
sdk: docker
pinned: false
---

# VFacts - Video Veracity & Analysis Platform

## Research Overview

The liarMP4 project investigates the efficacy of Generative AI (GenAI) systems in detecting "contextual malformation" in video content, as opposed to traditional Predictive AI (PredAI) which focuses on metadata and engagement velocity.

While traditional content moderation relies on scalar probabilities derived from tabular data (account age, keyword triggers), this research proposes a **Fractal Chain-of-Thought** methodology. This approach utilizes Multimodal Large Language Models to analyze the semantic dissonance between visual evidence, audio waveforms, and textual claims.

The system generates **Veracity Vectors**, multi-dimensional scores representing Visual Integrity, Audio Integrity, and Cross-Modal Alignment—outputting data in a strict Token-Oriented Object Notation (TOON) schema.

## Key Features

*   **Predictive Benchmarking:** Comparison against AutoGluon/Gradient Boosting models trained on engagement metadata.
*   **Fractal Chain-of-Thought (FCoT):** A recursive inference strategy that hypothesizes intent at a macro-scale and verifies pixel/audio artifacts at a meso-scale.
*   **TOON Schema:** A standardized output format ensuring strict type adherence for database integration.
*   **Human-in-the-Loop (HITL) Protocol:** A browser-based grounding workflow to calibrate AI "reasoning" against human authorial intent.

## Project Resources

*   **Live Demonstration (Hugging Face):** [https://huggingface.co/spaces/GlazedDon0t/liarMP4](https://huggingface.co/spaces/GlazedDon0t/liarMP4)
*   **Source Code (GitHub):** [https://github.com/DevKlim/LiarMP4](https://github.com/DevKlim/LiarMP4)

## Repository Structure

*   **src/**: Core inference logic for the Generative AI pipeline and FCoT implementation.
*   **preprocessing_tools/**: Scripts for training Predictive AI models on tabular datasets.
*   **extension/**: Browser extension source code for the Human-in-the-Loop labeling workflow.
*   **data/**: Benchmark datasets containing engagement metadata and manual veracity labels.

## Installation and Usage

This project is containerized to ensure reproducibility across different environments. The entire pipeline, including the inference logic and database connections, can be deployed using Docker.

### Prerequisites

*   Docker Engine
*   Docker Compose

### Deployment Instructions

1.  Clone the repository:
    ```bash
    git clone https://github.com/DevKlim/LiarMP4.git
    ```

2.  Navigate to the project directory:
    ```bash
    cd LiarMP4/liarMP4
    ```

3.  Build and run the containerized environment:
    ```bash
    docker-compose up --build
    ```
### Extension Instructions

1. Unzip extension.zip to obtain folder

2. Open Chrome - Input `chrome://extensions` in address bar - Trun on `Developer mode` (top right) - Click `Load unpacked` & select the folder just obtained - Activate the extensions in `All Extensions`.

3. Go to any post on X/Twitter
- button at the bottom right can open the interface
- buttons in each post can select the post and allow user to label the post in the interface.
The system will initialize the backend services and expose the necessary endpoints for the analysis pipeline.

## License

This research project is open-source. Please refer to the LICENSE file in the repository for specific terms regarding usage and distribution.

## Authors

Kliment Ho, Shiwei Yang, Keqing Li