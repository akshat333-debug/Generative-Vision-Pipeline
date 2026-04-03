# ANTIGRAVITY - Project Vision

**Status:** In Progress
**Environment:** Target Deployment -> Kaggle T4 GPUs (Lightweight & LoRA optimized)

## System Goal
An AI-powered execution system handling end-to-end Text-to-Image Generation and Conditional GAN rendering. The repository functions as a dynamic, extendable Image Generation Pipeline built for rapid iteration.

## Core Pillars
1. **Extensibility:** The original single-file SD pipeline has grown into a modular machine-learning workspace.
2. **Reproducibility:** Code must run cleanly on constraint environments (Kaggle T4), strictly using efficient tuning methods like LoRA and memory slicing.
3. **Professional Analytics:** Deep EDA on large-scale generative datasets before any training is implemented.

## Tech Stack
- **Languages:** Python
- **Core ML:** PyTorch
- **Transformers/Diffusion:** HuggingFace `diffusers`, `transformers`, `peft`
- **UI:** Gradio
- **Analytics:** Matplotlib, Seaborn, Pandas

## Target Architecture (Post-Internship Tasks)
- `Stable_Diffusion.ipynb` -> Core App UI Wrapper
- `notebooks/` -> EDA, Explorations, NLP processing
- `scripts/` -> Reusable Python components for NLP embeddings and Training Loops
- `models/` -> Raw PyTorch components (e.g. CGANs with Self-Attention)
