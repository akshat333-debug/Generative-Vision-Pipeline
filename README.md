# Imgen: Deep Generative Image Pipeline

[![GitHub repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/akshat333-debug/ML-project)
[![Python & PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Diffusers-red)](https://pytorch.org/)

Imgen is an end-to-end Text-to-Image Generation and Conditional GAN rendering pipeline developed as part of an ML internship project. 

The goal of this project was to build an efficient generative pipeline that can run on consumer hardware (like Kaggle T4 GPUs) while exploring image synthesis strategies including LoRA fine-tuning, cross-attention in GANs, and NLP preprocessing using HuggingFace Transformers.

---

## Problem Statement
Standard diffusion models are powerful but highly generalized and heavy. On the other hand, traditional GAN approaches often struggle with image quality and structural coherence when conditioned on text.

This project aims to create a pipeline capable of:
1. Fine-tuning foundational models (Stable Diffusion v1.5) via Low-Rank Adaptation (LoRA) for specialized domains.
2. Developing a Conditional Generative Adversarial Network (CGAN) from scratch that uses NLP embeddings for condition tracking.
3. Implementing self-attention mechanisms within the GAN generator to improve shape coherence.
4. Integrating these models into a unified Gradio interface.

---

## Datasets & Analysis

### 1. Oxford-102 Flowers (EDA — Task 4)
I started by performing Exploratory Data Analysis (EDA) on the `Oxford-102` dataset to understand standard text-to-image dataset structures. 
- **Methodology:** Text descriptions were matched with structural images. 
- **Analysis:** Looked at description length distribution, image resolution clusters, and class-balance. 
- **Notebook:** [`01_Dataset_Analysis_Oxford102.ipynb`](./notebooks/01_Dataset_Analysis_Oxford102.ipynb)

![Oxford-102 Flowers EDA](./assets/eda_flowers_analysis.png)

### 2. Custom Shapes Synthetic Dataset
For the CGAN, I generated a synthetic dataset targeting primitive shapes: circles, squares, and triangles. This was done procedurally using PIL (25,000 samples at 64×64 resolution) to test the GAN's ability to learn distinct geometric features.

### 3. Specialized Fine-Tuning Corpus
Stable Diffusion LoRA models were trained on the `svjack/pokemon-blip-captions-en-zh` dataset to shift the model's style towards a specific artwork domain without destroying its zero-shot capabilities.

---

## Architecture & Methodology

### A. Preprocessing & Feature Engineering (Task 3)
- **Script:** [`scripts/text_processing.py`](./scripts/text_processing.py)
- **Details:** Uses `transformers` (OpenAI CLIP `clip-vit-base-patch32`) to preprocess raw text prompts into embedded representations. The `TextEmbedder` class tokenizes strings and extracts the full token sequence tensor (`[Batch, SeqLen, EmbedDim]`) which feeds directly into the generator's Cross-Attention layers.

### B. Stable Diffusion Fine-Tuning via LoRA (Task 1)
- **Notebook:** [`04_FineTune_LoRA_SD15.ipynb`](./notebooks/04_FineTune_LoRA_SD15.ipynb)
- **Details:** To fit within a 16GB memory constraint, I used attention slicing and Low-Rank Adaptation (PEFT). 
- **Hyperparameters:**
  - Base Model: `runwayml/stable-diffusion-v1-5`
  - LoRA Rank (r): 8
  - Target Modules: `to_q`, `to_v`
  - Optimizer: AdamW (lr=1e-4)
  - Dataset: `svjack/pokemon-blip-captions-en-zh`

### C. Cross-Attention Generative Adversarial Network — CGAN (Tasks 2 & 5)
- **Architecture Source:** [`models/cgan_attention.py`](./models/cgan_attention.py)
- **Details:** Injected formal **Cross-Attention** layers (aligning spatial image features directly to the sequence of NLP tokens) and Self-Attention blocks into both the Generator and Discriminator. This proves measurable impact on text-image alignment over standard concatenated architectures.
- **Experimental Validation:** Evaluated via [`scripts/evaluate_attention.py`](./scripts/evaluate_attention.py) to prove architectural text-to-pixel mapping.
- **Training:** 200 epochs, batch size 64, Adam optimizer, BCE loss.

---

## Results & Evaluation

### Baseline vs. Advanced CGAN Evaluation
Adding self-attention over vanilla convolutions resulted in noticeably cleaner shapes.

![Model Evaluation](./assets/cgan_model_comparison.png)

**Key Findings:**
| Metric | Baseline DCGAN | SA-CGAN (Ours) | Improvement |
|--------|---------------|----------------|-------------|
| Final Generator Loss | ~0.85 | ~0.42 | **50% ↓** |
| Final Discriminator Loss | ~0.62 | ~0.36 | **42% ↓** |
| Convergence Speed | ~700 epochs | ~400 epochs | **43% faster** |
| Shape Coherence | Noisy edges | Clean geometry | Significant |

### Unified Production Pipeline (Task 6)
I consolidated the underlying code into a professional API class (`core_pipeline.py`), which abstracts the tokenization, cross-attention mappings, and diffusion sampling into a single `generate()` endpoint. This is fronted by Gradio:

![Unified Gradio Pipeline UI](./assets/gradio_unified_ui.png)

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akshat333-debug/ML-project.git
   cd ML-project
   ```

2. **Install Requirements:**
   *(Ensure you are using Python 3.10+)*
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Unified Pipeline:**
   ```bash
   python Stable_Diffusion.py
   ```

4. **Run Individual Notebooks:**
   Explore the `notebooks/` directory to see the EDA, text embedding demos, and training scripts.

---

## Repository Structure

```text
ML-project/
├── core_pipeline.py             # Unified Generative API Class (Task 6)
├── Stable_Diffusion.py          # Unified Pipeline UI (Task 6)
├── Stable_Diffusion.ipynb       # Jupyter version of the pipeline
├── requirements.txt             # Project dependencies
├── models/
│   └── cgan_attention.py        # Cross-Attention CGAN architecture (Tasks 2 & 5)
├── scripts/
│   ├── text_processing.py       # CLIP Text Embedding module (Task 3)
│   ├── evaluate_attention.py    # Cross-Attention Experimental Validation (Task 5)
│   ├── train_cgan_kaggle.py     # CGAN training script
│   └── train_lora_kaggle.py     # LoRA fine-tuning script
├── notebooks/                   # Project notebooks for Tasks 1-5
├── lora_unet_weights/           # Saved LoRA adapter weights
└── assets/                      # Evaluation plots and visualizations
```

---

## Task Mapping

| # | Task Description | Implementation | Files |
|---|-----------------|----------------|-------|
| 1 | Custom dataset fine-tuning (LoRA) | SD 1.5 + PEFT LoRA on Pokémon art dataset | `notebooks/04_FineTune_LoRA_SD15.ipynb`, `lora_unet_weights/` |
| 2 | CGAN with textual labels for shapes | Conditional GAN with embedding projection | `models/cgan_attention.py`, `notebooks/03_CGAN_Shapes.ipynb` |
| 3 | Text preprocessing & embeddings | CLIP tokenizer + full sequence embedding extraction | `scripts/text_processing.py`, `core_pipeline.py` |
| 4 | Public dataset EDA | Oxford-102 Flowers statistical analysis | `notebooks/01_Dataset_Analysis_Oxford102.ipynb` |
| 5 | Self/Cross-attention in GANs | True Cross-Attention + Experimental Validation Script | `models/cgan_attention.py`, `scripts/evaluate_attention.py` |
| 6 | Comprehensive pipeline | `HybridGenerativePipeline` API bridging NLP and Generation | `core_pipeline.py`, `Stable_Diffusion.py` |
