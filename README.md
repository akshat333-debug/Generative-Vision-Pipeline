# 🚀 ANTIGRAVITY: Deep Generative Image Pipeline

[![GitHub repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/akshat333-debug/ML-project)
[![Python & PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Diffusers-red)](https://pytorch.org/)

ANTIGRAVITY is a robust, dynamic end-to-end Text-to-Image Generation and Conditional GAN rendering pipeline designed specifically for professional AI execution and reproducible ML research. 

This repository was built targeting high efficiency (running constraints down to Kaggle T4 GPUs) while supporting complex image synthesis strategies including LoRA fine-tuning, cross-attention in local GANs, and HF Transformers-backed NLP pipelines.

---

## 📖 Problem Statement
Standard diffusion models are exceptionally powerful but highly generalized, heavy, and difficult to isolate for edge-case deployment. Furthermore, many fundamental generative approaches (like standard CGANs) suffer in image quality due to a lack of structural understanding of contextual prompts.

**Our Objective:** Create an extensible architecture capable of:
1. Fine-tuning heavy foundational models (Stable Diffusion v1.5) via Low-Rank Adaptation (LoRA) for specialized domains.
2. Developing from scratch a Conditional Generative Adversarial Network (CGAN) that uses state-of-the-art NLP representations (HuggingFace tokenizers).
3. Utilizing self-attention mechanisms natively within the GAN generator to drastically improve shape and target coherence.
4. Binding the entire array of models into a single, unified Gradio interface.

## 📊 Datasets & Analysis

### 1. Oxford-102 Flowers (EDA)
To establish analytical baselines, we performed heavy Exploratory Data Analysis (EDA) on the `Oxford-102` dataset. 
- **Methodology:** Text descriptions were matched with structural images. 
- **Analysis Captured:** Description length distribution, image resolution clusters, and class-balance evaluation. 
- **Notebook:** [`01_Dataset_Analysis_Oxford102.ipynb`](./notebooks/01_Dataset_Analysis_Oxford102.ipynb)

*<-- INSERT EDA PLOT/VISUALIZATION HERE -->*
*(Place an image of the word-length distributions or dataset samples)*

### 2. Custom Shapes Synthetic Dataset
For our CGAN, we utilized a synthetic dataset targeting primitive label generation: `circles`, `squares`, and `triangles` embedded directly against one-hot labels.

### 3. Specialized Fine-Tuning Corpus (Art/Medical)
Stable Diffusion LoRA models were trained on a specialized, narrow sub-domain dataset to forcefully shift the `.unet` style without destroying zero-shot capabilities.

---

## 🛠️ Architecture & Methodology

Our execution spans three major pillars:

### A. Preprocessing & Feature Engineering
- **Script:** [`scripts/text_processing.py`](./scripts/text_processing.py)
- **Execution:** Uses `transformers` to preprocess raw string prompts into tokenized, embedded encoded representations to prevent semantic loss before entering generative models.

### B. Stable Diffusion Fine-Tuning (LoRA)
- **Script/Notebook:** [`04_FineTune_LoRA_SD15.ipynb`](./notebooks/04_FineTune_LoRA_SD15.ipynb)
- **Execution:** To keep memory sub-16GB (Kaggle T4), we sliced attention and injected Low-Rank Adaptation (PEFT). This explicitly trained only minimal cross-attention matrix layers. 

### C. Self-Attention Generative Adversarial Network (CGAN)
- **Architecture Source:** [`models/cgan_attention.py`](./models/cgan_attention.py)
- **Model Selection / Baseline Comparison:** We compared a standard DCGAN (Baseline) with our Self-Attention inject CGAN (Advanced). 
- **Result:** The advanced model effectively concentrates on pertinent pixel zones relative to text embeddings, stabilizing mode collapse and increasing perceptual quality. 

---

## 📈 Results & Visual Outputs

### Baseline vs. Advanced CGAN Evaluation
Our metrics indicated substantial improvement when adopting self/cross-attention over vanilla convolutional generation.

*<-- INSERT METRICS/CONFUSION MATRIX OR COMPARISON VISUAL HERE -->*
*(Place training loss curves or the side-by-side grid of generated images here)*

### Gradio Interface
We built a dynamic `.ipynb` and `.py` pipeline wrapper to interface live.

*<-- INSERT GRADIO UI SCREENSHOT HERE -->*

---

## 💻 How to Run (Reproducibility)

The code is strictly commented, modularized, and designed for clean reproduction.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akshat333-debug/ML-project.git
   cd ML-project
   ```

2. **Install Requirements:**
   *(Ensure you are using Python 3.10+)*
   ```bash
   pip install torch diffusers transformers peft gradio matplotlib pandas
   ```

3. **Launch the Core App:**
   ```bash
   python Stable_Diffusion.py
   ```
   *Alternatively, run through `Stable_Diffusion.ipynb` to evaluate via Jupyter.*

## 📂 Repository Structure
- `Stable_Diffusion.py` - Core App UI Wrapper
- `notebooks/` - Comprehensive Jupyter EDA, Text embedding, CGAN, and LoRA scripts.
- `scripts/` - Text processing / NLP logic.
- `models/` - PyTorch `cgan_attention` class structures.
- `lora_unet_weights/` - Saved state dictionaries for SD fine-tuning.
