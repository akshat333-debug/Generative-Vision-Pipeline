# Project Roadmap (GSD Phases)

- [x] **Phase 0: GSD Bootstrapping**  
  Initialize structural documentation, task tracker, and core directories.
- [ ] **Phase 1: Dataset Analysis (Task 4)**  
  Load and analyze public dataset (Oxford-102 Flowers) to map statistical baselines.
- [ ] **Phase 2: NLP Embedding Generation (Task 3)**  
  Integrate `transformers` library to build text tokenizer scripts.
- [ ] **Phase 3: CGAN with Textual Labels & Attention (Tasks 2 & 5)**  
  Develop an Attention-enhanced Conditional GAN capable of basic shape generation trained on a synthetic generated dataset.
- [ ] **Phase 4: Fine-Tuning Stable Diffusion (Task 1)**  
  Employ LoRA weights configuration over base SD1.5 on Kaggle T4 GPUs, tuning it towards a custom specialized subset (art/medical).
- [ ] **Phase 5: Pipeline Integration (Task 6)**  
  Bring the CGAN, Custom SD model, and text processing module together into an expanded `Stable_Diffusion.ipynb` Gradio dashboard.
