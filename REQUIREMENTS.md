# Project Requirements v1

## Existing Requirements
- HuggingFace Diffusers UI to quickly spin up text-to-image pre-trained variants (SD 1.5, Dreamshaper, Realistic Vision) on Gradio.

## Missing Essentials (Internship Tasks)
- **Task 1: Custom Dataset Fine-Tuning**  
  Must allow fine-tuning of SD/DALL-E via LoRA so it runs on Kaggle T4 resources. Target domain: specialized imagery (art/medical).
- **Task 2: CGAN for Shapes**  
  Must generate synthetic basic geometric shape dataset (circles, squares, triangles) if no appropriate standard dataset exists. Train CGAN on labels and condition.
- **Task 3: Text Embeddings module**  
  Use `transformers` to create tokenized/encoded text embeddings dynamically.
- **Task 4: Public Dataset EDA**  
  Detailed statistical analysis mapping out text descriptions combined with imagery (Oxford-102 Flowers or COCO).
- **Task 5: Self-Attention in GANs**  
  Elevate Task 2's GAN architecture by injecting self-attention / cross-attention blocks.
- **Task 6: Unified Pipeline UI**  
  Bind custom tuned SD + internal CGAN + text preprocessor into a single comprehensive Gradio UI.
