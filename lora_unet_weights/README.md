---
base_model: runwayml/stable-diffusion-v1-5
library_name: peft
tags:
- lora
- stable-diffusion
- text-to-image
- fine-tuning
---

# LoRA Fine-Tuned UNet Weights — ANTIGRAVITY Pipeline

## Model Details

This adapter contains Low-Rank Adaptation (LoRA) weights fine-tuned on top of `runwayml/stable-diffusion-v1-5` UNet for domain-specific image generation.

- **Developed by:** Akshat Agrawal
- **Model type:** LoRA adapter for UNet2DConditionModel  
- **Base Model:** `runwayml/stable-diffusion-v1-5`
- **Fine-tuned on:** `svjack/pokemon-blip-captions-en-zh` (stylized artwork sprites with BLIP captions)
- **Framework:** PEFT 0.18.1 + Diffusers + PyTorch

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 8 |
| LoRA Alpha | 32 |
| Target Modules | `to_q`, `to_v` (cross-attention projections) |
| LoRA Dropout | 0.0 |
| Bias | none |
| Training Precision | FP16 mixed |
| Optimizer | AdamW (lr=1e-4) |
| Hardware | Kaggle T4 GPU (16GB VRAM) |
| Noise Scheduler | DDPMScheduler |
| Image Resolution | 512×512 |
| VAE Scaling | Standard latent scaling factor |

## Usage

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.load_lora_weights("lora_unet_weights", weight_name="adapter_model.safetensors")
pipe = pipe.to("cuda")
```

## Files

- `adapter_config.json` — PEFT LoRA configuration
- `adapter_model.safetensors` — Trained weight deltas (~3.2MB)

### Framework versions

- PEFT 0.18.1