---
license: creativeml-openrail-m
base_model: runwayml/stable-diffusion-v1-5
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---

# LoRA Fine-Tuning - Stable Diffusion 1.5

These are LoRA adaptation weights for `runwayml/stable-diffusion-v1-5`. The weights were fine-tuned on the `svjack/pokemon-blip-captions-en-zh` dataset to generate domain-specific visuals.

## Intended uses & limitations
You can use this LoRA in the `diffusers` library to generate stylized Pokémon-like images. It is not intended for realistic image generation.

## Training details
This model was trained as part of an ML internship project exploring text-to-image pipeline generation.

- **Base Model**: runwayml/stable-diffusion-v1-5
- **Dataset**: svjack/pokemon-blip-captions-en-zh
- **Rank (r)**: 8
- **Alpha**: 32
- **Target Modules**: `to_q`, `to_v`
- **Learning Rate**: 1e-4 with cosine decay
- **Batch Size (effective)**: 8
- **Epochs**: 30

## Usage
```python
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.load_lora_weights("./lora_unet_weights")
pipeline.to("cuda")

image = pipeline("A cute fire-type pokemon").images[0]
image.save("pokemon.png")
```