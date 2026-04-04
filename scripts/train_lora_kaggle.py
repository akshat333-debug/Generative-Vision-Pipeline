#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script — Kaggle 2×T4 GPU
==========================================
Fine-tunes Stable Diffusion 1.5 UNet via LoRA on a domain-specific dataset.
Uses DataParallel for multi-GPU and gradient accumulation for effective batching.

Usage (from repo root):
    python scripts/train_lora_kaggle.py

Outputs:
    - lora_unet_weights/           (final adapter weights)
    - checkpoints/lora_step_*.pt   (periodic saves)
    - samples/lora_step_*.png      (inference samples during training)
"""

import os
import sys
import time
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATASET_ID = "svjack/pokemon-blip-captions-en-zh"
TEXT_COLUMN = "en_text"

EPOCHS = 30
BATCH_SIZE_PER_GPU = 1        # 1 per GPU (VRAM constrained)
GRADIENT_ACCUMULATION = 4     # effective batch = 1 * 2 GPUs * 4 = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0
IMAGE_SIZE = 512

# LoRA config
LORA_RANK = 8
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["to_q", "to_v"]

CHECKPOINT_EVERY = 500        # steps
SAMPLE_EVERY = 250            # generate inference samples every N steps
SAVE_DIR = "lora_unet_weights"

SAMPLE_PROMPTS = [
    "a cute fire-type pokemon with flames on its tail",
    "a blue water pokemon swimming in the ocean",
    "a green grass-type pokemon in a forest",
    "a yellow electric pokemon with lightning bolts",
]

# ============================================================================
# Dataset Wrapper
# ============================================================================

class TextImageDataset(Dataset):
    """Wraps a HuggingFace dataset for training."""

    def __init__(self, hf_dataset, transform, text_col="en_text"):
        self.dataset = hf_dataset
        self.transform = transform
        self.text_col = text_col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        image = self.transform(image)
        text = item[self.text_col]
        return {"pixel_values": image, "text": text}


# ============================================================================
# Generate Inference Samples
# ============================================================================

def generate_inference_samples(pipe, step, prompts, save_dir="samples"):
    """Generate images with current LoRA weights for visual progress."""
    os.makedirs(save_dir, exist_ok=True)
    pipe.unet.eval()

    fig, axes = plt.subplots(1, len(prompts), figsize=(5 * len(prompts), 5))
    if len(prompts) == 1:
        axes = [axes]

    with torch.no_grad(), torch.autocast("cuda"):
        for i, prompt in enumerate(prompts):
            result = pipe(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512, height=512,
            )
            axes[i].imshow(result.images[0])
            axes[i].set_title(prompt[:40] + "...", fontsize=8)
            axes[i].axis('off')

    plt.suptitle(f"LoRA Samples — Step {step}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"lora_step_{step:05d}.png"), dpi=100)
    plt.close()
    pipe.unet.train()
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    start_time = time.time()

    # ─── GPU Setup ───
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda")
    print(f"\n{'='*60}")
    print(f"LoRA FINE-TUNING — {num_gpus} GPU(s)")
    print(f"{'='*60}")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    total_batch = BATCH_SIZE_PER_GPU * max(num_gpus, 1) * GRADIENT_ACCUMULATION
    print(f"\nEffective batch size: {BATCH_SIZE_PER_GPU} × {num_gpus} GPUs × {GRADIENT_ACCUMULATION} accum = {total_batch}")

    # ─── Load Dataset ───
    print(f"\nLoading dataset: {DATASET_ID}...")
    from datasets import load_dataset
    raw_dataset = load_dataset(DATASET_ID, split="train")
    print(f"Dataset size: {len(raw_dataset)} samples")

    image_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_dataset = TextImageDataset(raw_dataset, image_transforms, TEXT_COLUMN)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_GPU * max(num_gpus, 1),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * EPOCHS
    print(f"Steps per epoch: {steps_per_epoch} | Total steps: {total_steps}")

    # ─── Load SD Components ───
    print(f"\nLoading Stable Diffusion components from {MODEL_ID}...")
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
    from transformers import CLIPTokenizer, CLIPTextModel
    from peft import LoraConfig, get_peft_model

    # Text encoder + tokenizer (frozen, GPU 0 only)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # VAE (frozen, GPU 0 only)
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # UNet with LoRA
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float16
    )
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.to(device)
    unet.train()

    # Multi-GPU for UNet
    if num_gpus > 1:
        print(f"Wrapping UNet in DataParallel across {num_gpus} GPUs")
        unet = torch.nn.DataParallel(unet)

    # Optimizer (only LoRA params are trainable)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # LR scheduler: cosine decay
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Create dirs
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\nTraining config:")
    print(f"  Epochs: {EPOCHS} | Steps/epoch: {steps_per_epoch}")
    print(f"  LR: {LEARNING_RATE} (cosine decay) | Weight decay: {WEIGHT_DECAY}")
    print(f"  LoRA: r={LORA_RANK}, alpha={LORA_ALPHA}, targets={LORA_TARGET_MODULES}")
    print(f"\n{'='*60}")
    print("Starting training...\n")

    # ─── Training Loop ───
    global_step = 0
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_dataloader):
            # 1. Encode images with VAE
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 2. Encode text prompts
            text_inputs = tokenizer(
                batch["text"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                encoder_hidden_states = text_encoder(
                    text_inputs.input_ids.to(device)
                )[0].to(dtype=torch.float16)

            # 3. Add noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,),
                device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 4. UNet forward pass (LoRA active)
            # return_dict=False gives plain tuple — avoids DataParallel + PEFT output issues
            noise_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]  # first element is the noise prediction tensor

            # 5. Loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()

            epoch_loss += loss.item() * GRADIENT_ACCUMULATION
            num_batches += 1

            # 6. Gradient accumulation step
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = (time.time() - start_time) / 60
                    print(f"  Step {global_step:5d} | Loss: {avg_loss:.5f} | LR: {lr_now:.2e} | {elapsed:.1f}min")

                # Checkpoint
                if global_step % CHECKPOINT_EVERY == 0:
                    raw_unet = unet.module if isinstance(unet, torch.nn.DataParallel) else unet
                    raw_unet.save_pretrained(f"checkpoints/lora_step_{global_step:05d}")
                    print(f"  → Checkpoint: checkpoints/lora_step_{global_step:05d}/")

                loss_history.append(epoch_loss / num_batches)

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        elapsed = (time.time() - start_time) / 60
        print(f"\n[Epoch {epoch}/{EPOCHS}] Avg Loss: {avg_epoch_loss:.5f} | "
              f"Global Step: {global_step} | Time: {elapsed:.1f}min\n")

    # ─── Save Final Weights ───
    raw_unet = unet.module if isinstance(unet, torch.nn.DataParallel) else unet
    raw_unet.save_pretrained(SAVE_DIR)
    print(f"\n{'='*60}")
    print(f"Training complete! LoRA weights saved to: {SAVE_DIR}/")
    print(f"Total steps: {global_step} | Total time: {(time.time()-start_time)/60:.1f} min")

    # Generate final inference samples
    print("\nGenerating final inference samples...")
    try:
        raw_unet_unwrapped = unet.module if isinstance(unet, torch.nn.DataParallel) else unet
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False
        )
        pipe.unet = raw_unet_unwrapped
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        generate_inference_samples(pipe, global_step, SAMPLE_PROMPTS)
        print(f"  → Final samples: samples/lora_step_{global_step:05d}.png")
        del pipe
    except Exception as e:
        print(f"  ⚠ Could not generate samples: {e}")

    # Save loss curve
    if loss_history:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(loss_history, alpha=0.8)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('MSE Loss')
        ax.set_title('LoRA Fine-Tuning Loss Curve')
        ax.grid(True, alpha=0.3)
        plt.savefig('samples/lora_loss_curve.png', dpi=150)
        plt.close()
        print("Loss curve: samples/lora_loss_curve.png")

    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone! Copy these files back to your project:")
    print(f"  - {SAVE_DIR}/adapter_config.json")
    print(f"  - {SAVE_DIR}/adapter_model.safetensors")
    print(f"  - cgan_generator.pth (if you also trained CGAN)")


if __name__ == "__main__":
    main()
