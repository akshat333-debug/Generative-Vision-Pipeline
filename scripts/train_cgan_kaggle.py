#!/usr/bin/env python3
"""
CGAN Training Script — Kaggle 2×T4 GPU
=======================================
Trains the Self-Attention Conditional GAN on synthetic shapes dataset.
Pre-computes CLIP embeddings to avoid DataParallel issues with HF models.

Usage (from repo root):
    python scripts/train_cgan_kaggle.py

Outputs:
    - cgan_generator.pth          (final weights)
    - checkpoints/cgan_epoch_*.pth (periodic saves)
    - samples/cgan_epoch_*.png     (visual progress)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure repo modules are importable
sys.path.insert(0, os.path.abspath('.'))
from models.cgan_attention import ConditionalGenerator, ConditionalDiscriminator

# ============================================================================
# CONFIG
# ============================================================================
EPOCHS = 200
BATCH_SIZE = 64          # 32 per GPU with 2 GPUs
Z_DIM = 100
EMBED_DIM = 512          # CLIP clip-vit-base-patch32 hidden size
LR_G = 0.0002
LR_D = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_SAMPLES = 10000      # dataset size
IMG_SIZE = 64
CHECKPOINT_EVERY = 25    # save every N epochs
SAMPLE_EVERY = 10        # generate sample grid every N epochs
G_STEPS = 2              # train G this many times per D step (stabilizes training)
LABEL_SMOOTH_REAL = 0.9  # label smoothing
LABEL_SMOOTH_FAKE = 0.1
GRAD_CLIP = 1.0
NUM_WORKERS = 2

# ============================================================================
# Dataset — Synthetic Shapes with Pre-computed Embeddings
# ============================================================================

class ShapesDataset(Dataset):
    """Generates circle/square/triangle images with pre-computed CLIP embeddings."""

    def __init__(self, num_samples=NUM_SAMPLES, img_size=IMG_SIZE):
        self.num_samples = num_samples
        self.img_size = img_size
        self.labels = ["circle", "square", "triangle"]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        print(f"Generating {num_samples} synthetic shape images...")
        self.data, self.label_indices = self._generate_data()

        # Pre-compute CLIP embeddings for each label (only 3 unique embeddings)
        print("Pre-computing CLIP embeddings for labels...")
        self.label_embeddings = self._precompute_embeddings()
        print(f"Embedding shape per label: {self.label_embeddings[0].shape}")

    def _generate_data(self):
        data = []
        label_indices = []
        colors = {
            "circle": ["blue", "cyan", "dodgerblue"],
            "square": ["red", "crimson", "tomato"],
            "triangle": ["green", "lime", "forestgreen"]
        }

        for i in range(self.num_samples):
            label_idx = np.random.randint(0, len(self.labels))
            shape_type = self.labels[label_idx]
            color = np.random.choice(colors[shape_type])

            img = Image.new('RGB', (self.img_size, self.img_size), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Randomize position slightly for variety
            cx, cy = self.img_size // 2, self.img_size // 2
            offset_x = np.random.randint(-5, 6)
            offset_y = np.random.randint(-5, 6)
            size = np.random.randint(18, 28)

            if shape_type == "circle":
                draw.ellipse([cx - size + offset_x, cy - size + offset_y,
                              cx + size + offset_x, cy + size + offset_y], fill=color)
            elif shape_type == "square":
                draw.rectangle([cx - size + offset_x, cy - size + offset_y,
                                cx + size + offset_x, cy + size + offset_y], fill=color)
            elif shape_type == "triangle":
                pts = [
                    (cx + offset_x, cy - size + offset_y),
                    (cx - size + offset_x, cy + size + offset_y),
                    (cx + size + offset_x, cy + size + offset_y)
                ]
                draw.polygon(pts, fill=color)

            data.append(img)
            label_indices.append(label_idx)

        return data, label_indices

    def _precompute_embeddings(self):
        """Compute CLIP embeddings for each label string once."""
        from scripts.text_processing import TextEmbedder
        embedder = TextEmbedder(device="cpu")  # CPU to avoid GPU memory issues during init

        embeddings = {}
        for i, label in enumerate(self.labels):
            # Get embedding and mean-pool over sequence length
            emb = embedder.get_text_embeddings([label])  # (1, seq_len, 512)
            emb = emb.mean(dim=1).squeeze(0)              # (512,)
            embeddings[i] = emb.detach()
            print(f"  '{label}' -> embedding norm: {emb.norm():.4f}")

        del embedder  # free memory
        return embeddings

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.transform(self.data[idx])
        label_idx = self.label_indices[idx]
        emb = self.label_embeddings[label_idx]  # pre-computed (512,)
        return img, emb, label_idx


# ============================================================================
# Weight Initialization
# ============================================================================

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ============================================================================
# Sample Generation for Visual Progress
# ============================================================================

def generate_samples(netG, label_embeddings, labels, epoch, device, save_dir="samples"):
    os.makedirs(save_dir, exist_ok=True)
    netG.eval()

    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    fig.suptitle(f'CGAN Samples — Epoch {epoch}', fontsize=14)

    with torch.no_grad():
        for row, (label_idx, label_name) in enumerate(zip(range(3), labels)):
            emb = label_embeddings[label_idx].unsqueeze(0).repeat(6, 1).to(device)
            noise = torch.randn(6, Z_DIM, 1, 1, device=device)

            if isinstance(netG, nn.DataParallel):
                fake = netG.module(noise, emb)
            else:
                fake = netG(noise, emb)

            for col in range(6):
                img = ((fake[col].cpu().permute(1, 2, 0) + 1.0) / 2.0).clamp(0, 1).numpy()
                axes[row][col].imshow(img)
                axes[row][col].axis('off')
                if col == 0:
                    axes[row][col].set_ylabel(label_name, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cgan_epoch_{epoch:04d}.png"), dpi=100)
    plt.close()
    netG.train()


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    start_time = time.time()

    # Device setup
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"CGAN TRAINING — {num_gpus} GPU(s) detected")
    print(f"{'='*60}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB)")

    # Dataset
    dataset = ShapesDataset(num_samples=NUM_SAMPLES, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    print(f"\nDataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Models
    netG = ConditionalGenerator(z_dim=Z_DIM, embed_dim=EMBED_DIM).to(device)
    netD = ConditionalDiscriminator(embed_dim=EMBED_DIM).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Multi-GPU (DataParallel — safe since CLIP is already done)
    if num_gpus > 1:
        print(f"Wrapping models in DataParallel across {num_gpus} GPUs")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    # Optimizers
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # Loss
    criterion = nn.BCELoss()

    # Create output dirs
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    print(f"\nTraining config:")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR_G}")
    print(f"  G steps per D step: {G_STEPS}")
    print(f"  Label smoothing: real={LABEL_SMOOTH_REAL}, fake={LABEL_SMOOTH_FAKE}")
    print(f"  Checkpoints every {CHECKPOINT_EVERY} epochs")
    print(f"\n{'='*60}")
    print("Starting training...\n")

    # Track losses
    g_losses = []
    d_losses = []

    for epoch in range(1, EPOCHS + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for i, (real_imgs, embeddings, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            embeddings = embeddings.to(device)
            b_size = real_imgs.size(0)

            # Labels with smoothing
            real_labels = torch.full((b_size,), LABEL_SMOOTH_REAL, device=device)
            fake_labels = torch.full((b_size,), LABEL_SMOOTH_FAKE, device=device)

            # ─── Train Discriminator ───
            optimizerD.zero_grad()

            output_real = netD(real_imgs, embeddings)
            d_loss_real = criterion(output_real, real_labels)

            noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
            fake_imgs = netG(noise, embeddings)

            output_fake = netD(fake_imgs.detach(), embeddings)
            d_loss_fake = criterion(output_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), GRAD_CLIP)
            optimizerD.step()

            # ─── Train Generator (multiple steps) ───
            g_loss_total = 0.0
            for _ in range(G_STEPS):
                optimizerG.zero_grad()
                noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
                fake_imgs = netG(noise, embeddings)
                output = netD(fake_imgs, embeddings)
                g_loss = criterion(output, real_labels)  # fool discriminator
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(netG.parameters(), GRAD_CLIP)
                optimizerG.step()
                g_loss_total += g_loss.item()

            epoch_g_loss += g_loss_total / G_STEPS
            epoch_d_loss += d_loss.item()
            num_batches += 1

        # Epoch stats
        avg_g = epoch_g_loss / num_batches
        avg_d = epoch_d_loss / num_batches
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        elapsed = time.time() - start_time
        print(f"[Epoch {epoch:3d}/{EPOCHS}] "
              f"D_Loss: {avg_d:.4f} | G_Loss: {avg_g:.4f} | "
              f"Time: {elapsed/60:.1f}min")

        # Sample visualization
        if epoch % SAMPLE_EVERY == 0 or epoch == 1:
            raw_G = netG.module if isinstance(netG, nn.DataParallel) else netG
            generate_samples(raw_G, dataset.label_embeddings, dataset.labels,
                           epoch, device)
            print(f"  → Saved sample grid: samples/cgan_epoch_{epoch:04d}.png")

        # Checkpoint
        if epoch % CHECKPOINT_EVERY == 0:
            raw_G = netG.module if isinstance(netG, nn.DataParallel) else netG
            torch.save(raw_G.state_dict(), f"checkpoints/cgan_epoch_{epoch:04d}.pth")
            print(f"  → Checkpoint saved: checkpoints/cgan_epoch_{epoch:04d}.pth")

    # ─── Save Final Model ───
    raw_G = netG.module if isinstance(netG, nn.DataParallel) else netG
    torch.save(raw_G.state_dict(), "cgan_generator.pth")
    print(f"\n{'='*60}")
    print(f"Training complete! Final model: cgan_generator.pth")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")

    # Save loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g_losses, label='Generator', alpha=0.8)
    ax.plot(d_losses, label='Discriminator', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('CGAN Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('samples/training_loss_curve.png', dpi=150)
    plt.close()
    print("Loss curve saved: samples/training_loss_curve.png")


if __name__ == "__main__":
    main()
