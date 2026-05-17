#!/usr/bin/env python3
"""
CGAN Training Script — Kaggle 2×T4 GPU (with Resume)

Trains the Self-Attention Conditional GAN on synthetic shapes dataset.
Pre-computes CLIP embeddings to avoid DataParallel issues with HF models.
AUTO-RESUMES from latest checkpoint if found.

Usage (from repo root):
    python scripts/train_cgan_kaggle.py              # fresh or auto-resume
    python scripts/train_cgan_kaggle.py --resume 50   # resume from epoch 50

Outputs:
    - cgan_generator.pth                (final weights)
    - checkpoints/cgan_state_*.pt       (full state: G + D + optimizers + epoch)
    - samples/cgan_epoch_*.png          (visual progress)
"""

import os
import sys
import time
import glob
import argparse
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
import gc

# Ensure repo modules are importable
sys.path.insert(0, os.path.abspath('.'))
from models.cgan_attention import ConditionalGenerator, ConditionalDiscriminator

# Configuration
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
CHECKPOINT_EVERY = 25    # save full state every N epochs
SAMPLE_EVERY = 10        # generate sample grid every N epochs
G_STEPS = 2              # train G this many times per D step
LABEL_SMOOTH_REAL = 0.9
LABEL_SMOOTH_FAKE = 0.1
GRAD_CLIP = 1.0
NUM_WORKERS = 2

# Dataset — Synthetic Shapes with Pre-computed Embeddings

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
        from scripts.text_processing import TextEmbedder
        embedder = TextEmbedder(device="cpu")

        embeddings = {}
        for i, label in enumerate(self.labels):
            emb = embedder.get_text_embeddings([label]).squeeze(0) # Keep SeqLen x EmbedDim
            embeddings[i] = emb.detach()
            print(f"  '{label}' -> embedding sequence shape: {emb.shape}")

        del embedder
        gc.collect()
        return embeddings

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.transform(self.data[idx])
        label_idx = self.label_indices[idx]
        emb = self.label_embeddings[label_idx]
        return img, emb, label_idx


# Weight Initialization

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Sample Generation

def generate_samples(netG, label_embeddings, labels, epoch, device, save_dir="samples"):
    os.makedirs(save_dir, exist_ok=True)
    netG.eval()

    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    fig.suptitle(f'CGAN Samples — Epoch {epoch}', fontsize=14)

    with torch.no_grad():
        for row, (label_idx, label_name) in enumerate(zip(range(3), labels)):
            emb = label_embeddings[label_idx].unsqueeze(0).repeat(6, 1, 1).to(device)
            noise = torch.randn(6, Z_DIM, 1, 1, device=device)
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


# Checkpoint Save / Load

def save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, g_losses, d_losses):
    """Save FULL training state for resume."""
    raw_G = netG.module if isinstance(netG, nn.DataParallel) else netG
    raw_D = netD.module if isinstance(netD, nn.DataParallel) else netD

    state = {
        'epoch': epoch,
        'netG_state_dict': raw_G.state_dict(),
        'netD_state_dict': raw_D.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'g_losses': g_losses,
        'd_losses': d_losses,
    }

    path = f"checkpoints/cgan_state_{epoch:04d}.pt"
    torch.save(state, path)

    # Also save standalone G weights (for inference)
    torch.save(raw_G.state_dict(), "cgan_generator.pth")

    return path


def find_latest_checkpoint():
    """Auto-detect latest checkpoint."""
    pattern = "checkpoints/cgan_state_*.pt"
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by epoch number
    files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return files[-1]


def load_checkpoint(path, netG, netD, optimizerG, optimizerD, device):
    """Load full training state from checkpoint."""
    print(f"Loading checkpoint: {path}")
    state = torch.load(path, map_location=device)

    raw_G = netG.module if isinstance(netG, nn.DataParallel) else netG
    raw_D = netD.module if isinstance(netD, nn.DataParallel) else netD

    raw_G.load_state_dict(state['netG_state_dict'])
    raw_D.load_state_dict(state['netD_state_dict'])
    optimizerG.load_state_dict(state['optimizerG_state_dict'])
    optimizerD.load_state_dict(state['optimizerD_state_dict'])

    epoch = state['epoch']
    g_losses = state.get('g_losses', [])
    d_losses = state.get('d_losses', [])

    print(f"Resumed from epoch {epoch} (G_Loss: {g_losses[-1]:.4f}, D_Loss: {d_losses[-1]:.4f})")
    return epoch, g_losses, d_losses


# Main Training Loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=-1,
                        help='Resume from specific epoch (-1 = auto-detect)')
    args = parser.parse_args()

    start_time = time.time()

    # Device setup
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"CGAN TRAINING — {num_gpus} GPU(s) detected")
    print(f"{'='*60}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")

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

    # Optimizers (created BEFORE DataParallel wrapping and checkpoint loading)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # ─── Resume from checkpoint ───
    start_epoch = 1
    g_losses = []
    d_losses = []

    checkpoint_path = None
    if args.resume > 0:
        checkpoint_path = f"checkpoints/cgan_state_{args.resume:04d}.pt"
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: Checkpoint {checkpoint_path} not found, starting fresh")
            checkpoint_path = None
    else:
        # Auto-detect
        checkpoint_path = find_latest_checkpoint()

    if checkpoint_path:
        start_epoch, g_losses, d_losses = load_checkpoint(
            checkpoint_path, netG, netD, optimizerG, optimizerD, device
        )
        start_epoch += 1  # resume from NEXT epoch
    else:
        print("No checkpoint found — starting fresh training")

    # Multi-GPU (AFTER checkpoint loading to avoid key mismatches)
    if num_gpus > 1:
        print(f"Wrapping models in DataParallel across {num_gpus} GPUs")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    # Loss
    criterion = nn.BCELoss()

    # Create output dirs
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    print(f"\nTraining config:")
    print(f"  Epochs: {start_epoch} → {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR_G}")
    print(f"  G steps per D step: {G_STEPS}")
    print(f"  Label smoothing: real={LABEL_SMOOTH_REAL}, fake={LABEL_SMOOTH_FAKE}")
    print(f"  Checkpoints every {CHECKPOINT_EVERY} epochs")
    print(f"\n{'='*60}")
    print("Starting training...\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for i, (real_imgs, embeddings, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            embeddings = embeddings.to(device)
            b_size = real_imgs.size(0)

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
                g_loss = criterion(output, real_labels)
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
        if epoch % SAMPLE_EVERY == 0 or epoch == start_epoch:
            raw_G = netG.module if isinstance(netG, nn.DataParallel) else netG
            generate_samples(raw_G, dataset.label_embeddings, dataset.labels,
                           epoch, device)
            print(f"  → Saved sample grid: samples/cgan_epoch_{epoch:04d}.png")

        # Full state checkpoint
        if epoch % CHECKPOINT_EVERY == 0:
            raw_G_ref = netG.module if isinstance(netG, nn.DataParallel) else netG
            raw_D_ref = netD.module if isinstance(netD, nn.DataParallel) else netD

            # Temporarily unwrap for saving
            save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, g_losses, d_losses)
            print(f"  → Full checkpoint: checkpoints/cgan_state_{epoch:04d}.pt")
            print(f"  → Generator snapshot: cgan_generator.pth")

        # Flush CUDA cache periodically to prevent hangs
        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

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
