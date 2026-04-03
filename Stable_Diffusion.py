#!/usr/bin/env python
# coding: utf-8

# In[9]:


# get_ipython().system('nvidia-smi')


# In[10]:


import torch
print(f'pytorch version: {torch.__version__}')
print(f'cuda version: {torch.version.cuda}')
print(f'cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
  print(f'cuda device name: {torch.cuda.get_device_name(0)}')
  print(f'GPU name: {torch.cuda.get_device_name (0)}')


# In[11]:


# get_ipython().system('pip install -U pip setuptools wheel')

# Core
# get_ipython().system('pip install torch torchvision torchaudio')

# Diffusion ecosystem
# get_ipython().system('pip install diffusers transformers accelerate safetensors')

# Performance
# get_ipython().system('pip install xformers')

# Utilities
# get_ipython().system('pip install Pillow numpy matplotlib gradio')


# In[12]:


print(f'Pytorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"}')


# In[13]:


import warnings
warnings.filterwarnings("ignore")

import torch
from torch import autocast
import numpy as np
from PIL import Image
import os
import time
import gc
from typing import Optional, Tuple, List
from datetime import datetime
from importlib.metadata import version

from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler
)

import gradio as gr

# Performance boost
torch.backends.cuda.matmul.allow_tf32 = True


# In[14]:


class StableDiffusionGenerator:

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "auto"):
        try:
            self.device = self._setup_device(device)
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

            print(f"🚀 Initializing Stable Diffusion on {self.device}")
            print(f"📊 Using precision: {self.dtype}")

            print(f"📦 PyTorch version: {version('torch')}")
            print(f"📦 Diffusers version: {version('diffusers')}")

            self.pipe = self._load_pipeline(model_id)

            self.current_scheduler = "euler_a"
            self.schedulers = {
                "euler_a": ("Euler Ancestral", "Fast, creative"),
                "euler": ("Euler", "Deterministic"),
                "ddim": ("DDIM", "Classic"),
                "dpm_solver": ("DPM Solver", "High quality"),
                "lms": ("LMS", "Stable")
            }

            print("✅ Stable Diffusion Generator Ready!")

        except Exception as e:
            print(f"❌ Initialization Error: {str(e)}")
            raise

    # ==========================
    # Device Setup
    # ==========================
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                device = "cpu"
                print("💻 Using CPU")
        return torch.device(device)

    # ==========================
    # Load Pipeline
    # ==========================
    def _load_pipeline(self, model_id: str) -> StableDiffusionPipeline:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        print("🔧 Applying optimizations...")

        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✓ xFormers enabled")
        except Exception:
            print("⚠ xFormers not available")

        if self.device.type == "cuda":
            try:
                pipe = pipe.to(self.device)
                print("✓ Loaded on GPU")

                # 🔥 Performance boost
                pipe.unet = torch.compile(pipe.unet)

            except RuntimeError:
                print("⚠ VRAM low → CPU offload")
                pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()

        return pipe

    # ==========================
    # Scheduler
    # ==========================
    def set_scheduler(self, scheduler_name: str) -> bool:
        if scheduler_name not in self.schedulers:
            print(f"❌ Unknown scheduler: {scheduler_name}")
            return False

        if scheduler_name == self.current_scheduler:
            return True

        scheduler_map = {
            "euler_a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "ddim": DDIMScheduler,
            "dpm_solver": DPMSolverMultistepScheduler,
            "lms": LMSDiscreteScheduler
        }

        try:
            scheduler_class = scheduler_map[scheduler_name]
            self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)

            self.current_scheduler = scheduler_name
            print(f"🔄 Scheduler → {scheduler_name}")
            return True

        except Exception as e:
            print(f"❌ Scheduler Error: {e}")
            return False

    # ==========================
    # Generate Image
    # ==========================
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        scheduler: str = "euler_a",
        num_images: int = 1   # 🔥 NEW
    ) -> Tuple[list, dict]:

        prompt = prompt or ""
        negative_prompt = negative_prompt or ""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if scheduler != self.current_scheduler:
            self.set_scheduler(scheduler)

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        # 🔥 FIX: safer generator
        generator = torch.Generator(device=self.device.type).manual_seed(seed)

        width = (width // 8) * 8
        height = (height // 8) * 8

        print(f"\n🎨 Prompt: {prompt}")
        print(f"📏 {width}x{height} | Steps: {num_inference_steps} | CFG: {guidance_scale}")
        print(f"🎲 Seed: {seed} | Scheduler: {scheduler}")

        start_time = time.time()

        try:
            with torch.inference_mode():

                if self.device.type == "cuda":
                    with autocast("cuda"):   # 🔥 FIX
                        result = self.pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            num_images_per_prompt=num_images   # 🔥 NEW
                        )
                else:
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        num_images_per_prompt=num_images
                    )

            generation_time = time.time() - start_time

            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "seed": seed,
                "generation_time": round(generation_time, 2),
                "device": str(self.device),
                "dtype": str(self.dtype)
            }

            print(f"✅ Generated in {generation_time:.2f}s")

            return result.images, metadata   # 🔥 returns list now

        except torch.cuda.OutOfMemoryError:
            self._cleanup_memory()
            raise RuntimeError("❌ GPU OOM → reduce size or steps")

        finally:
            self._cleanup_memory()

    # ==========================
    # Cleanup
    # ==========================
    def _cleanup_memory(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ==========================
    # Memory Stats
    # ==========================
    def get_memory_usage(self) -> dict:
        if self.device.type == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        return {"device": "cpu"}

    # ==========================
    # Save Images (UPDATED)
    # ==========================
    def save_images(self, images, metadata, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = []

        for i, image in enumerate(images):
            filename = f"sd_{timestamp}_{i}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            paths.append(filepath)

        print(f"💾 Saved {len(paths)} images")
        return paths


# In[15]:


class StableDiffusionUI:

    def __init__(self):
        self.generator = None
        self.gallery_images = []
        self.generation_history = []

    # ==========================
    # Initialize Model
    # ==========================
    def initialize_generator(self, model_choice: str, device_choice: str) -> str:
        try:
            model_map = {
                "Stable Diffusion 1.5 (Recommended)": "runwayml/stable-diffusion-v1-5",
                "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
                "Realistic Vision (RealVisXL)": "SG161222/RealVisXL_V4.0"
            }

            device_map = {
                "Auto (Recommended)": "auto",
                "GPU (CUDA)": "cuda",
                "CPU (Slower)": "cpu"
            }

            model_id = model_map.get(model_choice)
            device = device_map.get(device_choice)

            self.generator = StableDiffusionGenerator(model_id=model_id, device=device)

            memory_info = self.generator.get_memory_usage()
            return f"✅ Model loaded!\n{memory_info}"

        except Exception as e:
            return f"❌ Initialization failed: {str(e)}"

    # ==========================
    # Generate Image
    # ==========================
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        scheduler: str,
        seed: int,
        save_image: bool,
        num_images: int   # 🔥 NEW
    ):

        if self.generator is None:
            return None, "❌ Initialize model first", "", []

        if not prompt.strip():
            return None, "❌ Prompt required", "", []

        try:
            seed = None if seed == -1 else int(seed)

            images, metadata = self.generator.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                scheduler=scheduler,
                seed=seed,
                num_images=num_images
            )

            info_text = self._format_generation_info(metadata)

            saved_path = ""
            if save_image:
                paths = self.generator.save_images(images, metadata)
                saved_path = "\n".join(paths)

            # Update gallery
            self.gallery_images.extend(images)
            self.gallery_images = self.gallery_images[-12:]

            return images[0], info_text, saved_path, self.gallery_images

        except Exception as e:
            return None, f"❌ Generation failed: {str(e)}", "", []

    # ==========================
    # Info Formatter
    # ==========================
    def _format_generation_info(self, metadata: dict) -> str:
        return f"""
✅ Generation Complete!

Prompt: {metadata['prompt']}
Size: {metadata['width']} x {metadata['height']}
Steps: {metadata['steps']}
CFG: {metadata['guidance_scale']}
Scheduler: {metadata['scheduler']}
Seed: {metadata['seed']}

Time: {metadata['generation_time']}s
Device: {metadata['device']}
"""

    # ==========================
    # Interface
    # ==========================
    def create_interface(self) -> gr.Blocks:

        with gr.Blocks(title="Stable Diffusion Generator") as interface:

            gr.Markdown("# 🎨 Stable Diffusion Generator")

            with gr.Row():

                # LEFT PANEL
                with gr.Column():

                    model_choice = gr.Dropdown(
                        ["Stable Diffusion 1.5 (Recommended)",
                         "Stable Diffusion 2.1",
                         "Realistic Vision (RealVisXL)"],
                        value="Stable Diffusion 1.5 (Recommended)",
                        label="Model"
                    )

                    device_choice = gr.Dropdown(
                        ["Auto (Recommended)", "GPU (CUDA)", "CPU (Slower)"],
                        value="Auto (Recommended)",
                        label="Device"
                    )

                    init_btn = gr.Button("Initialize Model")
                    init_status = gr.Textbox()

                    prompt = gr.Textbox(label="Prompt", lines=3)
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=2)

                    generate_btn = gr.Button("Generate", variant="primary")

                # RIGHT PANEL
                with gr.Column():

                    width = gr.Slider(256, 1024, 512, step=64, label="Width")
                    height = gr.Slider(256, 1024, 512, step=64, label="Height")

                    steps = gr.Slider(10, 50, 20, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 15.0, 7.5, step=0.5, label="CFG")

                    scheduler = gr.Dropdown(
                        ["euler_a", "euler", "ddim", "dpm_solver", "lms"],
                        value="euler_a",
                        label="Scheduler"
                    )

                    seed = gr.Number(-1, label="Seed (-1 = random)")
                    num_images = gr.Slider(1, 4, 1, step=1, label="Batch Size")  # 🔥 NEW

                    save_image = gr.Checkbox(True, label="Save Images")

            output_image = gr.Image(label="Output")
            gallery = gr.Gallery(label="Gallery", columns=4)

            generation_info = gr.Textbox(lines=8)
            saved_path = gr.Textbox()

            # ==========================
            # EVENTS
            # ==========================
            init_btn.click(
                self.initialize_generator,
                inputs=[model_choice, device_choice],
                outputs=init_status
            )

            generate_btn.click(
                self.generate_image,
                inputs=[
                    prompt, negative_prompt, width, height,
                    steps, guidance, scheduler, seed,
                    save_image, num_images
                ],
                outputs=[
                    output_image,
                    generation_info,
                    saved_path,
                    gallery
                ]
            )

        return interface


# In[16]:


ui = StableDiffusionUI()
interface = ui.create_interface()

interface.queue()  # Important for handling multiple requests safely

interface.launch(
    share=True,              # Use True only in Colab / remote access
    server_name="0.0.0.0",   # Required for external access
    debug=False,
    show_error=True
)

