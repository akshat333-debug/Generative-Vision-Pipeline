#!/usr/bin/env python
# coding: utf-8
# Comprehensive Text-to-Image Pipeline (Task 6)
# Integrates Stable Diffusion with LoRA and a Custom Self-Attention CGAN
# using a Gradio interface.

# Colab Environment Setup
import os
import sys
if not os.path.exists('scripts'):
    print("Warning: 'scripts/' not found. If running in Colab, clone the repo first.")

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import autocast
import numpy as np
from PIL import Image
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


# Removed old standalone generators in favor of core_pipeline.py 
# to satisfy Task 6 (Unified End-to-End Pipeline)


# Unified Gradio Pipeline UI
# Combines SD with CGAN and NLP text embeddings

import torchvision.transforms as T

# Ensure modules are discoverable locally and in Colab envs
sys.path.append(os.path.abspath('.'))

try:
    from core_pipeline import HybridGenerativePipeline
    PIPELINE_AVAILABLE = True
except Exception as e:
    print("Integration Warning: core_pipeline.py not instantly reachable. Error:", e)
    PIPELINE_AVAILABLE = False


class IntegratedGeneratorUI:
    """
    Unified Gradio interface that acts simply as a UI wrapper around the 
    core_pipeline.HybridGenerativePipeline production class.
    """

    def __init__(self):
        self.pipeline = None
        self.gallery_images = []

    def initialize_system(self, engine_choice: str, model_choice: str, device_choice: str):
        """Initialize the selected inference engine via core_pipeline."""
        try:
            device = "cuda" if torch.cuda.is_available() and "CPU" not in device_choice else "cpu"
            
            if not PIPELINE_AVAILABLE:
                return "❌ core_pipeline.py module not found in environment root."
                
            self.pipeline = HybridGenerativePipeline(device=device)

            if engine_choice == "Custom CGAN (Shapes)":
                self.pipeline.load_cgan()
                return "✅ Custom CGAN Pipeline Initialized (Text Tokenization + Cross-Attention GAN Linked)!"
            else:
                model_map = {
                    "Stable Diffusion 1.5 (Recommended)": "runwayml/stable-diffusion-v1-5",
                    "DreamShaper 8 (Open Alternative)": "Lykon/dreamshaper-8",
                    "Realistic Vision (SD 1.5)": "SG161222/Realistic_Vision_V5.1_noVAE"
                }

                model_id = model_map.get(model_choice, "runwayml/stable-diffusion-v1-5")
                self.pipeline.load_stable_diffusion(model_id=model_id)
                return "✅ Stable Diffusion Pipeline loaded successfully!"

        except Exception as e:
            return f"❌ Initialization failed: {str(e)}"

    def generate(self, engine, prompt, neg_prompt, width, height, steps, cfg, backend, seed, num_imgs):
        """Route generation to the selected engine via core_pipeline."""
        if self.pipeline is None:
            return None, "❌ Initialize Pipeline First", []
            
        try:
            # The core_pipeline hides all the architectural complexity and acts
            # as a pure unified text-to-image API.
            is_cgan = engine == "Custom CGAN (Shapes)"
            engine_arg = "cgan" if is_cgan else "sd"
            
            final_img = self.pipeline.generate(
                prompt=prompt,
                engine=engine_arg,
                width=int(width),
                height=int(height),
                steps=int(steps),
                cfg=float(cfg)
            )
            
            self.gallery_images.append(final_img)
            return final_img, f"✅ Generation Completed: {prompt}", self.gallery_images
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}", []

    def create_interface(self):
        """Build the unified Gradio Blocks interface."""
        with gr.Blocks(title="Unified Text-to-Image Pipeline") as interface:
            gr.Markdown("# 🚀 Comprehensive Hybrid Text-to-Image Pipeline (Task 6)")
            gr.Markdown(
                "Select **Stable Diffusion** for heavily aesthetic generation, "
                "or **Custom CGAN** to parse basic geometric labels using our "
                "internal NLP extractor and GAN models."
            )

            with gr.Row():
                with gr.Column():
                    engine = gr.Radio(
                        ["Stable Diffusion (HuggingFace)", "Custom CGAN (Shapes)"],
                        value="Stable Diffusion (HuggingFace)",
                        label="Inference Engine"
                    )
                    model = gr.Dropdown(
                        ["Stable Diffusion 1.5 (Recommended)",
                         "DreamShaper 8 (Open Alternative)",
                         "Realistic Vision (SD 1.5)"],
                        value="Stable Diffusion 1.5 (Recommended)",
                        label="SD Model"
                    )
                    dev = gr.Dropdown(
                        ["Auto (GPU)", "CPU"],
                        value="Auto (GPU)",
                        label="Hardware Backend"
                    )
                    init_btn = gr.Button("Initialize Selected Pipeline")
                    status = gr.Textbox(label="System Status")

                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter detailed prompt or 'circle'/'square' for CGAN..."
                    )
                    neg = gr.Textbox(label="Negative Prompt")
                    gen_btn = gr.Button("Generate Matrix", variant="primary")

                with gr.Column():
                    w = gr.Slider(64, 1024, 512, step=64, label="Width")
                    h = gr.Slider(64, 1024, 512, step=64, label="Height")
                    steps = gr.Slider(10, 50, 20, step=1, label="Steps")
                    cfg = gr.Slider(1.0, 15.0, 7.5, label="CFG Scale")
                    sched = gr.Dropdown(
                        ["euler_a", "ddim", "dpm_solver", "lms"],
                        value="euler_a",
                        label="Scheduler"
                    )
                    seed = gr.Number(-1, label="Seed Matrix")
                    batch = gr.Slider(1, 4, 1, step=1, label="Batch Size")

            with gr.Row():
                out = gr.Image(label="Active Render")
                info = gr.Textbox(label="Processing Details")

            gal = gr.Gallery(label="Session Gallery")

            # Route UI events
            init_btn.click(self.initialize_system, [engine, model, dev], status)
            gen_btn.click(
                self.generate,
                [engine, prompt, neg, w, h, steps, cfg, sched, seed, batch],
                [out, info, gal]
            )

        return interface


# Application Entry Point

if __name__ == "__main__":
    ui = IntegratedGeneratorUI()
    interface = ui.create_interface()
    interface.launch(share=True)
