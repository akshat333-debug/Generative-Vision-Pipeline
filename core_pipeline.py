import torch
import os
from PIL import Image

from scripts.text_processing import TextEmbedder
from models.cgan_attention import ConditionalGenerator
from diffusers import StableDiffusionPipeline

class HybridGenerativePipeline:
    """
    A unified, production-grade end-to-end pipeline that routes raw text prompts 
    through Hugging Face tokenization and generates images using either:
      1. A from-scratch Cross-Attention CGAN
      2. A LoRA fine-tuned Stable Diffusion model
      
    This directly integrates the NLP Embedding sequence into the generation cycle,
    satisfying end-to-end production workflow requirements.
    """
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.text_embedder = None
        self.cgan = None
        self.sd_pipe = None
        
    def load_cgan(self, cgan_weights_path="cgan_generator.pth"):
        """Loads the NLP text embedder and CGAN into memory."""
        print("Loading TextEmbedder (OpenAI CLIP)...")
        self.text_embedder = TextEmbedder(device=self.device)
        
        print("Loading Cross-Attention CGAN...")
        self.cgan = ConditionalGenerator().to(self.device)
        
        if os.path.exists(cgan_weights_path):
            self.cgan.load_state_dict(torch.load(cgan_weights_path, map_location=self.device))
            print("✅ CGAN weights loaded successfully.")
        else:
            print("⚠ CGAN weights not found, using raw initialized weights.")
        self.cgan.eval()
        
    def load_stable_diffusion(self, model_id="runwayml/stable-diffusion-v1-5", lora_weights_dir="lora_unet_weights"):
        """Loads Stable Diffusion with optional LoRA injection."""
        print(f"Loading Stable Diffusion: {model_id}...")
        
        # CPU requires float32, GPU handles float16 faster
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False
        )
        self.sd_pipe = self.sd_pipe.to(self.device)
        
        if os.path.exists(lora_weights_dir):
            print("Injecting Custom LoRA Weights...")
            self.sd_pipe.load_lora_weights(lora_weights_dir, weight_name='adapter_model.safetensors')
            print("✅ Stable Diffusion + LoRA loaded.")
            
    def generate(self, prompt: str, engine: str = "cgan", width=512, height=512, steps=20, cfg=7.5) -> Image.Image:
        """
        Unified generation method matching Hugging Face pipeline design patterns.
        engine: 'cgan' or 'sd'
        """
        if engine == "cgan":
            if self.cgan is None or self.text_embedder is None:
                raise RuntimeError("CGAN or TextEmbedder not loaded. Call load_cgan() first.")
                
            with torch.no_grad():
                # 1. NLP Pipeline: Raw text -> Full sequence embedding (Batch, SeqLen, EmbedDim)
                text_sequence = self.text_embedder.get_text_embeddings([prompt])
                
                # 2. Noise Generation
                noise = torch.randn(1, 100, 1, 1, device=self.device)
                
                # 3. CGAN Cross-Attention Generation (End-to-End Link)
                fake = self.cgan(noise, text_sequence)
                
                # Format to image
                rendered = ((fake[0].cpu().permute(1, 2, 0) + 1.0) / 2.0).clamp(0, 1)
                img = (rendered.numpy() * 255).astype("uint8")
                img = Image.fromarray(img).resize((width, height), Image.BICUBIC)
                
                return img
                
        elif engine == "sd":
            if self.sd_pipe is None:
                raise RuntimeError("Stable Diffusion not loaded. Call load_stable_diffusion() first.")
                
            img = self.sd_pipe(
                prompt=prompt, 
                width=width, 
                height=height, 
                num_inference_steps=steps, 
                guidance_scale=cfg
            ).images[0]
            
            return img
            
        else:
            raise ValueError(f"Unknown engine: {engine}")

if __name__ == "__main__":
    print("Testing HybridGenerativePipeline...")
    pipeline = HybridGenerativePipeline(device="cpu")
    pipeline.load_cgan()
    
    img = pipeline.generate("circle", engine="cgan")
    print(f"Success! CGAN Image generated with shape: {img.size}")
