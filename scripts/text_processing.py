import torch
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List, Tuple

class TextEmbedder:
    """
    A unified software module to preprocess text descriptions into
    tokenized and encoded representations.
    
    This fulfills Task 3 by utilizing Hugging Face Transformers.
    By default, it uses the OpenAI CLIP model which is standard for 
    Stable Diffusion implementations.
    """
    
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        Initializes the tokenizer and text encoder model.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading tokenizer and text encoder: {model_id} on {self.device}")
        
        # Disable SDPA to prevent CUDA misaligned address errors on certain architectures
        import os
        os.environ["TORCH_DISABLE_SDPA"] = "1"
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        # Use explicit mapping to the device and specify float32 as a fallback
        self.text_model = CLIPTextModel.from_pretrained(model_id, torch_dtype=torch.float32).to(self.device)
        self.text_model.eval()
        
    def tokenize_text(self, prompts: List[str]) -> dict:
        """
        Converts text descriptions into tokenized inputs (input_ids and attention_mask).
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        return tokens
        
    @torch.no_grad()
    def get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """
        Generates encoded embedding representations from text prompts.
        These continuous vectors are fed directly into conditional inputs (like CGANs).
        """
        tokens = self.tokenize_text(prompts)
        
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # We return the pooled output (for generic latent conditioning) 
        # or the last_hidden_state (for cross-attention mechanisms).
        # We'll return last_hidden_state as it's required for advanced GANs/Diffusion.
        return output.last_hidden_state

if __name__ == "__main__":
    # Quick sanity check
    embedder = TextEmbedder()
    test_prompts = ["a beautiful sunset over the mountains", "a geometric circle"]
    embeddings = embedder.get_text_embeddings(test_prompts)
    print(f"Output embeddings shape: {embeddings.shape}")
    print("Embedding generation successful.")
