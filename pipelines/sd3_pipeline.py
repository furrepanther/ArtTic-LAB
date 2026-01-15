import torch
from diffusers import StableDiffusion3Pipeline
from .base_pipeline import ArtTicPipeline
import logging

logger = logging.getLogger("arttic_lab")

SD3_BASE_MODEL_REPO = "stabilityai/stable-diffusion-3-medium-diffusers"


class SD3Pipeline(ArtTicPipeline):
    def load_pipeline(self, progress, use_quantization=False):
        progress(0.2, "Loading base SD3 components from Hugging Face...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            SD3_BASE_MODEL_REPO,
            torch_dtype=self.dtype,
            use_safetensors=True,
            progress_bar_config={"disable": True},
        )

        progress(0.5, "Injecting local model weights...")
        self.pipe.load_lora_weights(self.model_path)
