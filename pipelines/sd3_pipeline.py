import torch
from diffusers import StableDiffusion3Pipeline
from .base_pipeline import ArtTicPipeline, CPUTextEncoderWrapper
import logging

logger = logging.getLogger("arttic_lab")

SD3_BASE_MODEL_REPO = "stabilityai/stable-diffusion-3-medium-diffusers"


class SD3Pipeline(ArtTicPipeline):
    def load_pipeline(self, progress):
        progress(0.2, "Loading base SD3 components from Hugging Face...")
        try:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                SD3_BASE_MODEL_REPO,
                torch_dtype=self.dtype,
                use_safetensors=True,
                progress_bar_config={"disable": True},
            )
        except Exception as e:
            logger.error(
                f"Failed to download SD3 base model. Check internet connection. Error: {e}"
            )
            raise RuntimeError(
                "Could not download base SD3 components from Hugging Face."
            )

        progress(0.5, "Injecting local model weights...")
        self.pipe.load_lora_weights(self.model_path)
        logger.info(f"Successfully injected weights from '{self.model_path}'")

    def _wrap_text_encoders_for_xpu(self):
        logger.info("XPU Strategy: Wrapping SD3 Text Encoders to run on CPU (Stable).")
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder:
            self.pipe.text_encoder = CPUTextEncoderWrapper(self.pipe.text_encoder)
        if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2:
            self.pipe.text_encoder_2 = CPUTextEncoderWrapper(self.pipe.text_encoder_2)
        if hasattr(self.pipe, "text_encoder_3") and self.pipe.text_encoder_3:
            self.pipe.text_encoder_3 = CPUTextEncoderWrapper(self.pipe.text_encoder_3)