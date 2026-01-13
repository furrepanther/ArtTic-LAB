from diffusers import StableDiffusionXLPipeline
from .base_pipeline import ArtTicPipeline, CPUTextEncoderWrapper
import logging

try:
    from sdnq.loader import apply_sdnq_options_to_model
    SDNQ_AVAILABLE = True
except ImportError:
    SDNQ_AVAILABLE = False

logger = logging.getLogger("arttic_lab")


class SDXLPipeline(ArtTicPipeline):
    def load_pipeline(self, progress):
        progress(0.2, "Loading StableDiffusionXLPipeline...")
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            self.model_path,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16",
            safety_checker=None,
            progress_bar_config={"disable": True},
        )

        if SDNQ_AVAILABLE and hasattr(self.pipe, "unet"):
            logger.info("Applying SDNQ quantization optimizations to SDXL UNet...")
            self.pipe.unet = apply_sdnq_options_to_model(
                self.pipe.unet, use_quantized_matmul=True
            )

    def _wrap_text_encoders_for_xpu(self):
        logger.info("XPU Strategy: Wrapping SDXL Text Encoders to run on CPU (Stable).")
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder:
            self.pipe.text_encoder = CPUTextEncoderWrapper(self.pipe.text_encoder)
        if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2:
            self.pipe.text_encoder_2 = CPUTextEncoderWrapper(self.pipe.text_encoder_2)