from diffusers import StableDiffusionPipeline
from .base_pipeline import ArtTicPipeline, CPUTextEncoderWrapper
import logging

try:
    from sdnq.loader import apply_sdnq_options_to_model
    SDNQ_AVAILABLE = True
except ImportError:
    SDNQ_AVAILABLE = False

logger = logging.getLogger("arttic_lab")


class SD2Pipeline(ArtTicPipeline):
    def load_pipeline(self, progress):
        progress(0.2, "Loading StableDiffusionPipeline (v2)...")
        self.pipe = StableDiffusionPipeline.from_single_file(
            self.model_path,
            torch_dtype=self.dtype,
            use_safetensors=True,
            safety_checker=None,
            progress_bar_config={"disable": True},
        )

        if SDNQ_AVAILABLE and hasattr(self.pipe, "unet"):
            logger.info("Applying SDNQ quantization optimizations to SD2 UNet...")
            self.pipe.unet = apply_sdnq_options_to_model(
                self.pipe.unet, use_quantized_matmul=True
            )

    def _wrap_text_encoders_for_xpu(self):
        if hasattr(self.pipe, "text_encoder"):
            logger.info("XPU Strategy: Wrapping Text Encoder to run on CPU.")
            self.pipe.text_encoder = CPUTextEncoderWrapper(self.pipe.text_encoder)