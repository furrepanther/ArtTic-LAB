from diffusers import StableDiffusionPipeline
import torch
from .base_pipeline import ArtTicPipeline
import logging

try:
    from sdnq.loader import apply_sdnq_options_to_model

    SDNQ_AVAILABLE = True
except ImportError:
    SDNQ_AVAILABLE = False

logger = logging.getLogger("arttic_lab")


class SD15Pipeline(ArtTicPipeline):
    def load_pipeline(self, progress, use_quantization=False):
        logger.info("Initializing StableDiffusionPipeline (v1.5)...")
        progress(0.2, "Loading StableDiffusionPipeline (v1.5)...")

        self.pipe = StableDiffusionPipeline.from_single_file(
            self.model_path,
            torch_dtype=self.dtype,
            use_safetensors=True,
            safety_checker=None,
            # Progress bar enabled by default per user request
        )
        logger.info("Base pipeline loaded.")

        if use_quantization and SDNQ_AVAILABLE and hasattr(self.pipe, "unet"):
            # Determine if we can use Triton (CUDA only usually)
            use_triton = torch.cuda.is_available()
            mode_str = (
                "Triton Kernels"
                if use_triton
                else "Standard Ops (Storage Optimization)"
            )

            logger.info(f"Applying SDNQ to SD1.5 UNet using {mode_str}...")

            try:
                self.pipe.unet = apply_sdnq_options_to_model(
                    self.pipe.unet, use_quantized_matmul=use_triton
                )
                logger.info(f"SDNQ quantization applied successfully via {mode_str}.")
            except Exception as e:
                logger.warning(f"SDNQ failed: {e}")
                logger.warning("Proceeding with standard precision.")

        elif use_quantization and not SDNQ_AVAILABLE:
            logger.warning("SDNQ requested but 'sdnq' library not found.")
