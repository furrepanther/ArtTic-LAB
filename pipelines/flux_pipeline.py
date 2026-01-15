import torch
import logging
from diffusers import FluxPipeline, FluxTransformer2DModel
from .base_pipeline import ArtTicPipeline

try:
    from sdnq.loader import apply_sdnq_options_to_model

    SDNQ_AVAILABLE = True
except ImportError:
    SDNQ_AVAILABLE = False

logger = logging.getLogger("arttic_lab")

FLUX_DEV_BASE_REPO = "black-forest-labs/FLUX.1-dev"
FLUX_SCHNELL_BASE_REPO = "black-forest-labs/FLUX.1-schnell"


class ArtTicFLUXPipeline(ArtTicPipeline):
    def __init__(self, model_path, dtype=torch.bfloat16, is_schnell=False):
        super().__init__(model_path, dtype)
        self.is_schnell = is_schnell

    def load_pipeline(self, progress, use_quantization=False):
        repo_id = FLUX_SCHNELL_BASE_REPO if self.is_schnell else FLUX_DEV_BASE_REPO
        logger.info(
            f"Initializing FLUX Pipeline ({'Schnell' if self.is_schnell else 'Dev'})..."
        )
        progress(0.2, f"Loading FLUX {'Schnell' if self.is_schnell else 'Dev'}...")

        transformer = FluxTransformer2DModel.from_single_file(
            self.model_path, torch_dtype=self.dtype
        )
        logger.info("Transformer loaded from local file.")

        if use_quantization and SDNQ_AVAILABLE:
            use_triton = torch.cuda.is_available()
            mode_str = (
                "Triton Kernels"
                if use_triton
                else "Standard Ops (Storage Optimization)"
            )

            logger.info(f"Applying SDNQ to FLUX Transformer using {mode_str}...")
            try:
                transformer = apply_sdnq_options_to_model(
                    transformer, use_quantized_matmul=use_triton
                )
                logger.info(f"SDNQ quantization applied successfully via {mode_str}.")
            except Exception as e:
                logger.warning(f"SDNQ failed: {e}")
                logger.warning("Proceeding with standard precision.")
        elif use_quantization and not SDNQ_AVAILABLE:
            logger.warning("SDNQ requested but 'sdnq' library not found.")

        logger.info(f"Loading remaining components from HuggingFace ({repo_id})...")
        self.pipe = FluxPipeline.from_pretrained(
            repo_id,
            transformer=transformer,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        logger.info("FLUX Pipeline assembled.")
