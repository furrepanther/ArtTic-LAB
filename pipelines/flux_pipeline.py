import torch
import logging
import os
import requests
from diffusers import FluxPipeline, FluxTransformer2DModel
from huggingface_hub.errors import GatedRepoError
from .base_pipeline import ArtTicPipeline, CPUTextEncoderWrapper

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

    def load_pipeline(self, progress):
        if self.is_schnell:
            repo_id = FLUX_SCHNELL_BASE_REPO
            desc = "Loading FLUX.1 Schnell components..."
        else:
            repo_id = FLUX_DEV_BASE_REPO
            desc = "Loading FLUX.1 DEV components..."

        progress(0.2, desc)

        try:
            logger.info(f"Loading transformer from local file: {self.model_path}")
            transformer = FluxTransformer2DModel.from_single_file(
                self.model_path, torch_dtype=self.dtype
            )

            # Apply SDNQ Optimization if available
            if SDNQ_AVAILABLE:
                logger.info(
                    "Applying SDNQ quantization optimizations to FLUX Transformer..."
                )
                transformer = apply_sdnq_options_to_model(
                    transformer, use_quantized_matmul=True
                )

            logger.info("Local transformer loaded successfully.")

            progress(0.4, f"Loading remaining components from {repo_id}...")
            self.pipe = FluxPipeline.from_pretrained(
                repo_id,
                transformer=transformer,
                torch_dtype=self.dtype,
                use_safetensors=True,
                progress_bar_config={"disable": True},
            )

            logger.info("Pipeline constructed with local transformer.")

        except GatedRepoError as e:
            logger.error(
                "Hugging Face Gated Repo Error: User needs to be logged in and have accepted the license for FLUX models."
            )
            raise RuntimeError(
                "Access to FLUX base model is restricted. Please run 'huggingface-cli login' "
                "and ensure you have accepted the license for 'black-forest-labs/FLUX.1-dev' on the Hugging Face website."
            ) from e
        except (requests.exceptions.RequestException, FileNotFoundError) as e:
            logger.error(
                f"Failed to download FLUX components from '{repo_id}'. This is likely due to a network issue or a corrupted cache. Error: {e}"
            )
            raise RuntimeError(
                "Could not download base FLUX components. Check internet connection or clear Hugging Face cache."
            ) from e

        model_type = "Schnell" if self.is_schnell else "DEV"
        logger.info(
            f"Successfully loaded FLUX {model_type} model '{os.path.basename(self.model_path)}'"
        )

    def _wrap_text_encoders_for_xpu(self):
        logger.info("XPU Strategy: Wrapping FLUX Text Encoders to run on CPU (Stable).")
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder:
            self.pipe.text_encoder = CPUTextEncoderWrapper(self.pipe.text_encoder)
        if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2:
            self.pipe.text_encoder_2 = CPUTextEncoderWrapper(self.pipe.text_encoder_2)

    def generate(self, *args, **kwargs):
        if self.is_schnell and "negative_prompt" in kwargs:
            logger.info(
                "FLUX Schnell does not use a negative prompt. It will be ignored."
            )
            kwargs.pop("negative_prompt")
        return super().generate(*args, **kwargs)