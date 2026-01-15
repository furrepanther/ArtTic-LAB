import os
from safetensors import safe_open
from .sd15_pipeline import SD15Pipeline
from .sd2_pipeline import SD2Pipeline
from .sdxl_pipeline import SDXLPipeline
from .sd3_pipeline import SD3Pipeline
from .flux_pipeline import ArtTicFLUXPipeline
import logging

logger = logging.getLogger("arttic_lab")
MODELS_DIR = "./models"


def _is_flux(keys):
    return (
        any("transformer." in k for k in keys)
        or any("double_blocks." in k for k in keys)
    ) and not (any("input_blocks" in k for k in keys))


def _is_sd3(keys):
    return any(k.startswith("text_encoders.") for k in keys)


def _is_xl(keys):
    return any(k.startswith("conditioner.embedders.1") for k in keys)


def _is_v2(keys):
    return (
        "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight"
        in keys
    )


def get_pipeline_for_model(model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.safetensors")
    try:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception as e:
        logger.error(f"Inspect failed '{model_name}': {e}. Defaulting SD1.5")
        return SD15Pipeline(model_path)

    if _is_sd3(keys):
        return SD3Pipeline(model_path)
    elif _is_xl(keys):
        return SDXLPipeline(model_path)
    elif _is_flux(keys):
        return ArtTicFLUXPipeline(
            model_path, is_schnell="schnell" in model_name.lower()
        )
    elif _is_v2(keys):
        return SD2Pipeline(model_path)
    else:
        return SD15Pipeline(model_path)
