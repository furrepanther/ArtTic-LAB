import torch
import logging
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

logger = logging.getLogger("arttic_lab")

# Keep reference to the original static method
_original_make_causal_mask = AttentionMaskConverter._make_causal_mask


def _patched_make_causal_mask(
    input_ids_shape,
    dtype,
    device,
    past_key_values_length: int = 0,
    sliding_window=None,
):
    """
    Patched version of AttentionMaskConverter._make_causal_mask.
    Forces mask generation on CPU for Intel XPU to avoid driver crashes.
    Note: This must be registered as a staticmethod.
    """
    target_device = device
    if isinstance(device, str):
        target_device = torch.device(device)

    # Check if target is XPU
    if hasattr(target_device, "type") and target_device.type == "xpu":
        # Generate on CPU (stable)
        # We do NOT pass 'self' here because the original is a static method
        mask = _original_make_causal_mask(
            input_ids_shape,
            dtype,
            torch.device("cpu"),
            past_key_values_length,
            sliding_window,
        )
        # Move result to XPU (safe for Float tensors)
        return mask.to(target_device)

    # Default behavior for other devices
    return _original_make_causal_mask(
        input_ids_shape,
        dtype,
        device,
        past_key_values_length,
        sliding_window,
    )


def apply_xpu_patch():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        logger.info("Applying Intel XPU Patch for Transformers Attention Masks...")
        # Patch as a static method explicitly to match original signature/behavior
        AttentionMaskConverter._make_causal_mask = staticmethod(
            _patched_make_causal_mask
        )