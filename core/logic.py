import torch
import os
import time
import random
import logging
import math
import re
import asyncio
import sys
from glob import glob
from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
)
from pipelines import get_pipeline_for_model
from .prompt_book import prompt_book
from .metadata_handler import metadata_handler
from pyngrok import ngrok


class OOMError(Exception):
    pass


app_state = {
    "current_pipe": None,
    "current_model_name": "",
    "current_lora_name": "",
    "is_model_loaded": False,
    "status_message": "No model loaded.",
    "current_cpu_offload_state": False,
    "current_vae_tiling_state": True,
    "current_sdnq_state": False,
    "current_model_type": "",
    "default_width": 512,
    "default_height": 512,
    "public_url": None,
}

APP_LOGGER_NAME = "arttic_lab"
RESTART_EXIT_CODE = 21
logger = logging.getLogger(APP_LOGGER_NAME)

SCHEDULER_MAP = {
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DDIM": DDIMScheduler,
    "UniPC": UniPCMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "LMS": LMSDiscreteScheduler,
}


def get_app_status():
    return {
        "is_model_loaded": app_state["is_model_loaded"],
        "status_message": app_state["status_message"],
        "public_url": app_state["public_url"],
    }


def get_config():
    return {
        "models": get_available_models(),
        "loras": get_available_loras(),
        "schedulers": list(SCHEDULER_MAP.keys()),
        "gallery_images": get_output_images(),
        "prompts": prompt_book.get_all_prompts(),
        "public_url": app_state["public_url"],
    }


def get_available_models():
    models_path = os.path.join("./models", "*.safetensors")
    return sorted([os.path.splitext(os.path.basename(p))[0] for p in glob(models_path)])


def get_available_loras():
    os.makedirs("./loras", exist_ok=True)
    loras_path = os.path.join("./loras", "*.safetensors")
    return sorted([os.path.splitext(os.path.basename(p))[0] for p in glob(loras_path)])


def get_output_images():
    outputs_path = os.path.join("./outputs", "*.png")
    image_files = []
    for f in sorted(glob(outputs_path), key=os.path.getmtime, reverse=True):
        filename = os.path.basename(f)
        metadata = metadata_handler.extract_metadata_from_image(f)
        image_info = {"filename": filename, "has_metadata": metadata is not None}
        if metadata:
            image_info.update(
                {
                    "prompt_preview": (
                        (metadata.get("prompt", "")[:50] + "...")
                        if len(metadata.get("prompt", "")) > 50
                        else metadata.get("prompt", "")
                    ),
                    "model_name": metadata.get("model_name", ""),
                    "timestamp_generation": metadata.get("timestamp_generation", ""),
                }
            )
        image_files.append(image_info)
    return image_files


def get_model_files():
    return sorted([os.path.basename(f) for f in glob("./models/*.safetensors")])


def get_lora_files():
    return sorted([os.path.basename(f) for f in glob("./loras/*.safetensors")])


def delete_model_file(filename):
    if not filename:
        raise ValueError("Filename cannot be empty.")
    models_dir = os.path.abspath("./models")
    file_path = os.path.abspath(os.path.join(models_dir, filename))
    if os.path.commonpath([file_path, models_dir]) != models_dir:
        raise PermissionError("Cannot delete files outside of models directory.")
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Deleted model file: {filename}")
        return {"status": "success", "message": f"Deleted {filename}"}
    return {"status": "error", "message": "File not found"}


def delete_lora_file(filename):
    if not filename:
        raise ValueError("Filename cannot be empty.")
    loras_dir = os.path.abspath("./loras")
    file_path = os.path.abspath(os.path.join(loras_dir, filename))
    if os.path.commonpath([file_path, loras_dir]) != loras_dir:
        raise PermissionError("Cannot delete files outside of loras directory.")
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Deleted LoRA file: {filename}")
        return {"status": "success", "message": f"Deleted {filename}"}
    return {"status": "error", "message": "File not found"}


def _get_next_image_number():
    files = get_output_images()
    if not files:
        return 1
    highest_num = 0
    pattern = re.compile(r"ArtTic-LAB_(\d+)\.png")
    for f in files:
        match = pattern.match(f["filename"])
        if match:
            num = int(match.group(1))
            if num > highest_num:
                highest_num = num
    return highest_num + 1


def delete_image(filename):
    if not filename:
        raise ValueError("Filename cannot be empty.")
    outputs_dir = os.path.abspath("./outputs")
    file_path = os.path.abspath(os.path.join(outputs_dir, filename))
    if os.path.commonpath([file_path, outputs_dir]) != outputs_dir:
        raise PermissionError("Cannot delete files outside of outputs directory.")
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Deleted output image: {filename}")
        return {"status": "success", "message": f"Deleted {filename}"}
    return {"status": "error", "message": "File not found"}


def unload_model():
    if not app_state["is_model_loaded"]:
        return {"status_message": "No model loaded."}

    logger.info("Unloading current model and clearing VRAM...")
    if app_state["current_pipe"]:
        del app_state["current_pipe"]

    app_state.update(
        {
            "current_pipe": None,
            "current_model_name": "",
            "current_lora_name": "",
            "is_model_loaded": False,
            "status_message": "No model loaded.",
            "current_model_type": "",
            "current_sdnq_state": False,
            "current_cpu_offload_state": False,
            "current_vae_tiling_state": True,
            "default_width": 512,
            "default_height": 512,
        }
    )

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Unload complete.")
    return {"status_message": app_state["status_message"]}


def _calculate_max_resolution(model_type):
    GB = 1024**3
    free_mem = 0
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(0)
        free_mem = (props.total_memory - torch.xpu.memory_reserved(0)) / GB
    elif torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_mem = (props.total_memory - torch.cuda.memory_reserved(0)) / GB
    else:
        return 1024

    vram_per_mp = {
        "SD 1.5": 0.9,
        "SD 2.x": 1.2,
        "SDXL": 2.5,
        "SD3": 3.0,
        "FLUX Dev": 3.2,
        "FLUX Schnell": 2.8,
    }.get(model_type, 1.5)

    base_res_mp = 0.26 if model_type == "SD 1.5" else 1.05
    effective_mem = max(0, free_mem - 0.5)

    try:
        max_add_mp = effective_mem / vram_per_mp
        total_mp = base_res_mp + max_add_mp
        side = math.sqrt(total_mp * 1024 * 1024)
        return max(512, min(4096, int(side // 64 * 64)))
    except:
        return 1024


def load_model(
    model_name,
    scheduler_name,
    vae_tiling,
    cpu_offload,
    sdnq_enabled,
    lora_name,
    progress_callback=None,
    loop=None,
):
    def update_progress(p, d):
        if progress_callback and loop:
            asyncio.run_coroutine_threadsafe(progress_callback(p, d), loop)

    try:
        unload_model()
        logger.info(f"Starting load sequence for: {model_name}")
        update_progress(0.1, f"Loading {model_name}...")

        pipe_obj = get_pipeline_for_model(model_name)
        pipe_obj.load_pipeline(update_progress, use_quantization=sdnq_enabled)

        if scheduler_name in SCHEDULER_MAP and hasattr(pipe_obj.pipe, "scheduler"):
            logger.info(f"Setting scheduler to {scheduler_name}")
            pipe_obj.pipe.scheduler = SCHEDULER_MAP[scheduler_name].from_config(
                pipe_obj.pipe.scheduler.config
            )

        pipe_obj.place_on_device(use_cpu_offload=cpu_offload)

        if vae_tiling and hasattr(pipe_obj.pipe, "vae"):
            logger.info("Enabling VAE Tiling/Slicing.")
            pipe_obj.pipe.vae.enable_tiling()
            pipe_obj.pipe.vae.enable_slicing()

        if lora_name and lora_name != "None":
            lora_path = os.path.join("./loras", f"{lora_name}.safetensors")
            if os.path.exists(lora_path):
                logger.info(f"Loading LoRA: {lora_name}")
                pipe_obj.pipe.load_lora_weights(lora_path)
                app_state["current_lora_name"] = lora_name

        # Determine defaults based on class
        model_type = "SDXL"  # Fallback
        if "FLUX" in pipe_obj.__class__.__name__:
            model_type = "FLUX Schnell" if pipe_obj.is_schnell else "FLUX Dev"
        elif "SD3" in pipe_obj.__class__.__name__:
            model_type = "SD3"
        elif "SD2" in pipe_obj.__class__.__name__:
            model_type = "SD 2.x"
        elif "SD15" in pipe_obj.__class__.__name__:
            model_type = "SD 1.5"

        default_res = (
            512 if model_type == "SD 1.5" else 1024 if model_type != "SD 2.x" else 768
        )

        app_state.update(
            {
                "current_pipe": pipe_obj,
                "current_model_name": model_name,
                "is_model_loaded": True,
                "status_message": f"Ready: {model_name} ({model_type})",
                "current_model_type": model_type,
                "current_vae_tiling_state": vae_tiling,
                "current_cpu_offload_state": cpu_offload,
                "current_sdnq_state": sdnq_enabled,
                "default_width": default_res,
                "default_height": default_res,
            }
        )

        logger.info(f"Load complete. Model Ready.")
        update_progress(1.0, "Model Ready")
        return {
            "status_message": app_state["status_message"],
            "max_res_vram": _calculate_max_resolution(model_type),
            "width": default_res,
            "height": default_res,
            "model_type": model_type,
        }
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        unload_model()
        raise


def generate_image(
    prompt,
    negative_prompt,
    steps,
    guidance,
    seed,
    width,
    height,
    lora_weight,
    progress_callback=None,
    loop=None,
    **kwargs,
):
    if not app_state["is_model_loaded"]:
        raise RuntimeError("No model loaded.")

    seed = int(seed if seed >= 0 else random.randint(0, 2**32 - 1))
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def pipe_cb(pipe, step, timestep, callback_kwargs):
        if progress_callback and loop:
            asyncio.run_coroutine_threadsafe(
                progress_callback(step / steps, f"Sampling {step}/{steps}"), loop
            )
        return callback_kwargs

    gen_kwargs = {
        "prompt": prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "width": int(width),
        "height": int(height),
        "generator": generator,
        "callback_on_step_end": pipe_cb,
    }

    if app_state["current_lora_name"] and float(lora_weight) > 0:
        gen_kwargs["cross_attention_kwargs"] = {"scale": float(lora_weight)}

    if negative_prompt:
        gen_kwargs["negative_prompt"] = negative_prompt

    logger.info(
        f"Generating image. Seed: {seed}, Size: {width}x{height}, Steps: {steps}"
    )

    try:
        start_time = time.time()
        result = app_state["current_pipe"].generate(**gen_kwargs)
        image = result.images[0]

        duration = time.time() - start_time
        logger.info(f"Generation finished in {duration:.2f}s")

        filename = f"ArtTic-LAB_{_get_next_image_number()}.png"
        filepath = os.path.join("./outputs", filename)
        image.save(filepath)

        meta = metadata_handler.create_metadata(
            prompt,
            negative_prompt,
            app_state["current_model_name"],
            seed,
            width,
            height,
            steps,
            guidance,
            lora_info=(
                {"name": app_state["current_lora_name"], "weight": lora_weight}
                if app_state["current_lora_name"]
                else None
            ),
        )
        metadata_handler.embed_metadata_to_image(filepath, meta)

        return {"image_filename": filename, "info": f"Seed: {seed}"}
    except torch.OutOfMemoryError:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        logger.error("OOM Error: VRAM exhausted.")
        raise OOMError("VRAM exhausted.")


def toggle_share():
    if app_state["public_url"]:
        logger.info("Disconnecting public share...")
        ngrok.disconnect(app_state["public_url"])
        app_state["public_url"] = None
        return {"status": "disconnected", "url": None}
    else:
        try:
            logger.info("Connecting Ngrok tunnel...")
            url = ngrok.connect(7860).public_url
            app_state["public_url"] = url
            logger.info(f"Public URL established: {url}")
            return {"status": "connected", "url": url}
        except Exception as e:
            logger.error(f"Ngrok connection failed: {e}")
            return {"status": "error", "message": str(e)}


def get_prompts():
    return prompt_book.get_all_prompts()


def add_prompt(t, p, np=""):
    return {"success": prompt_book.add_prompt(t, p, np)}


def update_prompt(ot, nt, p, np):
    return {"success": prompt_book.update_prompt(ot, nt, p, np)}


def delete_prompt(t):
    return {"success": prompt_book.delete_prompt(t)}


def get_image_metadata(filename):
    path = os.path.join("./outputs", filename)
    if not os.path.exists(path):
        return {"success": False, "error": "File not found"}
    m = metadata_handler.extract_metadata_from_image(path)
    return (
        {"success": True, "metadata": m}
        if m
        else {"success": False, "error": "No metadata"}
    )


def restart_backend():
    logger.info("Restart command initiated.")
    sys.exit(RESTART_EXIT_CODE)


def clear_cache():
    logger.info("Clearing VRAM cache...")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "success"}
