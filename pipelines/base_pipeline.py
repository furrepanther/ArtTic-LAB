import torch
import logging

logger = logging.getLogger("arttic_lab")


class ArtTicPipeline:
    def __init__(self, model_path, dtype=torch.bfloat16):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device("xpu")
            self.device_type = "xpu"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"

        self.pipe = None
        self.model_path = model_path
        self.dtype = dtype
        self.is_offloaded = False

    def load_pipeline(self, progress, use_quantization=False):
        raise NotImplementedError

    def place_on_device(self, use_cpu_offload=False):
        if not self.pipe:
            raise RuntimeError("Pipeline must be loaded before placing on device.")

        if use_cpu_offload and self.device_type != "cpu":
            self.pipe.enable_model_cpu_offload()
            self.is_offloaded = True
        else:
            self.pipe.to(self.device)
            self.is_offloaded = False

        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            self.pipe.vae.to(dtype=torch.float32)

    def generate(self, *args, **kwargs):
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded.")

        with torch.autocast(
            device_type=self.device_type,
            enabled=(self.device_type != "cpu"),
            dtype=self.dtype,
        ):
            result = self.pipe(*args, **kwargs)

        return result
