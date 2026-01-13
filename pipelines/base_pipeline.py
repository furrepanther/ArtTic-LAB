import torch
import logging

logger = logging.getLogger("arttic_lab")


class CPUTextEncoderWrapper(torch.nn.Module):
    """
    Wraps a Text Encoder to force execution on CPU.
    This bypasses Intel XPU driver bugs related to specific tensor operations
    in Transformers (e.g. attention masks, integer casting).
    Inputs are moved to CPU -> Encoder runs on CPU -> Outputs moved to XPU.
    """

    def __init__(self, original_encoder):
        super().__init__()
        # Use simple assignment to register the submodule properly in torch.nn.Module
        self.original_encoder = original_encoder.to("cpu")

    @property
    def dtype(self):
        return self.original_encoder.dtype

    @property
    def device(self):
        return self.original_encoder.device

    @property
    def config(self):
        return self.original_encoder.config

    def __getattr__(self, name):
        # Prevent recursion by not intercepting special attributes
        if name in ["original_encoder", "_modules", "_parameters", "_buffers"]:
            return super().__getattr__(name)
        return getattr(self.original_encoder, name)

    def to(self, *args, **kwargs):
        # Prevent moving the underlying encoder to XPU.
        # We absorb the .to() call but ignore device changes if they target XPU/CUDA.
        device_arg = None
        if args and (isinstance(args[0], torch.device) or isinstance(args[0], str)):
            device_arg = args[0]
        elif "device" in kwargs:
            device_arg = kwargs["device"]

        # If trying to move to XPU/CUDA, ignore it for the encoder
        if device_arg and str(device_arg).startswith(("xpu", "cuda")):
            # Just return self, effectively pinning it to CPU
            return self

        # Allow other casts (like dtype changes) if needed
        self.original_encoder = self.original_encoder.to(*args, **kwargs)
        return self

    def forward(self, *args, **kwargs):
        # 1. Move inputs to CPU
        cpu_args = [
            arg.to("cpu") if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        cpu_kwargs = {
            k: (v.to("cpu") if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }

        # 2. Run on CPU
        outputs = self.original_encoder(*cpu_args, **cpu_kwargs)

        # 3. Move outputs back to XPU
        def to_xpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to("xpu")
            elif isinstance(obj, (tuple, list)):
                return type(obj)(to_xpu(x) for x in obj)
            elif hasattr(obj, "to"):  # Handle ModelOutput objects
                return obj.to("xpu")
            return obj

        return to_xpu(outputs)


class ArtTicPipeline:
    def __init__(self, model_path, dtype=torch.bfloat16):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.pipe = None
        self.model_path = model_path
        self.dtype = dtype
        self.is_offloaded = False

    def load_pipeline(self, progress):
        raise NotImplementedError("Subclasses must implement load_pipeline")

    def _wrap_text_encoders_for_xpu(self):
        """
        Subclasses should implement this to wrap their specific text encoders
        using CPUTextEncoderWrapper if running on XPU.
        """
        pass

    def place_on_device(self, use_cpu_offload=False):
        if not self.pipe:
            raise RuntimeError("Pipeline must be loaded before placing on device.")

        # XPU Specific: Wrap Text Encoders to run on CPU
        if self.device == "xpu":
            self._wrap_text_encoders_for_xpu()

        if use_cpu_offload and self.device != "cpu":
            logger.info("Enabling Model CPU Offload for low VRAM usage.")
            self.pipe.enable_model_cpu_offload()
            self.is_offloaded = True
        else:
            logger.info(f"Moving model to {self.device} for maximum performance.")
            self.pipe.to(self.device)
            self.is_offloaded = False

    def generate(self, *args, **kwargs):
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded.")

        if self.device == "xpu":
            # Direct execution for XPU (Text Encoders are wrapped to handle precision/device safety)
            return self.pipe(*args, **kwargs)
        elif self.device == "cuda":
            with torch.autocast("cuda", enabled=True, dtype=self.dtype):
                return self.pipe(*args, **kwargs)
        else:
            return self.pipe(*args, **kwargs)