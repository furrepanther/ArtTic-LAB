import logging
import sys
import os
import torch
import diffusers
import psutil
import platform
import subprocess

APP_LOGGER_NAME = "arttic_lab"
APP_VERSION = "4.1.0"


class ArtTicFilter(logging.Filter):
    def filter(self, record):
        return record.name == APP_LOGGER_NAME


class CustomFormatter(logging.Formatter):
    MINT = "\x1b[38;2;82;183;136m"
    GREEN = "\x1b[38;2;34;197;94m"
    BLUE = "\x1b[38;2;59;130;246m"
    RED = "\x1b[38;2;239;68;68m"
    ORANGE = "\x1b[38;2;249;115;22m"
    GREY = "\x1b[38;2;156;163;175m"
    WHITE = "\x1b[37m"
    BOLD = "\x1b[1m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.INFO: f"{MINT}[ArtTic-LAB]{RESET} > %(message)s",
        logging.WARNING: f"{ORANGE}[ArtTic-LAB] [WARN]{RESET} > %(message)s",
        logging.ERROR: f"{RED}[ArtTic-LAB] [ERROR]{RESET} > %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self._fmt)
        return logging.Formatter(log_fmt).format(record)


def get_cpu_model():
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif platform.system() == "Windows":
            return (
                subprocess.check_output("wmic cpu get name", shell=True)
                .decode()
                .split("\n")[1]
                .strip()
            )
        elif platform.system() == "Darwin":
            return (
                subprocess.check_output(
                    ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
                )
                .decode()
                .strip()
            )
    except Exception:
        pass
    return platform.processor() or "Generic CPU"


def get_distro_name():
    try:
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                data = {}
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        data[k] = v.strip('"')
            return f"{data.get('NAME', 'Linux')} {data.get('VERSION_ID', '')}".strip()
    except:
        pass
    return f"{platform.system()} {platform.release()}"


def get_gpu_info():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        name = torch.xpu.get_device_name(0)
        mem = torch.xpu.get_device_properties(0).total_memory / (1024**3)
        return "Intel XPU", name, f"{mem:.2f} GB"
    elif torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return "NVIDIA CUDA", name, f"{mem:.2f} GB"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple Metal", "M-Series GPU", "Shared"
    return "CPU", "Software Rendering", "N/A"


def log_system_info():
    c = CustomFormatter
    cpu_info = get_cpu_model()
    ram = psutil.virtual_memory().total / (1024**3)
    os_name = get_distro_name()
    kernel = platform.release()
    try:
        py_ver = sys.version.split()[0]
    except:
        py_ver = platform.python_version()

    acc_type, gpu_name, vram = get_gpu_info()
    user = os.getlogin() if hasattr(os, "getlogin") else "user"
    hostname = platform.node()

    art = f"""{c.MINT}
    █████╗ ██████╗ ████████╗████████╗██╗ ██████╗       
   ██╔══██╗██╔══██╗╚══██╔══╝╚══██╔══╝██║██╔════╝       
   ███████║██████╔╝   ██║      ██║   ██║██║            
   ██╔══██║██╔══██╗   ██║      ██║   ██║██║            
   ██║  ██║██║  ██║   ██║      ██║   ██║╚██████╗       
   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝ ╚═════╝       
    {c.RESET}"""

    print(art)

    def print_kv(key, val):
        print(f"   {c.MINT}{c.BOLD}{key:<12}{c.RESET} : {c.WHITE}{val}{c.RESET}")

    print_kv("User", f"{user}@{hostname}")
    print(f"   {c.GREY}----------------------------------------{c.RESET}")
    print_kv("OS", os_name)
    print_kv("Kernel", kernel)
    print_kv("CPU", cpu_info)
    print_kv("GPU", f"{gpu_name} ({vram})")
    print_kv("Memory", f"{ram:.2f} GB")
    print_kv("Backend", acc_type)
    print(f"   {c.GREY}----------------------------------------{c.RESET}")
    print_kv("Python", py_ver)
    print_kv("Torch", torch.__version__)
    print_kv("Diffusers", diffusers.__version__)
    print_kv("App Version", f"v{APP_VERSION}")
    print("")


def setup_logging(disable_filters=False):
    # Configure root logger to capture everything first
    logging.basicConfig(level=logging.ERROR, format="%(message)s")

    # Silence specific noisy libraries
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("uvicorn").setLevel(logging.ERROR)
    logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

    # Setup our custom logger
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False

    if app_logger.hasHandlers():
        app_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    app_logger.addHandler(handler)


def setup_web_logging():
    # Double ensure uvicorn is quiet
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
