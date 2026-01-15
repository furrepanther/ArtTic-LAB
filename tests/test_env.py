import unittest
import sys
import importlib
import warnings

warnings.filterwarnings("ignore")


class TestEnvironment(unittest.TestCase):
    def test_imports(self):
        pkgs = ["torch", "diffusers", "sdnq", "fastapi"]
        for p in pkgs:
            with self.subTest(package=p):
                importlib.import_module(p)

    def test_device(self):
        import torch

        print(f"\nPyTorch: {torch.__version__}")

        has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        has_cuda = torch.cuda.is_available()

        if has_xpu:
            print(f"[SUCCESS] Intel XPU: {torch.xpu.get_device_name(0)}")
        elif has_cuda:
            print(f"[SUCCESS] NVIDIA CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] Running on CPU.")

        self.assertTrue(True)  # Pass regardless, this is just a check


if __name__ == "__main__":
    unittest.main()
