import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import logic

class TestLogic(unittest.TestCase):

    @patch("core.logic.torch.cuda.is_available", return_value=False)
    @patch("core.logic.torch.xpu.is_available", return_value=False)
    def test_resolution_cpu(self, mock_xpu, mock_cuda):
        # CPU should default to 1024
        res = logic._calculate_max_resolution("SDXL")
        self.assertEqual(res, 1024)

    @patch("os.path.exists", return_value=False)
    def test_delete_model_file_not_found(self, mock_exists):
        # Test the not_found status logic
        result = logic.delete_model_file("missing.safetensors")
        self.assertEqual(result["status"], "not_found")

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch("core.logic.os.path.commonpath")
    def test_delete_model_file_success(self, mock_common, mock_remove, mock_exists):
        # Mock commonpath to allow deletion validation
        mock_common.return_value = os.path.abspath("./models")
        
        result = logic.delete_model_file("exist.safetensors")
        self.assertEqual(result["status"], "success")

if __name__ == "__main__":
    unittest.main()
