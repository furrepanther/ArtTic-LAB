import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines import get_pipeline_for_model
# Import pipeline classes to check types
from pipelines.sd15_pipeline import SD15Pipeline
from pipelines.sdxl_pipeline import SDXLPipeline
from pipelines.flux_pipeline import ArtTicFLUXPipeline

class TestPipelines(unittest.TestCase):
    
    @patch("pipelines.safe_open")
    @patch("os.path.exists") # safetensors might check file existence
    @patch("pipelines.SD15Pipeline.__init__", return_value=None)
    @patch("pipelines.SDXLPipeline.__init__", return_value=None)
    @patch("pipelines.ArtTicFLUXPipeline.__init__", return_value=None)
    def test_detection_flux(self, mock_flux, mock_xl, mock_sd15, mock_exists, mock_safe_open):
        # Setup mock for safe_open context manager
        mock_file = MagicMock()
        mock_file.keys.return_value = ["transformer.blocks.0.weight"]
        mock_safe_open.return_value.__enter__.return_value = mock_file
        
        # Test FLUX detection
        pipe = get_pipeline_for_model("flux_dev")
        self.assertIsInstance(pipe, ArtTicFLUXPipeline)

    @patch("pipelines.safe_open")
    @patch("pipelines.SD15Pipeline.__init__", return_value=None)
    @patch("pipelines.SDXLPipeline.__init__", return_value=None)
    def test_detection_sdxl(self, mock_xl, mock_sd15, mock_safe_open):
        mock_file = MagicMock()
        # Key specific to XL in _is_xl
        mock_file.keys.return_value = ["conditioner.embedders.1.model"] 
        mock_safe_open.return_value.__enter__.return_value = mock_file
        
        pipe = get_pipeline_for_model("sdxl_turbo")
        self.assertIsInstance(pipe, SDXLPipeline)

    @patch("pipelines.safe_open")
    @patch("pipelines.SD15Pipeline.__init__", return_value=None)
    def test_detection_default(self, mock_sd15, mock_safe_open):
        mock_file = MagicMock()
        mock_file.keys.return_value = ["random.key"]
        mock_safe_open.return_value.__enter__.return_value = mock_file
        
        pipe = get_pipeline_for_model("unknown_model")
        self.assertIsInstance(pipe, SD15Pipeline)

if __name__ == "__main__":
    unittest.main()
