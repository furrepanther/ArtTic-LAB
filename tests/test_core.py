import unittest
import sys
import os
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.prompt_book import PromptBook
from core.metadata_handler import MetadataHandler


class TestPromptBook(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_prompts.toml"
        self.book = PromptBook(prompts_file=self.test_file)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_add_and_get_prompt(self):
        success = self.book.add_prompt("Test Title", "Test Prompt", "Negative")
        self.assertTrue(success)

        prompts = self.book.get_all_prompts()
        found = next((p for p in prompts if p["title"] == "Test Title"), None)
        self.assertIsNotNone(found)
        self.assertEqual(found["prompt"], "Test Prompt")


class TestMetadata(unittest.TestCase):
    def test_create_metadata(self):
        handler = MetadataHandler()
        meta = handler.create_metadata(
            prompt="A cat",
            negative_prompt="not a dog",
            model_name="test_model",
            seed=12345,
            width=512,
            height=512,
            steps=20,
            cfg_scale=7.5,
        )
        self.assertEqual(meta["prompt"], "A cat")
        self.assertEqual(meta["seed"], 12345)
        self.assertTrue("hash" in meta)


if __name__ == "__main__":
    unittest.main()
