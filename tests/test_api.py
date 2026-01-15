import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import app (requires mocking core imports if they have side effects, 
# but server.py imports are mostly safe until startup)
from web.server import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_read_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    @patch("web.server.core.get_app_status")
    def test_get_status(self, mock_get_status):
        mock_get_status.return_value = {"is_model_loaded": False}
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"is_model_loaded": False})

    @patch("web.server.core.add_prompt")
    def test_add_prompt(self, mock_add):
        mock_add.return_value = {"success": True}
        response = self.client.post("/api/prompts", json={
            "title": "New", 
            "prompt": "Test", 
            "negative_prompt": "Neg"
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        mock_add.assert_called_with("New", "Test", "Neg")

    @patch("web.server.core.delete_prompt")
    def test_delete_prompt(self, mock_delete):
        mock_delete.return_value = {"success": True}
        # DELETE with path parameter
        response = self.client.delete("/api/prompts/To%20Delete")
        self.assertEqual(response.status_code, 200)
        mock_delete.assert_called_with("To Delete")

if __name__ == "__main__":
    unittest.main()
