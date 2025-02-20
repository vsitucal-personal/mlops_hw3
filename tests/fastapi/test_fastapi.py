import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi_dir.app import app  # Ensure correct module import


class TestFastAPIApp(unittest.TestCase):
    @patch("mlflow.pyfunc")
    def test_predict(self, mock_get_model):
        """Test the /predict endpoint."""
        client = TestClient(app)

        # Mock MLflow Model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 2]
        mock_get_model.return_value = mock_model

        # Send a request
        response = client.post("/predict", json={"data": [[5.1, 3.5, 1.4, 0.2], [6.1, 2.8, 4.7, 1.2]]})

        # Assertions
        assert response.status_code == 200
        assert "predictions" in response.json()
