from fastapi.testclient import TestClient
from pathlib import Path
import numpy as np
import pandas as pd
from main import app  # import your FastAPI app
import io

client = TestClient(app)

def test_upload_endpoint(tmp_path):
    """
    Test the /upload endpoint with dummy CSV and FITS files.
    """

    # --- Create dummy CSV ---
    csv_path = "test_tce.csv"
    zipfold_path = "test_lc_data.zip"

    # --- Upload files ---
    with open(csv_path, "rb") as csv_f, open(fits_path, "rb") as fits_f:
        response = client.post(
            "/upload",
            files={
                "csv": ("test.csv", csv_f, "text/csv"),
                "fits": ("dummy.fits", fits_f, "application/fits")
            }
        )

    # --- Check response ---
    assert response.status_code == 200
    json_data = response.json()
    print(json_data)

    # Example assertions
    assert "probs" in json_data
    assert "kepid" in json_data
    assert isinstance(json_data["probs"], list)
