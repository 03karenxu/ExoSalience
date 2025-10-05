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
    csv_data = pd.DataFrame({
        "kepid": [1, 2, 3],
        "tce_period": [10.0, 15.0, 20.0],
        "tce_time0bk": [2455000, 2455001, 2455002]
    })
    csv_file_path = tmp_path / "dummy.csv"
    csv_data.to_csv(csv_file_path, index=False)

    # --- Create dummy FITS ---
    fits_file_path = tmp_path / "dummy.fits"
    # minimal FITS file with astropy (or just dummy bytes)
    fits_file_path.write_bytes(b"SIMPLE  =                    T / file does not really matter")

    # --- Upload files ---
    with open(csv_file_path, "rb") as csv_f, open(fits_file_path, "rb") as fits_f:
        response = client.post(
            "/upload",
            files={
                "csv": ("dummy.csv", csv_f, "text/csv"),
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
