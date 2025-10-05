from typing import Union

import os
import torch
import numpy as np
import tempfile
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel, UploadFile, File, HTTPException
from src.infer import infer

app = FastAPI()

_PATH_TO_OUTPUT = "/path/"

class Input(BaseModel):
    ...

    
class Output(BaseModel):
    probs: list[float]

app = FastAPI()

@app.post("/upload", responseModel=Output)
async def upload_data(csv: UploadFile = File(...), fits: UploadFile = File(...)):

    # save csv to tmpfile
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    csv_path = tmp_csv.name
    content = await csv.read()
    tmp_csv.write(content)

    # save fits to tmpfile
    tmp_fits = tempfile.NamedTemporaryFile(delete=False, suffix='.fits')
    fits_path = tmp_fits.name
    content = await fits.read()
    tmp_fits.write(content)

    # run create_input on tmp csv and tmp fits
    subprocess.run(
                ["python", "data/create_input.py", f"--input_tce_csv_file {csv_path}", f"--kepler_data_dir {fits_path}", f"--output_dir {_PATH_TO_OUTPUT}"],
                check=True
            )

    os.remove(csv_path)
    os.remove(fits_path)

    # open the output .npz
    data = np.load(_PATH_TO_OUTPUT)

    # convert to np arrays
    g_np = data['global_view']
    l_np = data['local_view']
    kepid = data['kepid']
    tab = data.get['tab']

    # convert to tensors
    x_global = torch.from_numpy(g_np)
    x_local = torch.from_numpy(l_np)
    x_tab = torch.from_numpy(tab)

    # infer
    prob_list = infer(x_global=x_global, x_local=x_local, x_tab=x_tab)

    return {"probs":prob_list, "kepid":kepid }
