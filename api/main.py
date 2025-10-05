from typing import Union

import os
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel, UploadFile, File, HTTPException
app = FastAPI()

class Input(BaseModel):
    ...

    
class Output(BaseModel):
    ...

app = FastAPI()

@app.post("/upload")
async def upload_data(csv: UploadFile, fits: UploadFile):

    # save csv to tmpfile
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    csv_path = tmp_csv.name
    content = await csv.read()
    tmp_csv.write(content)

    tmp_fits = tempfile.NamedTemporaryFile(delte=False, suffix='.fits')
    fits_path = tmp_fits.name
    content = await fits.read()
    tmp_fits.write(content)

    processed_data = preprocess(tmp_csv, tmp_fits)


    # return {prediction:x, heatmap:x,  graph: x}
    
    return item

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
