"""project paths, so every script finds data/ and models/ no matter where it is run from"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"              # input catalogues, e.g. data/tce_catalogue.csv
PROCESSED_DIR = DATA_DIR / "processed"  # build_dataset output, one folder per dataset
MODELS_DIR = ROOT / "models"          # train.py output, one folder per dataset

# create the project folders on first import; scripts create their own subfolders inside these
for _dir in (DATA_DIR, PROCESSED_DIR, MODELS_DIR):
    _dir.mkdir(exist_ok=True)
