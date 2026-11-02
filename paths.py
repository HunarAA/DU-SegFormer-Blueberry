"""Project paths. By default everything is relative to this folder:

    <repo>/DB-1/Images   AgriVision DB-1 images (.bmp)
    <repo>/DB-1/Masks    AgriVision DB-1 masks (.png)
    <repo>/splits.csv    fixed source-video-disjoint split (included)
    <repo>/runs/         checkpoints and result CSVs (created automatically)

Override with environment variables if your data lives elsewhere:
    BLUEBERRY_ROOT  project folder (splits.csv and runs/)
    BLUEBERRY_DATA  dataset folder containing Images/ and Masks/
"""
import os
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("BLUEBERRY_ROOT", Path(__file__).resolve().parent))
DATASET_DIR = Path(os.environ.get("BLUEBERRY_DATA", PROJECT_DIR / "DB-1"))
