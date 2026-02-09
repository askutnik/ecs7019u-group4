from __future__ import annotations
import os
from pathlib import Path

def get_data_dir()->Path:
    val = os.environ.get("PROJECT_DATA_DIR")
    if not val:
        raise EnvironmentError("PROJECT_DATA_DIR is not set!")
    return Path(val).expanduser().resolve()

DATA_DIR: Path = get_data_dir()
RAW_DIR: Path = DATA_DIR/"raw"
PROCESSED_DIR: Path = DATA_DIR/"processed"
MODELS_DIR: Path = DATA_DIR/"models"