import random
import numpy as np
import torch
from pathlib import Path

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_npy_files(path):
    return sorted(Path(path).rglob("*.npy"))

def normalize_spectral(arr):
    """
    STTORM raw arrays are H,W,C uint16 DN values.
    Returns float32 H,W,C using the official project's log-style
    normalization for the 10 selected bands.
    """
    x = arr.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 1, 100000)
    x = np.log(x)

    # Same band ranges used by the STTORM-CD reference preprocessing.
    x0 = np.array([6.9, 6.5, 6.2, 6.1, 6.5, 6.5, 6.5, 6.5, 6.0, 6.0], dtype=np.float32)
    x1 = np.array([7.5, 7.4, 7.5, 7.5, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float32)
    x = 2.0 * ((x - x0.reshape(1,1,-1)) / (x1-x0).reshape(1,1,-1)) - 1.0
    return np.clip(x, -3.0, 3.0).astype(np.float32)

def to_chw(x):
    return np.transpose(x, (2,0,1)).copy()

def save_checkpoint(model, path, **extra):
    payload = {"model": model.state_dict(), **extra}
    torch.save(payload, path)

def load_checkpoint(model, path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    return payload
