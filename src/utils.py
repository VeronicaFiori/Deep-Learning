import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(save_dir, name, model, optimizer, epoch, best_metric, extra=None):
    ensure_dir(save_dir)
    path = os.path.join(save_dir, name)
    payload = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optim") is not None:
        optimizer.load_state_dict(ckpt["optim"])
    return ckpt
