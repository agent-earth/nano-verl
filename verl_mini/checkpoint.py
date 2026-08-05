from pathlib import Path

import torch


def optimizer_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / "optimizer.pt"


def save_optimizer(optimizer, checkpoint_dir: str | Path) -> None:
    torch.save(optimizer.state_dict(), optimizer_path(checkpoint_dir))


def load_optimizer(optimizer, checkpoint_dir: str | Path) -> None:
    path = optimizer_path(checkpoint_dir)
    if path.exists():
        optimizer.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
