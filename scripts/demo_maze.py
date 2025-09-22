from __future__ import annotations

from pathlib import Path
from src.engine.maze import run_from_config

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = root / "config" / "maze.json"
    run_from_config(cfg)
