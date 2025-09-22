from __future__ import annotations

import arcade
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "project.json"
ASSETS = ROOT / "assets" / "images"


def load_config(path: Path) -> dict:
    if not path.exists():
        return {
            "app": {"title": "RL Template App", "width": 1024, "height": 640, "show_overlay": True},
            "rng_seed": 123,
        }
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class TemplateWindow(arcade.Window):
    def __init__(self, width: int, height: int, title: str, show_overlay: bool = True):
        super().__init__(width=width, height=height, title=title, resizable=True)
        arcade.set_background_color(arcade.color.BLACK)
        self.show_overlay = show_overlay
        self._txt = arcade.Text("Template RL App", 20, height - 40, arcade.color.GOLD, 18)

    def on_draw(self):
        self.clear()
        self._txt.draw()


if __name__ == "__main__":
    cfg = load_config(CONFIG)
    app_cfg = cfg.get("app", {})
    win = TemplateWindow(
        width=int(app_cfg.get("width", 1024)),
        height=int(app_cfg.get("height", 640)),
        title=str(app_cfg.get("title", "RL Template App")),
        show_overlay=bool(app_cfg.get("show_overlay", True)),
    )
    arcade.run()
