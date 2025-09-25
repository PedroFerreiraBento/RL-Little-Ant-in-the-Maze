from __future__ import annotations

from pathlib import Path
import json

from src.envs.gridworld_v1 import GridworldV1
from src.engine.gridworld_renderer import GridworldV1Window, VizCfg


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "gridworld_v1.json"

    # Read method from base config and optionally method-specific params from an external file
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    method = str(raw_cfg.get("method", "vi")).lower()
    method_cfg_path = raw_cfg.get("method_config")
    method_params: dict = {}
    if isinstance(method_cfg_path, str) and method_cfg_path:
        mpath = (root / Path(method_cfg_path)).resolve()
        if mpath.exists():
            try:
                method_params = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:
                method_params = {}
    # Extract SARSA params (load even if initial method != sarsa so M-switch works)
    sarsa_params = method_params if isinstance(method_params, dict) else {}

    env = GridworldV1.from_json(cfg_path)
    # Configure env method and SARSA params, then compute policies (and train SARSA if configured)
    try:
        env.set_method(method)
    except Exception:
        env.method = "vi"
    env.sarsa_params = sarsa_params if isinstance(sarsa_params, dict) else {}
    env.recompute_policies_and_maybe_train()

    viz = VizCfg(cell_size=32, step_interval=0.10)
    window = GridworldV1Window(
        env,
        viz=viz,
    )
    import arcade

    arcade.run()
