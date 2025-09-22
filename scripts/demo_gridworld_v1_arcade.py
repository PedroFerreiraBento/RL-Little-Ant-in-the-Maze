from __future__ import annotations

from pathlib import Path

from src.envs.gridworld_v1 import GridworldV1
from src.agents.planning.gridworld_planning import (
    GWConfig,
    value_iteration,
)
from src.engine.gridworld_renderer import GridworldV1Window, VizCfg


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "gridworld_v1.json"

    env = GridworldV1.from_json(cfg_path)

    gwcfg = GWConfig(
        cols=env.cols,
        rows=env.rows,
        start=env.cfg.start,
        goal=env.cfg.goal,
        step_reward=env.rew.step,
        goal_reward=env.rew.goal,
        collision_reward=env.rew.collision,
        gamma=env.cfg.gamma,
    )

    # Compute optimal policy via Value Iteration
    _, pi = value_iteration(gwcfg)

    viz = VizCfg(cell_size=32, step_interval=0.10)
    window = GridworldV1Window(env, policy=pi, viz=viz)
    import arcade

    arcade.run()
