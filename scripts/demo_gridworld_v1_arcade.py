from __future__ import annotations

from pathlib import Path

from src.envs.gridworld_v1 import GridworldV1
from src.agents.planning.gridworld_planning import (
    GWConfig,
    value_iteration,
    policy_iteration,
)
from src.engine.gridworld_renderer import GridworldV1Window, VizCfg


if __name__ == "__main__":
    import argparse
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "gridworld_v1.json"

    parser = argparse.ArgumentParser(description="Gridworld V1 Arcade demo with planning")
    parser.add_argument("--method", choices=["vi", "pi"], default="vi", help="Planning method: vi=value iteration, pi=policy iteration")
    args = parser.parse_args()

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
        walls=set(getattr(env, "walls", set())),
    )

    # Compute both policies so the UI can toggle between them
    _, pi_vi = value_iteration(gwcfg)
    _, pi_pi = policy_iteration(gwcfg)

    viz = VizCfg(cell_size=32, step_interval=0.10)
    window = GridworldV1Window(env, viz=viz, pi_vi=pi_vi, pi_pi=pi_pi, method=args.method)
    import arcade

    arcade.run()
