from __future__ import annotations

from pathlib import Path
from src.envs.gridworld_v1 import GridworldV1
from src.agents.planning.gridworld_planning import (
    GWConfig,
    value_iteration,
    policy_iteration,
    arrows,
)

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
        walls=set(env.walls),
    )

    V_vi, pi_vi = value_iteration(gwcfg)
    V_pi, pi_pi = policy_iteration(gwcfg)

    print("Value Iteration policy (arrows):")
    print("\n".join(arrows(pi_vi, env.cols, env.rows, env.cfg.goal)))
    print()
    print("Policy Iteration policy (arrows):")
    print("\n".join(arrows(pi_pi, env.cols, env.rows, env.cfg.goal)))
