from __future__ import annotations

from pathlib import Path
import numpy as np

from src.envs.gridworld_v1 import GridworldV1, UP, DOWN, LEFT, RIGHT


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "gridworld_v1.json"

    env = GridworldV1.from_json(cfg_path)
    rng = np.random.default_rng(123)

    episodes = 5
    for ep in range(1, episodes + 1):
        s = env.reset(rng=rng)
        total_r = 0.0
        for t in range(env.max_steps):
            # Random policy over {UP, DOWN, LEFT, RIGHT}
            a = int(rng.integers(0, 4))
            s, r, done, info = env.step(a)
            total_r += r
            # print step summary
            print(f"ep={ep:02d} t={t:03d} a={a} s={s} r={r:+.1f}")
            if done:
                print(f"--> done at t={t}, total_r={total_r:+.1f}\n")
                break
        else:
            print(f"--> max steps reached, total_r={total_r:+.1f}\n")
