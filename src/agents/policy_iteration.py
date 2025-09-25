from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

# Reuse canonical config and helpers from the planning package
from src.utils.gridworld_core import (
    GWConfig,
    ACTIONS,
    UP,
    step_model,
    interior_states,
)


def policy_iteration(
    cfg: GWConfig,
    theta: float = 1e-6,
    max_eval_iters: int = 10_000,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], int]]:
    """Compute optimal values and policy via Policy Iteration.

    Alternates between policy evaluation (iterative Bellman expectation backup
    following the current policy) and policy improvement (greedy w.r.t. the
    current value function), over interior, non-wall states.

    Args:
        cfg: GWConfig with grid shape, rewards, discount factor, and walls.
        theta: Convergence threshold for policy evaluation step.
        max_eval_iters: Max iterations per evaluation phase.

    Returns:
        - V: Mapping state -> value under the converged optimal policy.
        - pi: Optimal deterministic policy mapping state -> action.
    """
    states = interior_states(cfg.cols, cfg.rows, cfg.walls)
    # init arbitrary policy
    pi: Dict[Tuple[int, int], int] = {s: int(np.random.choice(ACTIONS)) for s in states}
    pi[cfg.goal] = UP
    V: Dict[Tuple[int, int], float] = {s: 0.0 for s in states}
    V[cfg.goal] = 0.0

    policy_stable = False
    while not policy_stable:
        # policy evaluation
        for _ in range(max_eval_iters):
            delta = 0.0
            for s in states:
                if s == cfg.goal:
                    continue
                v_old = V[s]
                a = pi[s]
                s2, r, done = step_model(cfg, s, a)
                V[s] = r + (0.0 if done else cfg.gamma * V.get(s2, 0.0))
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break
        # policy improvement
        policy_stable = True
        for s in states:
            if s == cfg.goal:
                continue
            old_a = pi[s]
            best_a = old_a
            best = -1e18
            for a in ACTIONS:
                s2, r, done = step_model(cfg, s, a)
                val = r + (0.0 if done else cfg.gamma * V.get(s2, 0.0))
                if val > best:
                    best = val
                    best_a = a
            pi[s] = best_a
            if best_a != old_a:
                policy_stable = False
    return V, pi
