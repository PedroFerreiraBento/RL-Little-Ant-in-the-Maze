from __future__ import annotations

from typing import Dict, Tuple

# Reuse canonical config and helpers from the planning package
from src.utils.gridworld_core import (
    GWConfig,
    ACTIONS,
    UP,
    step_model,
    interior_states,
)


def value_iteration(
    cfg: GWConfig,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], int]]:
    """Compute optimal state values and a greedy policy via Value Iteration.

    This runs the standard Bellman optimality backup over the interior (non-wall)
    states of the grid until convergence. Transitions and rewards are given by
    ``step_model`` which respects border bounds and interior ``cfg.walls``.

    Args:
        cfg: GWConfig with grid shape, rewards, discount factor, and walls.
        theta: Convergence threshold for the maximum absolute value change per sweep.
        max_iters: Maximum number of sweeps before giving up.

    Returns:
        - V: Mapping from state (x, y) to optimal value estimate.
        - pi: Greedy policy mapping from state (x, y) to best action in ACTIONS.

    Notes:
        - Goal state's value is fixed to 0.0 here (absorbing). If you prefer the
          value to reflect the terminal reward, you can set it differently, but
          since we stop at the goal, 0.0 is conventional and sufficient.
        - The algorithm is deterministic given cfg (no stochastic dynamics).
    """
    # Collect all non-wall interior states to evaluate
    states = interior_states(cfg.cols, cfg.rows, cfg.walls)

    # Initialize the value function. Setting goal to 0 (absorbing) is common.
    V: Dict[Tuple[int, int], float] = {s: 0.0 for s in states}
    V[cfg.goal] = 0.0

    # Main VI loop: repeatedly apply Bellman optimality backup until stable
    for _ in range(max_iters):
        delta = 0.0  # track the largest change this sweep
        for s in states:
            if s == cfg.goal:
                # Skip backing up the terminal state
                continue
            v_old = V[s]

            # Evaluate one-step lookahead for all actions and pick the best
            best = -1e18
            for a in ACTIONS:
                s2, r, done = step_model(cfg, s, a)
                # If next is terminal, no discounted continuation
                val = r + (0.0 if done else cfg.gamma * V.get(s2, 0.0))
                if val > best:
                    best = val
            V[s] = best
            delta = max(delta, abs(v_old - V[s]))

        # Converged if the largest change is below tolerance
        if delta < theta:
            break

    # Derive a greedy policy w.r.t. the converged value function
    pi: Dict[Tuple[int, int], int] = {}
    for s in states:
        if s == cfg.goal:
            # Arbitrary at terminal; keep UP as placeholder
            pi[s] = UP
            continue
        best_a = UP
        best = -1e18
        for a in ACTIONS:
            s2, r, done = step_model(cfg, s, a)
            val = r + (0.0 if done else cfg.gamma * V.get(s2, 0.0))
            if val > best:
                best = val
                best_a = a
        pi[s] = best_a
    return V, pi
