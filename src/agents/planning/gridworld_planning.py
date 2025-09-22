from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set
import numpy as np

# Actions mapping to match env: 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)


@dataclass
class GWConfig:
    """Configuration for Gridworld planning.

    Attributes:
        cols: Number of columns in the grid.
        rows: Number of rows in the grid.
        start: Start state (x, y).
        goal: Goal state (x, y).
        step_reward: Reward for a normal (non-collision, non-terminal) step.
        goal_reward: Reward upon reaching the goal (terminal transition).
        collision_reward: Reward when an attempted move is blocked.
        gamma: Discount factor in [0, 1).
        walls: Set of interior wall coordinates (x, y) that are not traversable.
    """
    cols: int
    rows: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    step_reward: float
    goal_reward: float
    collision_reward: float
    gamma: float
    walls: Set[Tuple[int, int]] = field(default_factory=set)


def interior_states(cols: int, rows: int, walls: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """List all traversable interior states (excluding borders and walls).

    Args:
        cols: Grid width (number of columns).
        rows: Grid height (number of rows).
        walls: Set of (x, y) coordinates that are blocked/impassable.

    Returns:
        List of (x, y) coordinates for interior, non-wall states. Borders
        (x in {0, cols-1} or y in {0, rows-1}) are excluded.
    """
    return [(x, y) for y in range(1, rows - 1) for x in range(1, cols - 1) if (x, y) not in walls]


def step_model(cfg: GWConfig, s: Tuple[int, int], a: int) -> Tuple[Tuple[int, int], float, bool]:
    """One-step deterministic transition model consistent with env rules.

    Applies a single action to state ``s``. Movement is clamped to the interior
    (borders are walls) and blocked by any interior walls given in
    ``cfg.walls``.

    Args:
        cfg: Planning config with grid, rewards, gamma and walls.
        s: Current state (x, y).
        a: Action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).

    Returns:
        s2: Next state after applying the action and constraints.
        r: Reward received (goal, collision, or step reward).
        done: True if the next state is the goal (terminal), else False.
    """
    x, y = s
    nx, ny = x, y
    if a == UP:
        ny = min(cfg.rows - 2, y + 1)
    elif a == DOWN:
        ny = max(1, y - 1)
    elif a == LEFT:
        nx = max(1, x - 1)
    elif a == RIGHT:
        nx = min(cfg.cols - 2, x + 1)

    bumped = False
    if (nx, ny) in cfg.walls:
        nx, ny = x, y
        bumped = True
    s2 = (nx, ny)
    done = s2 == cfg.goal
    if done:
        r = cfg.goal_reward
    else:
        r = cfg.collision_reward if bumped else cfg.step_reward
    return s2, float(r), bool(done)


def value_iteration(cfg: GWConfig, theta: float = 1e-6, max_iters: int = 10_000) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], int]]:
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


def policy_iteration(cfg: GWConfig, theta: float = 1e-6, max_eval_iters: int = 10_000) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], int]]:
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
    pi: Dict[Tuple[int, int], int] = {s: np.random.choice(ACTIONS) for s in states}
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


def arrows(pi: Dict[Tuple[int, int], int], cols: int, rows: int, goal: Tuple[int, int]) -> List[str]:
    """Render a textual arrow map for a given policy.

    Args:
        pi: Policy mapping (x, y) -> action (UP/DOWN/LEFT/RIGHT).
        cols: Grid width.
        rows: Grid height.
        goal: Goal coordinate to mark with 'G'.

    Returns:
        List of strings, each string is one row with border '#', goal 'G', and
        arrows ('^', 'v', '<', '>') for interior cells.
    """
    grid = [["#" if (x in (0, cols - 1) or y in (0, rows - 1)) else " " for x in range(cols)] for y in range(rows)]
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if (x, y) == goal:
                grid[y][x] = "G"
            else:
                a = pi.get((x, y), UP)
                grid[y][x] = {UP: "^", DOWN: "v", LEFT: "<", RIGHT: ">"}[a]
    return ["".join(row) for row in grid]
