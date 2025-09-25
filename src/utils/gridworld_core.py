from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set

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
    """List all traversable interior states (excluding borders and walls)."""
    return [(x, y) for y in range(1, rows - 1) for x in range(1, cols - 1) if (x, y) not in walls]


def step_model(cfg: GWConfig, s: Tuple[int, int], a: int) -> Tuple[Tuple[int, int], float, bool]:
    """One-step deterministic transition model consistent with env rules."""
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
