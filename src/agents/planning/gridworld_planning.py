from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set
import numpy as np

# Actions mapping to match env: 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)


@dataclass
class GWConfig:
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
    return [(x, y) for y in range(1, rows - 1) for x in range(1, cols - 1) if (x, y) not in walls]


def step_model(cfg: GWConfig, s: Tuple[int, int], a: int) -> Tuple[Tuple[int, int], float, bool]:
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
    states = interior_states(cfg.cols, cfg.rows, cfg.walls)
    V: Dict[Tuple[int, int], float] = {s: 0.0 for s in states}
    V[cfg.goal] = 0.0

    for _ in range(max_iters):
        delta = 0.0
        for s in states:
            if s == cfg.goal:
                continue
            v_old = V[s]
            best = -1e18
            for a in ACTIONS:
                s2, r, done = step_model(cfg, s, a)
                val = r + (0.0 if done else cfg.gamma * V.get(s2, 0.0))
                if val > best:
                    best = val
            V[s] = best
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break

    # derive greedy policy
    pi: Dict[Tuple[int, int], int] = {}
    for s in states:
        if s == cfg.goal:
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
    states = interior_states(cfg.cols, cfg.rows)
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
    grid = [["#" if (x in (0, cols - 1) or y in (0, rows - 1)) else " " for x in range(cols)] for y in range(rows)]
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if (x, y) == goal:
                grid[y][x] = "G"
            else:
                a = pi.get((x, y), UP)
                grid[y][x] = {UP: "^", DOWN: "v", LEFT: "<", RIGHT: ">"}[a]
    return ["".join(row) for row in grid]
