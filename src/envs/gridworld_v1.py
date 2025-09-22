from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Set, List, Optional
import json
from pathlib import Path

import numpy as np


Action = int
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)


@dataclass
class Rewards:
    step: float = -1.0
    goal: float = 10.0
    collision: float = 0.0  # set to -5.0 to punish bumping walls


@dataclass
class GridCfg:
    cols: int = 11
    rows: int = 11
    max_steps: int = 200
    start: Tuple[int, int] = (1, 1)
    goal: Tuple[int, int] = (9, 9)
    randomize_start_neighbors: bool = False
    rewards: Rewards = field(default_factory=Rewards)
    gamma: float = 0.95
    layout: Optional[str] = None  # e.g., "corridor"
    walls: Optional[List[Tuple[int, int]]] = None  # interior walls


class GridworldV1:
    """Deterministic 11x11 gridworld.

    - Actions: UP, DOWN, LEFT, RIGHT
    - Transitions: move one cell; hitting a wall keeps agent in place
    - Rewards: step, goal, collision
    - Termination: reaching goal or max_steps
    """

    def __init__(self, cfg: GridCfg):
        self.cfg = cfg
        self.cols = cfg.cols
        self.rows = cfg.rows
        self.max_steps = cfg.max_steps
        self.rew = cfg.rewards
        self._t = 0
        self._pos = cfg.start
        # Build interior walls set
        self.walls: Set[Tuple[int, int]] = set()
        if cfg.layout == "corridor":
            cy = self.rows // 2
            for y in range(1, self.rows - 1):
                for x in range(1, self.cols - 1):
                    if y != cy:
                        self.walls.add((x, y))
            # Ensure start/goal lie on corridor row
            sx, sy = self.cfg.start
            gx, gy = self.cfg.goal
            self.cfg.start = (1, cy)
            self.cfg.goal = (self.cols - 2, cy)
            self._pos = self.cfg.start
        elif cfg.walls:
            # Accept provided walls list, but clip to interior
            for x, y in cfg.walls:
                if 1 <= x < self.cols - 1 and 1 <= y < self.rows - 1:
                    self.walls.add((x, y))

    @staticmethod
    def from_json(path: Path) -> "GridworldV1":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rewards = Rewards(**data.get("rewards", {}))
        gc = GridCfg(
            cols=int(data.get("grid", {}).get("cols", 11)),
            rows=int(data.get("grid", {}).get("rows", 11)),
            max_steps=int(data.get("max_steps", 200)),
            start=tuple(data.get("start", [1, 1])),
            goal=tuple(data.get("goal", [9, 9])),
            randomize_start_neighbors=bool(data.get("randomize_start_neighbors", False)),
            rewards=rewards,
            gamma=float(data.get("gamma", 0.95)),
            layout=data.get("layout"),
            walls=[tuple(p) for p in data.get("walls", [])],
        )
        return GridworldV1(gc)

    # --- API ---
    def reset(self, rng: np.random.Generator | None = None) -> Tuple[int, int]:
        self._t = 0
        if self.cfg.randomize_start_neighbors:
            sx, sy = self.cfg.start
            candidates = [(sx, sy), (sx + 1, sy), (sx - 1, sy), (sx, sy + 1), (sx, sy - 1)]
            candidates = [
                (x, y)
                for x, y in candidates
                if 1 <= x < self.cols - 1 and 1 <= y < self.rows - 1 and (x, y) != self.cfg.goal and (x, y) not in self.walls
            ]
            if rng is None:
                rng = np.random.default_rng()
            self._pos = candidates[int(rng.integers(0, len(candidates)))]
        else:
            self._pos = self.cfg.start
            if self._pos in self.walls:
                # find nearest free cell on same row if possible
                x, y = self._pos
                for dx in range(1, max(self.cols, self.rows)):
                    for sx in (x - dx, x + dx):
                        if 1 <= sx < self.cols - 1 and (sx, y) not in self.walls:
                            self._pos = (sx, y)
                            break
                    else:
                        continue
                    break
        return self._pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """Perform a single environment transition.

        Deterministic dynamics on a grid with border walls and optional interior
        walls (``self.walls``). The agent attempts to move one cell in the
        direction specified by ``action``. If the target cell is a wall or
        outside the interior bounds, the agent remains in place (collision).

        Actions
        - 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT

        Rewards
        - Step reward: ``self.rew.step`` on regular movement
        - Collision reward: ``self.rew.collision`` if movement is blocked
        - Goal reward: ``self.rew.goal`` when reaching ``self.cfg.goal``

        Termination
        - When the agent reaches the goal or ``self._t`` reaches ``self.max_steps``

        Returns
        - state: tuple[int, int] — new position (x, y)
        - reward: float — reward obtained on this transition
        - done: bool — whether the episode has terminated
        - info: dict — extra information (e.g., current timestep ``t``)
        """
        self._t += 1
        x, y = self._pos
        nx, ny = x, y
        if action == UP:
            ny = min(self.rows - 2, y + 1)
        elif action == DOWN:
            ny = max(1, y - 1)
        elif action == LEFT:
            nx = max(1, x - 1)
        elif action == RIGHT:
            nx = min(self.cols - 2, x + 1)

        # Attempted move
        bumped = False
        if (nx, ny) in self.walls:
            nx, ny = x, y
            bumped = True

        self._pos = (nx, ny)

        done = self._pos == self.cfg.goal or self._t >= self.max_steps
        reward = self.rew.goal if self._pos == self.cfg.goal else self.rew.step
        if bumped and self._pos != self.cfg.goal:
            reward = self.rew.collision

        info = {"t": self._t}
        return self._pos, float(reward), bool(done), info

    # Convenience helpers for agents
    def action_space(self) -> int:
        return len(ACTIONS)

    def state(self) -> Tuple[int, int]:
        return self._pos
