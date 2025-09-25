from __future__ import annotations

"""Gridworld V1 environment.

Deterministic 2D grid environment with optional wall layouts (corridor/maze).
This module defines:
- Constants for actions: ``UP``, ``DOWN``, ``LEFT``, ``RIGHT`` and ``ACTIONS``.
- Dataclasses ``Rewards`` and ``GridCfg`` for environment configuration.
- Class ``GridworldV1`` implementing a minimal RL environment API with
  ``reset`` and ``step`` methods, plus convenience helpers.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Set, List, Optional, Literal, TYPE_CHECKING
import json
from pathlib import Path

import numpy as np
from src.utils.gridworld_core import GWConfig
from src.agents.value_iteration import value_iteration
from src.agents.policy_iteration import policy_iteration
from src.agents.sarsa import sarsa_train
import threading
try:
    import arcade  # type: ignore
    from pyglet.window import key as pygkey  # type: ignore
except Exception:
    arcade = None
    pygkey = None


Action = int
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)


@dataclass
class Rewards:
    """Rewards used by the environment.

    Attributes:
        step: Reward obtained on a standard (non-terminal, non-collision) step.
        goal: Reward provided when the agent reaches the goal state.
        collision: Reward when an attempted move is blocked (by a wall/border).
    """
    step: float = -1.0
    goal: float = 10.0
    collision: float = 0.0  # set to -5.0 to punish bumping walls


@dataclass
class GridCfg:
    """Gridworld configuration parameters.

    Attributes:
        cols: Grid width (number of columns), including border walls.
        rows: Grid height (number of rows), including border walls.
        max_steps: Maximum steps per episode before termination.
        start: Start state (x, y) coordinate inside the interior.
        goal: Goal state (x, y) coordinate inside the interior.
        randomize_start_neighbors: If True, randomize start among start and its
            4-neighborhood (filtered to valid interior cells not equal to goal).
        rewards: Reward specification used by the environment.
        gamma: Discount factor (exposed for planning/reference).
        layout: Optional named layout; one of ``"corridor"`` or ``"maze"``.
        walls: Optional explicit interior wall coordinates (x, y).
        rng_seed: Optional seed for the environment RNG.
    """
    cols: int = 11
    rows: int = 11
    max_steps: int = 200
    start: Tuple[int, int] = (1, 1)
    goal: Tuple[int, int] = (9, 9)
    randomize_start_neighbors: bool = False
    rewards: Rewards = field(default_factory=Rewards)
    gamma: float = 0.95
    layout: Optional[Literal["corridor", "maze"]] = None  # e.g., "corridor" or "maze"
    walls: Optional[List[Tuple[int, int]]] = None  # interior walls
    rng_seed: Optional[int] = None


class GridworldV1:
    """Deterministic 2D gridworld with borders and optional interior walls.

    - Actions: ``UP``, ``DOWN``, ``LEFT``, ``RIGHT`` (integers 0..3)
    - Transitions: move one cell; bumping a wall keeps agent in place
    - Rewards: step, goal, collision
    - Termination: reaching goal or exceeding ``max_steps``
    """

    def __init__(self, cfg: GridCfg):
        """Initialize the environment from a ``GridCfg``.

        Args:
            cfg: Configuration with grid shape, rewards, start/goal, layout,
                and optional walls and RNG seed.
        """
        self.cfg = cfg
        self.cols = cfg.cols
        self.rows = cfg.rows
        self.max_steps = cfg.max_steps
        self.rew = cfg.rewards
        self._t = 0
        self._pos = cfg.start
        # RNG
        self.rng = np.random.default_rng(self.cfg.rng_seed)
        # Build interior walls set
        self.walls: Set[Tuple[int, int]] = set()
        self._build_walls()
        # Optional planning/RL artifacts managed at env level so UIs can query them
        self.pi_vi: Dict[Tuple[int, int], int] = {}
        self.pi_pi: Dict[Tuple[int, int], int] = {}
        self.pi_sarsa: Dict[Tuple[int, int], int] = {}
        # Parameters for SARSA training and shaping (optional)
        # Keys recognized by this environment and `sarsa_train`:
        # - Planning/learning:
        #   gamma, episodes, alpha, epsilon, epsilon_min, epsilon_decay, max_steps
        # - Orientation/state encoding:
        #   use_orientation_state (bool)
        # - Shaping penalties (repeat/novelty):
        #   state_action_repeat_penalty, state_action_repeat_penalty_power,
        #   state_action_repeat_penalty_mode ("episode" | "global")
        #   visit_penalty, visit_penalty_power, visit_penalty_mode (deprecated fallbacks)
        # - Anti-backtrack cost (applied in trainer/selector):
        #   anti_backtrack_cost
        # - Edge repeat costs:
        #   edge_repeat_cost, edge_repeat_cost_power, edge_repeat_cost_mode
        # - Novelty in action selection (not the same as shaping):
        #   selection_state_novel_coeff, selection_state_novel_power, selection_state_novel_mode
        #   selection_sa_novel_coeff, selection_sa_novel_power, selection_sa_novel_mode
        self.sarsa_params: Dict[str, Any] = {}
        # Current method selection for consumers ("vi", "pi", or "sarsa")
        self.method: Literal["vi", "pi", "sarsa"] = "vi"
        # SARSA shaping counters and last penalty (for HUD/analytics)
        self.global_sa_counts: Dict[Tuple[Tuple[int, int], int], int] = {}
        self.episode_sa_counts: Dict[Tuple[Tuple[int, int], int], int] = {}
        self.last_penalty: float = 0.0
        # Optional back-reference to a renderer window (set by the window itself)
        if TYPE_CHECKING:
            from src.engine.gridworld_renderer import GridworldV1Window  # type: ignore
        self.window: Optional["GridworldV1Window"] = None
        # Track last action taken (used for orientation-aware selection and backtrack avoidance)
        self.prev_action: Optional[int] = None

    @staticmethod
    def from_json(path: Path) -> "GridworldV1":
        """Load a ``GridworldV1`` from a JSON file.

        The JSON supports keys: ``grid.cols``, ``grid.rows``, ``max_steps``,
        ``start``, ``goal``, ``randomize_start_neighbors``, ``rewards``
        (``step``, ``goal``, ``collision``), ``gamma``, ``layout``, ``walls``,
        and ``rng_seed``.

        Args:
            path: Path to a JSON configuration file.

        Returns:
            A constructed ``GridworldV1`` instance.
        """
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
            rng_seed=data.get("rng_seed"),
        )
        return GridworldV1(gc)

    # --- Walls generation helpers ---
    def _build_walls(self) -> None:
        """Construct interior walls based on the current configuration.

        - For ``layout == 'corridor'``: create a single horizontal corridor.
        - For ``layout == 'maze'``: generate a random perfect maze.
        - Else if ``cfg.walls`` is provided: use those explicit walls.
        """
        self.walls.clear()
        if self.cfg.layout == "corridor":
            cy = self.rows // 2
            for y in range(1, self.rows - 1):
                for x in range(1, self.cols - 1):
                    if y != cy:
                        self.walls.add((x, y))
            # Start/goal aligned to corridor
            self.cfg.start = (1, cy)
            self.cfg.goal = (self.cols - 2, cy)
            self._pos = self.cfg.start
        elif self.cfg.layout == "maze":
            self._generate_maze_walls()
            # Place start/goal on passages
            s = (1, 1)
            g = (self.cols - 2, self.rows - 2)
            passages = self._passages_cache
            if s not in passages:
                s = next(iter(passages))
            if g not in passages:
                # choose farthest from start by Manhattan
                sx, sy = s
                g = max(passages, key=lambda xy: abs(xy[0] - sx) + abs(xy[1] - sy))
            self.cfg.start, self.cfg.goal = s, g
            self._pos = self.cfg.start
        elif self.cfg.walls:
            for x, y in self.cfg.walls:
                if 1 <= x < self.cols - 1 and 1 <= y < self.rows - 1:
                    self.walls.add((x, y))

    def _generate_maze_walls(self) -> None:
        """Generate a perfect maze with a recursive backtracker on odd cells.

        The interior is initialized as walls and then passages are carved on the
        odd grid coordinates using depth-first search.
        """
        # Recursive backtracker on odd cells
        cols, rows = self.cols, self.rows
        visited: Set[Tuple[int, int]] = set()
        passages: Set[Tuple[int, int]] = set()
        def neighbors(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
            x, y = cell
            cand = [(x+2,y),(x-2,y),(x,y+2),(x,y-2)]
            return [(nx,ny) for nx,ny in cand if 1 <= nx < cols-1 and 1 <= ny < rows-1 and (nx%2==1 and ny%2==1)]

        # Initialize all interior as walls; passages will be carved
        for y in range(1, rows-1):
            for x in range(1, cols-1):
                self.walls.add((x,y))

        # Pick starting odd cell
        odd_cells = [(x,y) for y in range(1, rows-1) for x in range(1, cols-1) if x%2==1 and y%2==1]
        start = (1,1) if (1,1) in odd_cells else odd_cells[0]
        stack = [start]
        visited.add(start)
        passages.add(start)
        self.walls.discard(start)

        while stack:
            cur = stack[-1]
            nbrs = [n for n in neighbors(cur) if n not in visited]
            if not nbrs:
                stack.pop()
                continue
            n = nbrs[int(self.rng.integers(0, len(nbrs)))]
            # carve wall between cur and n
            wx = (cur[0] + n[0]) // 2
            wy = (cur[1] + n[1]) // 2
            for cell in [n, (wx, wy)]:
                passages.add(cell)
                self.walls.discard(cell)
            visited.add(n)
            stack.append(n)

        # cache passages for start/goal placement
        self._passages_cache = passages

    def regenerate_walls(self) -> None:
        """Regenerate walls for layouts that support procedural walls.

        Applies when ``cfg.layout`` is either ``"maze"`` or ``"corridor"``.
        """
        if self.cfg.layout in {"maze", "corridor"}:
            self._build_walls()

    def toggle_layout(self) -> None:
        """Toggle between 'maze' and 'corridor' layouts and rebuild walls."""
        self.cfg.layout = "maze" if (self.cfg.layout != "maze") else "corridor"
        self.regenerate_walls()

    # --- Planning/RL helpers (env-level so renderers can just call) ---
    def compute_policies(self) -> None:
        """Compute VI and PI policies from the current environment configuration.

        Populates ``self.pi_vi`` and ``self.pi_pi``.
        """
        gwcfg = GWConfig(
            cols=self.cols,
            rows=self.rows,
            start=self.cfg.start,
            goal=self.cfg.goal,
            step_reward=self.rew.step,
            goal_reward=self.rew.goal,
            collision_reward=self.rew.collision,
            gamma=self.cfg.gamma,
            walls=set(self.walls),
        )
        _, self.pi_vi = value_iteration(gwcfg)
        _, self.pi_pi = policy_iteration(gwcfg)

    def train_sarsa(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Train SARSA on this environment, update ``pi_sarsa`` with greedy policy.

        Parameters in ``params`` (all optional) include typical SARSA hyperparameters
        and shaping/selection knobs referenced elsewhere in this env via
        ``self.sarsa_params``. Recognized keys:
        - gamma, episodes, alpha, epsilon, epsilon_min, epsilon_decay, max_steps
        - use_orientation_state
        - state_action_repeat_penalty, state_action_repeat_penalty_power,
          state_action_repeat_penalty_mode (episode|global)
        - edge_repeat_cost, edge_repeat_cost_power, edge_repeat_cost_mode
        - selection_state_novel_coeff, selection_state_novel_power, selection_state_novel_mode
        - selection_sa_novel_coeff, selection_sa_novel_power, selection_sa_novel_mode

        The resulting greedy policy may contain keys of the form
        ``((x, y), prev_action)`` when ``use_orientation_state`` is enabled.
        """
        p = params or self.sarsa_params or {}
        gamma = float(p.get("gamma", self.cfg.gamma))
        episodes = int(p.get("episodes", 500))
        alpha = float(p.get("alpha", 0.5))
        epsilon = float(p.get("epsilon", 0.1))
        epsilon_min = float(p.get("epsilon_min", 0.01))
        epsilon_decay = float(p.get("epsilon_decay", 0.995))
        max_steps = int(p.get("max_steps", self.max_steps))
        # penalties and novelty
        kwargs = dict(
            state_action_repeat_penalty=float(p.get("state_action_repeat_penalty", p.get("visit_penalty", 0.0))),
            state_action_repeat_penalty_power=float(p.get("state_action_repeat_penalty_power", p.get("visit_penalty_power", 1.0))),
            state_action_repeat_penalty_mode=str(p.get("state_action_repeat_penalty_mode", p.get("visit_penalty_mode", "episode"))),
            anti_backtrack_cost=float(p.get("anti_backtrack_cost", 0.0)),
            use_orientation_state=bool(p.get("use_orientation_state", False)),
            edge_repeat_cost=float(p.get("edge_repeat_cost", 0.0)),
            edge_repeat_cost_power=float(p.get("edge_repeat_cost_power", 1.0)),
            edge_repeat_cost_mode=str(p.get("edge_repeat_cost_mode", "episode")),
            selection_state_novel_coeff=float(p.get("selection_state_novel_coeff", 0.0)),
            selection_state_novel_power=float(p.get("selection_state_novel_power", 1.0)),
            selection_state_novel_mode=str(p.get("selection_state_novel_mode", "episode")),
            selection_sa_novel_coeff=float(p.get("selection_sa_novel_coeff", 0.0)),
            selection_sa_novel_power=float(p.get("selection_sa_novel_power", 1.0)),
            selection_sa_novel_mode=str(p.get("selection_sa_novel_mode", "episode")),
        )
        pi_sarsa, _Q = sarsa_train(
            self,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            max_steps=max_steps,
            **kwargs,
        )
        self.pi_sarsa = pi_sarsa

    def recompute_policies_and_maybe_train(self) -> None:
        """Recompute VI/PI policies and optionally retrain SARSA.

        This centralizes policy updates at the environment level so engines can
        just call this method when geometry or rewards change.
        """
        self.compute_policies()
        if self.sarsa_params:
            try:
                self.train_sarsa()
            except Exception:
                # Training is optional; ignore errors to keep UI responsive
                pass

    # --- Mechanics helpers to be called by UIs ---
    def on_geometry_changed(self) -> None:
        """Call when geometry (walls/layout/start/goal) changes.

        Recomputes policies (and optionally trains SARSA) and resets the env.
        """
        self.recompute_policies_and_maybe_train()
        self.reset()

    def on_goal_reached(self) -> None:
        """Call when an episode ends by reaching the goal.

        If layout is a maze, regenerate a new maze; then recompute policies and
        reset the environment.
        """
        if self.cfg.layout == "maze":
            self.regenerate_walls()
        self.on_geometry_changed()

    # --- Method selection helpers ---
    def set_method(self, method: str) -> None:
        """Set current method. Falls back to "vi" if invalid."""
        m = str(method).lower()
        self.method = m if m in ("vi", "pi", "sarsa") else "vi"

    def toggle_method(self) -> None:
        """Cycle method in order VI -> PI -> SARSA -> VI."""
        order = ["vi", "pi", "sarsa"]
        try:
            i = order.index(self.method)
        except ValueError:
            i = 0
        self.method = order[(i + 1) % len(order)]

    # --- Optional SARSA shaping helpers and step wrapper ---
    def _compute_sa_penalty(self, s: Tuple[int, int], a: int) -> float:
        """Compute optional SARSA shaping penalty for a state-action pair.

        Uses the configured counters (episode/global) and the parameters from
        ``self.sarsa_params`` to determine the penalty magnitude.
        Returns 0.0 if shaping is disabled or method != "sarsa".
        """
        if self.method != "sarsa":
            return 0.0
        cfgp = self.sarsa_params or {}
        coeff = float(cfgp.get("state_action_repeat_penalty", cfgp.get("visit_penalty", 0.0)))
        if coeff <= 0.0:
            return 0.0
        power = float(cfgp.get("state_action_repeat_penalty_power", cfgp.get("visit_penalty_power", 1.0)))
        mode = str(cfgp.get("state_action_repeat_penalty_mode", cfgp.get("visit_penalty_mode", "episode")))
        sa_key = (s, a)
        if mode == "global":
            base_count = self.global_sa_counts.get(sa_key, 0)
        else:
            base_count = self.episode_sa_counts.get(sa_key, 0)
        next_count = base_count + 1
        return coeff * (next_count ** power)

    def _bump_sa_counter(self, s: Tuple[int, int], a: int) -> None:
        cfgp = self.sarsa_params or {}
        mode = str(cfgp.get("state_action_repeat_penalty_mode", cfgp.get("visit_penalty_mode", "episode")))
        sa_key = (s, a)
        if mode == "global":
            self.global_sa_counts[sa_key] = self.global_sa_counts.get(sa_key, 0) + 1
        else:
            self.episode_sa_counts[sa_key] = self.episode_sa_counts.get(sa_key, 0) + 1

    def step_with_shaping(self, a: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """Step wrapper that applies optional SARSA shaping penalty.

        Returns the same tuple as ``step``, with reward adjusted by the penalty.
        Adds ``{"penalty": last_penalty}`` to the info dict.
        """
        s = self.state()
        penalty = self._compute_sa_penalty(s, a)
        self._bump_sa_counter(s, a)
        s2, r, done, info = self.step(a)
        self.last_penalty = float(penalty)
        info = dict(info or {})
        info["penalty"] = self.last_penalty
        return s2, float(r - penalty), bool(done), info

    # --- API ---
    def reset(self, rng: np.random.Generator | None = None) -> Tuple[int, int]:
        """Reset the environment to a start state.

        If ``randomize_start_neighbors`` is True, the initial position is drawn
        uniformly from the 4-neighborhood of ``cfg.start`` (including start),
        filtered to valid interior cells not equal to the goal and not walls.

        Also clears episode-level counters used for SARSA shaping/selection and
        resets ``self.prev_action`` (orientation/backtrack memory).

        Args:
            rng: Optional RNG to use for sampling the start position.

        Returns:
            The initial state ``(x, y)``.
        """
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
                rng = self.rng
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
        # Clear episode counters and orientation memory for a fresh episode
        self.episode_sa_counts.clear()
        self.prev_action = None
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
        """Return the number of discrete actions available."""
        return len(ACTIONS)

    def state(self) -> Tuple[int, int]:
        """Get the current agent position as ``(x, y)``."""
        return self._pos

    # --- Planning helpers exposed for engines/renderers ---
    def current_policy(self) -> Dict[Tuple[int, int], int]:
        """Return the currently selected policy mapping based on ``self.method``.

        For ``method == 'sarsa'`` and when orientation-aware training is enabled
        (``sarsa_params.use_orientation_state``), the policy may also contain
        keys of the form ``((x, y), prev_action)``. Callers that wish to use
        orientation-aware policies should pass ``prev_action`` into
        ``select_action`` rather than looking up this mapping directly.
        """
        if self.method == "vi":
            return self.pi_vi
        if self.method == "pi":
            return self.pi_pi
        if self.method == "sarsa":
            return self.pi_sarsa
        return self.pi_vi

    def next_from(self, s: Tuple[int, int], a: int) -> Tuple[int, int]:
        """Pure next-state model (no mutation), mirroring env movement rules.

        Applies interior bounds and wall collisions without changing ``self._pos``.
        """
        x, y = s
        nx, ny = x, y
        if a == UP:
            ny = min(self.rows - 2, y + 1)
        elif a == DOWN:
            ny = max(1, y - 1)
        elif a == LEFT:
            nx = max(1, x - 1)
        elif a == RIGHT:
            nx = min(self.cols - 2, x + 1)
        if (nx, ny) in self.walls:
            return (x, y)
        return (nx, ny)

    def greedy_to_goal(self, s: Tuple[int, int]) -> int:
        """Heuristic: prefer a move that reduces L1 distance to the goal.

        Tries axis-aligned moves toward the goal first, then any valid move.
        Returns an action that changes state if possible; defaults to ``UP``.
        """
        gx, gy = self.cfg.goal
        x, y = s
        candidates: List[int] = []
        if gx > x:
            candidates.append(RIGHT)
        elif gx < x:
            candidates.append(LEFT)
        if gy > y:
            candidates.append(UP)
        elif gy < y:
            candidates.append(DOWN)
        candidates += [UP, DOWN, LEFT, RIGHT]
        seen = set()
        ordered = [a for a in candidates if not (a in seen or seen.add(a))]
        for a in ordered:
            if self.next_from(s, a) != s:
                return a
        return UP

    def _is_opposite(self, a1: int, a2: int) -> bool:
        return (a1 == UP and a2 == DOWN) or \
               (a1 == DOWN and a2 == UP) or \
               (a1 == LEFT and a2 == RIGHT) or \
               (a1 == RIGHT and a2 == LEFT)

    def select_action(self, s: Tuple[int, int], prev_action: Optional[int] = None) -> int:
        """Select an action under the current method with robust fallbacks.

        Behavior:
        - Try the corresponding policy action for the current method.
          For SARSA with ``use_orientation_state``, try ``((s, prev_action_or_-1))``.
        - If missing or blocked, choose a valid fallback.
          * SARSA: prefer actions with the lowest observed state-action count
            (based on the configured counter mode), avoiding immediate
            backtracks to ``prev_action`` when possible.
          * VI/PI: prefer a move that reduces L1 distance to the goal.
        """
        pi = self.current_policy() or {}
        use_orient = bool((self.sarsa_params or {}).get("use_orientation_state", False))

        # Initial policy suggestion
        a: Optional[int] = None
        if self.method == "sarsa" and use_orient:
            pa = prev_action if prev_action is not None else -1
            a = pi.get((s, pa), None)  # type: ignore[index]
        if a is None:
            a = pi.get(s, None)  # type: ignore[index]

        # Build list of valid actions from state
        valids = [act for act in (UP, DOWN, LEFT, RIGHT) if self.next_from(s, act) != s]

        def sarsa_fallback() -> int:
            if not valids:
                return UP
            # Select by minimal SA count (novelty) using the same mode as shaping
            cfgp = self.sarsa_params or {}
            mode = str(cfgp.get("state_action_repeat_penalty_mode", cfgp.get("visit_penalty_mode", "episode")))
            def count_for(act: int) -> int:
                key = (s, act)
                if mode == "global":
                    return int(self.global_sa_counts.get(key, 0))
                return int(self.episode_sa_counts.get(key, 0))
            counts = {act: count_for(act) for act in valids}
            minc = min(counts.values())
            pool = [act for act in valids if counts[act] == minc]
            if prev_action is not None:
                non_opps = [act for act in pool if not self._is_opposite(act, prev_action)]
                pool = non_opps or pool
            # tie-break deterministically for reproducibility by ordering
            for cand in (UP, DOWN, LEFT, RIGHT):
                if cand in pool:
                    return cand
            return pool[0]

        # If no policy action or it is blocked/backtracks, use fallback
        if a is None:
            if self.method == "sarsa":
                a = sarsa_fallback()
            else:
                a = self.greedy_to_goal(s)
        if self.next_from(s, a) == s or (prev_action is not None and self._is_opposite(a, prev_action)):
            if self.method == "sarsa":
                a = sarsa_fallback()
            else:
                if valids:
                    gx, gy = self.cfg.goal
                    def next_d(act: int) -> int:
                        nx, ny = self.next_from(s, act)
                        return abs(nx - gx) + abs(ny - gy)
                    a = min(valids, key=lambda act: next_d(act))
        return a

    # --- Step helpers to support engines ---
    def step_under_policy(self) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any], int]:
        """Select an action per current policy and step the environment.

        Uses ``self.prev_action`` to provide orientation/backtrack awareness
        when selecting actions (especially under SARSA with orientation state).

        Returns (next_state, reward, done, info, action).
        """
        s = self.state()
        pa = self.prev_action
        a = self.select_action(s, pa)
        s2, r, done, info = self.step_with_shaping(a)
        # Update orientation memory
        self.prev_action = a if not done else None
        return s2, r, done, info, a

    def apply_manual_action(self, a: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """Apply a manual action 'a' and return the step result (with shaping).

        Also updates ``self.prev_action`` to reflect the last chosen action.
        """
        s2, r, done, info = self.step_with_shaping(a)
        self.prev_action = a if not done else None
        return s2, r, done, info

    # --- UI-integrated helpers (use env.window when available) ---
    def _ui_reset_episode(self) -> None:
        w = getattr(self, "window", None)
        if not w:
            return
        # Rebuild visuals and clear overlays/heatmap counters
        try:
            w._rebuild_tiles()
        except Exception:
            pass
        try:
            w._clear_heatmap()
        except Exception:
            pass
        # reset window-level episodic state
        try:
            w.total_reward = 0.0
            w.done = False
            w._accum = 0.0
            w.autoplay = True
            w._frozen_state = None
            w._move_sprite_to_state()
        except Exception:
            pass
        # reset env orientation memory too
        self.prev_action = None

    def trigger_sarsa_training(self) -> None:
        """Kick off SARSA training asynchronously and update UI via window."""
        w = getattr(self, "window", None)
        if sarsa_train is None or not w or getattr(w, "training", False):
            return
        try:
            w.training = True
            w.autoplay = False
            w._frozen_state = self.state()
            w._training_text.text = "Training SARSA..."
        except Exception:
            pass

        def run():
            try:
                self.train_sarsa()
            finally:
                def apply_ui_changes(_dt: float):
                    try:
                        w.training = False
                        w._training_text.text = ""
                    except Exception:
                        pass
                    # Rebuild visuals and recompute policies before reset
                    try:
                        w._rebuild_tiles()
                    except Exception:
                        pass
                    try:
                        self.recompute_policies_and_maybe_train()
                    except Exception:
                        try:
                            self.compute_policies()
                        except Exception:
                            pass
                    self.reset()
                    try:
                        w._clear_heatmap()
                        w.total_reward = 0.0
                        w.done = False
                        w._accum = 0.0
                        w.autoplay = True
                        w._frozen_state = None
                        w._move_sprite_to_state()
                    except Exception:
                        pass
                    self.prev_action = None

                # Schedule via arcade if available; else call directly
                if arcade is not None:
                    arcade.schedule_once(apply_ui_changes, 0)
                else:
                    apply_ui_changes(0)

        threading.Thread(target=run, daemon=True).start()

    def on_key_press(self, symbol: int) -> None:
        """Handle keyboard controls, updating both env and window as needed.

        Controls
        - Space: toggle autoplay (window only)
        - R: reset episode (env + UI)
        - M: toggle method (VI -> PI -> SARSA), train SARSA when selected
        - L: toggle layout (maze/corridor), rebuild geometry and recompute policies
        - N: new maze (when in maze layout)
        - Arrows/WASD: step manually one action

        Note: The renderer no longer maintains ``prev_action`` or
        ``_last_penalty``; overlays should read from this env when necessary.
        """
        w = getattr(self, "window", None)
        # Space: toggle autoplay
        if pygkey is not None and symbol == pygkey.SPACE and w is not None:
            w.autoplay = not w.autoplay
            return
        # R: Reset episode (env + UI)
        if pygkey is not None and symbol in (pygkey.R,):
            self.reset()
            self._ui_reset_episode()
            return
        # M: Toggle method; if SARSA, start async training
        if pygkey is not None and symbol in (pygkey.M,):
            try:
                self.toggle_method()
            except Exception:
                try:
                    self.set_method("vi")
                except Exception:
                    pass
            if getattr(self, "method", "vi") == "sarsa":
                self.trigger_sarsa_training()
            return
        # L: Toggle layout and rebuild
        if pygkey is not None and symbol in (pygkey.L,):
            try:
                self.toggle_layout()
            except Exception:
                self.cfg.layout = "maze" if (self.cfg.layout != "maze") else "corridor"
                self.regenerate_walls()
            try:
                self.on_geometry_changed()
            except Exception:
                self.recompute_policies_and_maybe_train()
                self.reset()
            self._ui_reset_episode()
            return
        # N: New maze (only when layout == 'maze')
        if pygkey is not None and symbol in (pygkey.N,):
            if self.cfg.layout == "maze":
                self.regenerate_walls()
                try:
                    self.on_geometry_changed()
                except Exception:
                    self.recompute_policies_and_maybe_train()
                    self.reset()
                self._ui_reset_episode()
            return
        # Manual steps via env
        a = None
        if pygkey is not None and (symbol == pygkey.UP or symbol == pygkey.W):
            a = UP
        elif pygkey is not None and (symbol == pygkey.DOWN or symbol == pygkey.S):
            a = DOWN
        elif pygkey is not None and (symbol == pygkey.LEFT or symbol == pygkey.A):
            a = LEFT
        elif pygkey is not None and (symbol == pygkey.RIGHT or symbol == pygkey.D):
            a = RIGHT
        if a is not None:
            if w is not None and getattr(w, "done", False):
                return
            s = self.state()
            s2, r, done, info = self.apply_manual_action(a)
            # Update renderer UI counters if available
            if w is not None:
                key = (s, a)
                tries, _lr, _ls2 = w.sa_debug.get(key, (0, 0.0, s))
                w.sa_debug[key] = (tries + 1, float(r), s2)
                w.total_reward = getattr(w, "total_reward", 0.0) + r
                w.done = bool(done)
                try:
                    w._move_sprite_to_state()
                except Exception:
                    pass
                if w.done and self.state() == self.cfg.goal:
                    try:
                        self.on_goal_reached()
                    except Exception:
                        if self.cfg.layout == "maze":
                            self.regenerate_walls()
                        self.recompute_policies_and_maybe_train()
                        self.reset()
                    self._ui_reset_episode()
