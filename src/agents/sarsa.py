from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

# Avoid circular import by not importing the environment at runtime.
# Import only for type checking if needed.
if TYPE_CHECKING:
    from src.envs.gridworld_v1 import GridworldV1  # pragma: no cover

# Define local action constants to decouple from the environment module.
ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT = 0, 1, 2, 3
ACTIONS = (ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT)


def sarsa_train(
    env: "GridworldV1",
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float | None = None,
    epsilon: float = 0.1,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    max_steps: int | None = None,
    # state-action repeat shaping
    state_action_repeat_penalty: float = 0.0,
    state_action_repeat_penalty_power: float = 1.0,
    state_action_repeat_penalty_mode: str = "episode",  # 'episode' or 'global'
    # anti-backtrack shaping
    anti_backtrack_cost: float = 0.0,
    # orientation-augmented state (key Q and pi by (s, prev_a|-1))
    use_orientation_state: bool = False,
    # goal-free shaping to discourage crossing the same corridor edge repeatedly
    edge_repeat_cost: float = 0.0,
    edge_repeat_cost_power: float = 1.0,
    edge_repeat_cost_mode: str = "episode",
    # selection-time novelty bias based on state visits (goal-free)
    selection_state_novel_coeff: float = 0.0,
    selection_state_novel_power: float = 1.0,
    selection_state_novel_mode: str = "episode",
    # selection-time novelty bias based on state-action repeats (goal-free)
    selection_sa_novel_coeff: float = 0.0,
    selection_sa_novel_power: float = 1.0,
    selection_sa_novel_mode: str = "episode",
) -> tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], np.ndarray]]:
    """Train an on-policy SARSA agent on GridworldV1.

    Args:
        env: Gridworld environment instance. Will be stepped/reset during training.
        episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor. If None, uses env.cfg.gamma.
        epsilon: Initial epsilon for epsilon-greedy exploration.
        epsilon_min: Minimum epsilon.
        epsilon_decay: Multiplicative decay applied per episode.
        max_steps: Cap steps per episode. If None, uses env.max_steps.

    Returns:
        - pi: Greedy policy dict mapping state (x, y) -> best action.
        - Q: Q-table mapping state (x, y) -> np.array of action values (len = len(ACTIONS)).
    """
    if gamma is None:
        gamma = float(env.cfg.gamma)
    if max_steps is None:
        max_steps = int(env.max_steps)

    # Q-table with zero init (optionally orientation-augmented)
    Q = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=float))

    rng = np.random.default_rng()

    def _next_from(s: Tuple[int, int], a: int) -> Tuple[int, int]:
        x, y = s
        nx, ny = x, y
        if a == ACT_UP:
            ny = min(env.rows - 2, y + 1)
        elif a == ACT_DOWN:
            ny = max(1, y - 1)
        elif a == ACT_LEFT:
            nx = max(1, x - 1)
        elif a == ACT_RIGHT:
            nx = min(env.cols - 2, x + 1)
        if (nx, ny) in getattr(env, "walls", set()):
            return (x, y)
        return (nx, ny)

    # No goal-based helpers in SARSA (purely on-policy + local env checks)

    def _valid_actions(s: Tuple[int, int]) -> list[int]:
        return [a for a in ACTIONS if _next_from(s, a) != s]

    def _key(s: Tuple[int, int], prev_a: int | None) -> tuple:
        return (s, (-1 if prev_a is None else int(prev_a))) if use_orientation_state else s

    def _greedy_tiebreak(s: Tuple[int, int], prev_a: int | None = None) -> int:
        q = Q[_key(s, prev_a)]
        if q.size == 0:
            return ACT_UP
        valids = _valid_actions(s)
        if not valids:
            return int(rng.integers(0, len(ACTIONS)))
        q_hat = q.copy()
        # Mask invalid actions to a very low value so they are never selected
        very_low = -1e12
        for i in range(len(q_hat)):
            if i not in valids:
                q_hat[i] = very_low
        # Selection-time novelty toward less-visited next states
        if selection_state_novel_coeff > 0.0:
            def state_count(st: Tuple[int, int]) -> int:
                return (episode_state_counts_state.get(st, 0) if selection_state_novel_mode == "episode" else global_state_counts.get(st, 0))
            for i in valids:
                s2 = _next_from(s, i)
                c = state_count(s2)
                q_hat[i] += float(selection_state_novel_coeff) / ((c + 1) ** float(selection_state_novel_power))
        # Selection-time novelty toward unseen (s,a)
        if selection_sa_novel_coeff > 0.0:
            def sa_count(sa_s: Tuple[int, int], sa_a: int) -> int:
                if selection_sa_novel_mode == "episode":
                    return episode_sa_counts.get((sa_s, sa_a), 0)
                return global_sa_counts.get((sa_s, sa_a), 0)
            for i in valids:
                csa = sa_count(s, i)
                q_hat[i] += float(selection_sa_novel_coeff) / ((csa + 1) ** float(selection_sa_novel_power))
        max_q = np.max(q_hat)
        cand = [i for i, val in enumerate(q_hat) if val == max_q]
        # Prefer candidates that are not the immediate opposite of the previous action (if any)
        if prev_a is not None:
            nonopps = [a for a in cand if not _is_opposite(a, prev_a)]
            if nonopps:
                return int(rng.choice(nonopps))
        return int(rng.choice(cand))

    def eps_greedy(s: Tuple[int, int], eps: float, prev_a: int | None = None) -> int:
        if rng.random() < eps:
            # Explore among valid moves that cause state change if possible
            valids = _valid_actions(s)
            if not valids:
                return int(rng.integers(0, len(ACTIONS)))
            # 1) Prefer actions never tried at (s, a)
            def sa_count(sa_s: Tuple[int, int], sa_a: int) -> int:
                if selection_sa_novel_mode == "episode":
                    return episode_sa_counts.get((sa_s, sa_a), 0)
                return global_sa_counts.get((sa_s, sa_a), 0)
            nsa = {a: sa_count(s, a) for a in valids}
            min_nsa = min(nsa.values())
            pool = [a for a in valids if nsa[a] == min_nsa]
            # 2) Among ties, prefer actions whose next state was less visited
            if len(pool) > 1:
                def state_count(st: Tuple[int, int]) -> int:
                    return (episode_state_counts_state.get(st, 0) if selection_state_novel_mode == "episode" else global_state_counts.get(st, 0))
                scored = [(a, state_count(_next_from(s, a))) for a in pool]
                minc = min(c for _, c in scored)
                pool = [a for a, c in scored if c == minc]
            # 3) Prefer non-opposite to avoid trivial backtracks
            if prev_a is not None:
                nonopps = [x for x in pool if not _is_opposite(x, prev_a)]
                if nonopps:
                    pool = nonopps
            return int(rng.choice(pool))
        # Greedy with smart tie-breaking, valid-action masking and novelty bias
        return _greedy_tiebreak(s, prev_a)

    def _is_opposite(a1: int, a2: int) -> bool:
        return (a1 == ACT_UP and a2 == ACT_DOWN) or \
               (a1 == ACT_DOWN and a2 == ACT_UP) or \
               (a1 == ACT_LEFT and a2 == ACT_RIGHT) or \
               (a1 == ACT_RIGHT and a2 == ACT_LEFT)

    # Track counts for shaping (state-action)
    global_sa_counts: Dict[Tuple[Tuple[int, int], int], int] = defaultdict(int)
    # Track counts for edge traversals (undirected) to discourage oscillations
    global_edge_counts: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = defaultdict(int)
    # Track counts for state visits to bias selection toward less-visited states
    global_state_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    for ep in range(episodes):
        s = env.reset()
        # Per-episode counters (must be defined before first eps_greedy call)
        episode_sa_counts: Dict[Tuple[Tuple[int, int], int], int] = defaultdict(int)
        episode_edge_counts: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = defaultdict(int)
        episode_state_counts_state: Dict[Tuple[int, int], int] = defaultdict(int)
        a = eps_greedy(s, epsilon, prev_a=None)
        steps = 0
        done = False
        prev_a: int | None = None
        while not done and steps < max_steps:
            s2, r, done, _info = env.step(a)
            # Shaping: small cost if we backtrack (current action opposite to previous)
            if anti_backtrack_cost > 0.0 and prev_a is not None and _is_opposite(a, prev_a):
                r = r - float(anti_backtrack_cost)
            # Edge-repeat penalty (goal-free): discourage traversing the same corridor back-and-forth
            if edge_repeat_cost > 0.0:
                # undirected edge key
                ekey = (s, s2) if s <= s2 else (s2, s)
                edge_counter = episode_edge_counts if edge_repeat_cost_mode == "episode" else global_edge_counts
                next_count = edge_counter[ekey] + 1
                # penalize only repeats (no penalty on first time over an edge)
                if next_count > 1:
                    r = r - float(edge_repeat_cost) * ((next_count - 1) ** float(edge_repeat_cost_power))
                edge_counter[ekey] = next_count
            # No goal-based potential shaping (user requested pure exploration)
            # Shaping: concave novelty bonus on (state, action)
            # Interpret state_action_repeat_penalty as the base magnitude of the novelty bonus
            # Bonus decreases with repeats: bonus = coeff / (count ** power)
            # Concave novelty bonus on (state, action) if enabled (goal-free)
            sa_key = (s, a)
            if state_action_repeat_penalty > 0.0:
                base_count = (episode_sa_counts[sa_key] if state_action_repeat_penalty_mode == "episode" else global_sa_counts[sa_key])
                next_count = base_count + 1
                coeff = float(state_action_repeat_penalty)
                power = float(state_action_repeat_penalty_power)
                bonus = coeff / (next_count ** power)
                r = r + bonus
            # Always bump SA counters (used for selection-time novelty)
            episode_sa_counts[sa_key] += 1
            global_sa_counts[sa_key] += 1
            # Update state visit counts for selection bias
            if selection_state_novel_coeff > 0.0:
                if selection_state_novel_mode == "episode":
                    episode_state_counts_state[s2] += 1
                else:
                    global_state_counts[s2] += 1
            # choose next action on-policy
            a2 = eps_greedy(s2, epsilon, prev_a=a) if not done else 0
            # SARSA update
            s_key = _key(s, prev_a)
            s2_key = _key(s2, a if not done else prev_a)
            td_target = r + (0.0 if done else gamma * Q[s2_key][a2])
            td_error = td_target - Q[s_key][a]
            Q[s_key][a] += alpha * td_error
            prev_a = a
            s, a = s2, a2
            steps += 1
        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Derive greedy policy
    pi: Dict = {}
    try:
        from src.utils.gridworld_core import interior_states as _core_interior_states
        _walls = set(getattr(env, "walls", set()))
        states = list(_core_interior_states(env.cfg.cols, env.cfg.rows, _walls))
    except Exception:
        states = list({k[0] if (use_orientation_state and isinstance(k, tuple) and isinstance(k[0], tuple)) else k for k in Q.keys()})

    if use_orientation_state:
        # Build oriented policy for all prev orientations including -1 (None)
        prevs = [-1, ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT]
        for s in states:
            for pa in prevs:
                q = Q[(s, pa)]
                if q.size:
                    pi[(s, pa)] = _greedy_tiebreak(s, None if pa == -1 else pa)
    else:
        for s in states:
            q = Q[s]
            if q.size:
                pi[s] = _greedy_tiebreak(s)
            else:
                pi[s] = _greedy_tiebreak(s)

    # Reset environment after training 
    env.reset()

    return pi, Q
