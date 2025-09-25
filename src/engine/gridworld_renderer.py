from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import arcade

from src.envs.gridworld_v1 import GridworldV1, UP, DOWN, LEFT, RIGHT

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "images"


@dataclass
class VizCfg:
    cell_size: int = 32
    step_interval: float = 0.12  # seconds between auto-steps


class GridworldV1Window(arcade.Window):
    """Arcade window responsible exclusively for visualization/UX.

    Responsibilities (UI only):
    - Render grid tiles (floor, walls, goal) and the actor sprite.
    - Draw overlays (status text, debug info, heatmap, fog-of-war).
    - Maintain window-local UI state (autoplay flag, accumulated dt, HUD text).
    - Delegate ALL environment mechanics (policy selection, stepping, resets,
      training, layout toggles, key handling) to the environment (`GridworldV1`).

    Notes:
    - Input handling is delegated by binding `on_key_press` dynamically to
      `env.on_key_press` during initialization (no handler is defined on the
      class). This keeps the renderer free of environment logic.
    - The environment holds a back-reference `env.window` to update visual
      elements when mechanics change (e.g., after training or geometry updates).
    """
    def __init__(
        self,
        env: GridworldV1,
        viz: VizCfg,
    ):
        """Initialize the visualization window.

        Parameters
        - env: `GridworldV1` instance (pre-configured). Owns all mechanics:
          stepping, policies (VI/PI/SARSA), training, toggles, and key handling.
        - viz: Visualization configuration (e.g., `cell_size`, `step_interval`).

        Behavior
        - Builds textures and sprites for the grid and actor.
        - Sets up overlay texts and heatmap state.
        - Binds `env.window = self` for UI callbacks from the environment.
        - Delegates keyboard events by assigning `self.on_key_press` to
          `env.on_key_press` (no key handling method is defined on the class).
        """
        width = env.cols * viz.cell_size
        height = env.rows * viz.cell_size
        super().__init__(width=width, height=height, title="Gridworld V1 (Planning)", resizable=False)

        arcade.set_background_color(arcade.color.BLACK)

        self.env = env
        # Visualization config and environment reference.
        # Policies, method selection and stepping are owned by the env.
        self.viz = viz

        # Textures
        self.tex_floor = arcade.load_texture(str(ASSETS / "floor.png"))
        self.tex_wall = arcade.load_texture(str(ASSETS / "wall.png"))
        self.tex_goal = arcade.load_texture(str(ASSETS / "exit.png"))
        self.tex_actor = arcade.load_texture(str(ASSETS / "actor.png"))

        cs = viz.cell_size

        self.floor_sprites = arcade.SpriteList()
        self.wall_sprites = arcade.SpriteList()
        self.goal_sprites = arcade.SpriteList()
        self.actor_sprites = arcade.SpriteList()

        s_actor = cs / max(1, self.tex_actor.width)

        self._rebuild_tiles()

        # Actor
        ax, ay = self.env.state()
        self.actor = arcade.Sprite()
        self.actor.texture = self.tex_actor
        self.actor.center_x = ax * cs + cs / 2
        self.actor.center_y = ay * cs + cs / 2
        self.actor.scale = s_actor
        self.actor_sprites.append(self.actor)

        # Control
        self.autoplay = True
        self._accum = 0.0
        self.done = False
        self.total_reward = 0.0

        # Overlay
        self._overlay = arcade.Text("", 10, height - 22, arcade.color.WHITE, 14)
        self._overlay2 = arcade.Text("", 10, height - 42, arcade.color.LIGHT_GRAY, 12)
        # Third overlay line for per-state action debug
        self._overlay3 = arcade.Text("", 10, height - 62, arcade.color.GRAY, 11)
        # Move training text further down to avoid overlap with _overlay3
        self._training_text = arcade.Text("", 10, height - 82, arcade.color.YELLOW, 12)
        self._hint = arcade.Text("Space: autoplay  |  R: reset  |  M: switch method  |  L: toggle layout  |  N: new maze", 10, 6, arcade.color.LIGHT_GRAY, 12)

        # Training UI state (used only for overlays/flow control in the window)
        self.training: bool = False
        self._frozen_state: Optional[Tuple[int, int]] = None

        # Heatmap state
        self._init_heatmap()
        self._mark_visit()
        # Last penalty is read from env.last_penalty when rendering overlays
        # Per-state action debug info shown on overlay: (tries, last_reward, last_next_state)
        self.sa_debug: Dict[Tuple[Tuple[int, int], int], Tuple[int, float, Tuple[int, int]]] = {}
        self.show_action_debug: bool = True

        # Link environment back to this window (optional reference)
        try:
            self.env.window = self  # type: ignore[attr-defined]
        except Exception:
            pass

        # Dynamically bind keyboard handler to delegate to env (no handler is
        # defined on the class). Keeps environment logic outside the renderer.
        try:
            self.on_key_press = lambda symbol, modifiers: self.env.on_key_press(symbol)  # type: ignore[assignment]
        except Exception:
            pass

    def _action_label(self, a: int) -> str:
        return {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R'}.get(a, '?')

    def _update_action_debug_overlay(self) -> None:
        if not self.show_action_debug or self.training:
            self._overlay3.text = ""
            return
        s = self._frozen_state if (self.training and self._frozen_state is not None) else self.env.state()
        valids = [a for a in (UP, DOWN, LEFT, RIGHT) if self.env.next_from(s, a) != s]
        if not valids:
            self._overlay3.text = "valid: -"
            return
        parts = []
        for a in valids:
            key = (s, a)
            tries, last_r, last_s2 = self.sa_debug.get(key, (0, 0.0, s))
            parts.append(f"{self._action_label(a)}:{tries}({last_r:+.2f})")
        self._overlay3.text = "valid: " + "  ".join(parts) + "  (0=untried gets priority; value=lastR)"

    def _update_overlay(self):
        """Update overlay text with mode, position, total reward and autoplay state."""
        x, y = self.env.state()
        if self.training and self._frozen_state is not None:
            x, y = self._frozen_state
        mode = {"vi": "VI", "pi": "PI", "sarsa": "SARSA"}.get(self.env.method, "VI")
        self._overlay.text = (
            f"mode={mode}  pos=({x},{y})  totalR={self.total_reward:+.1f}  autoplay={'ON' if self.autoplay else 'OFF'}"
        )
        # Build SARSA-specific info line
        if self.env.method == "sarsa":
            # Prefer state-action keys; fallback to visit_* for backward compat
            cfgp = getattr(self.env, "sarsa_params", {})
            coeff = float(cfgp.get("state_action_repeat_penalty", cfgp.get("visit_penalty", 0.0)))
            power = float(cfgp.get("state_action_repeat_penalty_power",
                                   cfgp.get("visit_penalty_power", 1.0)))
            mode = str(cfgp.get("state_action_repeat_penalty_mode", cfgp.get("visit_penalty_mode", "episode")))
            edge_c = float(cfgp.get("edge_repeat_cost", 0.0))
            edge_p = float(cfgp.get("edge_repeat_cost_power", 1.0))
            edge_m = str(cfgp.get("edge_repeat_cost_mode", "episode"))
            sel_c = float(cfgp.get("selection_state_novel_coeff", 0.0))
            sel_p = float(cfgp.get("selection_state_novel_power", 1.0))
            sel_m = str(cfgp.get("selection_state_novel_mode", "episode"))
            ssa_c = float(cfgp.get("selection_sa_novel_coeff", 0.0))
            ssa_p = float(cfgp.get("selection_sa_novel_power", 1.0))
            ssa_m = str(cfgp.get("selection_sa_novel_mode", "episode"))
            self._overlay2.text = (
                f"SARSA: sa_penalty={coeff:.2f}, power={power:.2f}, mode={mode}, anti_backtrack={float(cfgp.get('anti_backtrack_cost',0.0)):.2f}, edge={edge_c:.2f}/{edge_p:.1f}/{edge_m}, selS={sel_c:.2f}/{sel_p:.1f}/{sel_m}, selSA={ssa_c:.2f}/{ssa_p:.1f}/{ssa_m}, last_penalty={float(getattr(self.env, 'last_penalty', 0.0)):.2f}"
            )
        else:
            self._overlay2.text = ""

    def on_draw(self):
        """Arcade draw handler. Draws tiles, heatmap, actor, and overlays."""
        self.clear()
        self.floor_sprites.draw()
        self.wall_sprites.draw()
        self.goal_sprites.draw()
        # Heatmap overlay before actor
        self._draw_heatmap()
        # Fog-of-war for SARSA (draw after tiles/heatmap, before actor)
        self._draw_fog_of_war()
        self.actor_sprites.draw()
        self._update_overlay()
        self._overlay.draw()
        # Draw secondary overlay line (SARSA details)
        if self._overlay2.text:
            self._overlay2.draw()
        # Draw per-state debug line
        self._update_action_debug_overlay()
        if self._overlay3.text:
            self._overlay3.draw()
        # Draw training status
        if self.training:
            self._training_text.text = "Training SARSA..."
            self._training_text.draw()
        self._hint.draw()

    def on_update(self, dt: float):
        """Arcade update handler. Advances time and performs autoplay steps.

        Args:
            dt: Delta time in seconds since the last frame.
        """
        if self.done or not self.autoplay:
            return
        if self.training:
            return
        self._accum += dt
        if self._accum >= self.viz.step_interval:
            self._accum = 0.0
            # Step under current policy via environment
            s = self.env.state()
            s2, r, done, _info, a = self.env.step_under_policy()  # type: ignore[attr-defined]
            key = (s, a)
            tries, _lr, _ls2 = self.sa_debug.get(key, (0, 0.0, s))
            self.sa_debug[key] = (tries + 1, float(r), s2)
            self.total_reward += r
            self.done = done
            self._move_sprite_to_state()
            if self.done and self.env.state() == self.env.cfg.goal:
                try:
                    self.env.on_goal_reached()  # type: ignore[attr-defined]
                except Exception:
                    if self.env.cfg.layout == "maze":
                        self.env.regenerate_walls()
                    self.env.recompute_policies_and_maybe_train()  # type: ignore[attr-defined]
                    self.env.reset()
                # UI resets
                self._rebuild_tiles()
                self._clear_heatmap()
                self.total_reward = 0.0
                self.done = False
                self._accum = 0.0
                self.autoplay = True
                self._move_sprite_to_state()

    def _move_sprite_to_state(self):
        """Sync actor sprite position with the current environment state and mark visit."""
        cs = self.viz.cell_size
        x, y = self.env.state()
        if self.training and self._frozen_state is not None:
            x, y = self._frozen_state
        self.actor.center_x = x * cs + cs / 2
        self.actor.center_y = y * cs + cs / 2
        # mark visit for the new state
        if not self.training:
            self._mark_visit()

    # --- Heatmap helpers ---
    def _init_heatmap(self):
        """Initialize heatmap visit counters and normalization state."""
        self.visits = [[0 for _ in range(self.env.cols)] for __ in range(self.env.rows)]
        self.max_visit = 1

    def _clear_heatmap(self):
        """Clear heatmap data (used on resets, layout changes, and new mazes)."""
        self._init_heatmap()

    def _mark_visit(self):
        """Increment visit counter for the current state and track the max visit."""
        x, y = self.env.state()
        if getattr(self, "training", False) and getattr(self, "_frozen_state", None) is not None:
            x, y = self._frozen_state
        if 0 <= x < self.env.cols and 0 <= y < self.env.rows:
            self.visits[y][x] += 1
            if self.visits[y][x] > self.max_visit:
                self.max_visit = self.visits[y][x]

    def _draw_heatmap(self):
        """Draw a translucent red heatmap per-visit overlay over traversable cells."""
        cs = self.viz.cell_size
        walls = getattr(self.env, "walls", set())
        # Draw translucent red squares with alpha proportional to visit frequency
        for y in range(1, self.env.rows - 1):
            for x in range(1, self.env.cols - 1):
                if (x, y) in walls:
                    continue
                v = self.visits[y][x]
                if v <= 0:
                    continue
                # Normalize and clamp alpha (softer heatmap)
                alpha = int(min(120, 12 + 90 * (v / max(1, self.max_visit))))
                left = x * cs
                right = left + cs
                bottom = y * cs
                top = bottom + cs
                arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, (255, 0, 0, alpha))

    def _draw_fog_of_war(self):
        """Draw a semi-transparent shadow on unexplored cells for SARSA.

        Rules:
        - Only active when method == 'sarsa'.
        - Reveal cells within 1-block of the actor (Chebyshev distance <= 1).
        - Reveal already visited cells (visits[y][x] > 0).
        - Shadow everything else with a soft dark overlay.
        """
        if self.env.method != "sarsa":
            return
        cs = self.viz.cell_size
        # Use frozen state while training so the halo doesn't move during training frames
        ax, ay = self._frozen_state if (self.training and self._frozen_state is not None) else self.env.state()
        walls = getattr(self.env, "walls", set())
        # Slightly see-through dark mask
        color = (0, 0, 0, 150)
        for y in range(0, self.env.rows):
            for x in range(0, self.env.cols):
                is_border = (x == 0 or y == 0 or x == self.env.cols - 1 or y == self.env.rows - 1)
                if not is_border:
                    # Reveal within 1-block radius (Chebyshev distance) or if visited
                    if max(abs(x - ax), abs(y - ay)) <= 1:
                        continue
                    if self.visits[y][x] > 0:
                        continue
                # Draw shadow (including walls and borders) to emphasize unexplored area
                left = x * cs
                right = left + cs
                bottom = y * cs
                top = bottom + cs
                arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, color)

    def _rebuild_tiles(self):
        """Recreate floor/wall/goal tiles from the current env geometry."""
        cs = self.viz.cell_size
        s_floor = cs / max(1, self.tex_floor.width)
        s_wall = cs / max(1, self.tex_wall.width)
        s_goal = cs / max(1, self.tex_goal.width)

        self.floor_sprites = arcade.SpriteList()
        self.wall_sprites = arcade.SpriteList()
        self.goal_sprites = arcade.SpriteList()

        for y in range(self.env.rows):
            for x in range(self.env.cols):
                cx = x * cs + cs / 2
                cy = y * cs + cs / 2
                is_border = x in (0, self.env.cols - 1) or y in (0, self.env.rows - 1)
                is_interior_wall = (x, y) in getattr(self.env, "walls", set())
                if is_border or is_interior_wall:
                    spr = arcade.Sprite()
                    spr.texture = self.tex_wall
                    spr.center_x = cx
                    spr.center_y = cy
                    spr.scale = s_wall
                    self.wall_sprites.append(spr)
                else:
                    spr = arcade.Sprite()
                    spr.texture = self.tex_floor
                    spr.center_x = cx
                    spr.center_y = cy
                    spr.scale = s_floor
                    self.floor_sprites.append(spr)

        gx, gy = self.env.cfg.goal
        goal = arcade.Sprite()
        goal.texture = self.tex_goal
        goal.center_x = gx * cs + cs / 2
        goal.center_y = gy * cs + cs / 2
        goal.scale = s_goal
        self.goal_sprites.append(goal)