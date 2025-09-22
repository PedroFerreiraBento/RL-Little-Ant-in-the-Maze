from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import arcade
from pyglet.window import key as pygkey

from src.envs.gridworld_v1 import GridworldV1, UP, DOWN, LEFT, RIGHT
from src.agents.planning.gridworld_planning import GWConfig, value_iteration, policy_iteration

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "images"


@dataclass
class VizCfg:
    cell_size: int = 32
    step_interval: float = 0.12  # seconds between auto-steps


class GridworldV1Window(arcade.Window):
    def __init__(
        self,
        env: GridworldV1,
        viz: VizCfg,
        policy: Optional[Dict[Tuple[int, int], int]] = None,
        pi_vi: Optional[Dict[Tuple[int, int], int]] = None,
        pi_pi: Optional[Dict[Tuple[int, int], int]] = None,
        method: str = "vi",
    ):
        width = env.cols * viz.cell_size
        height = env.rows * viz.cell_size
        super().__init__(width=width, height=height, title="Gridworld V1 (Planning)", resizable=False)

        arcade.set_background_color(arcade.color.BLACK)

        self.env = env
        # Store policies (single or both). If only one is provided, use it for both.
        self.pi_vi: Dict[Tuple[int, int], int] = pi_vi or policy or {}
        self.pi_pi: Dict[Tuple[int, int], int] = pi_pi or policy or {}
        self.method = method if method in ("vi", "pi") else "vi"
        self.viz = viz

        # Textures
        self.tex_floor = arcade.load_texture(str(ASSETS / "floor.png"))
        self.tex_wall = arcade.load_texture(str(ASSETS / "wall.png"))
        self.tex_goal = arcade.load_texture(str(ASSETS / "exit.png"))
        self.tex_actor = arcade.load_texture(str(ASSETS / "actor.png"))

        cs = viz.cell_size
        def scale_for(tex: arcade.Texture) -> float:
            return cs / max(1, tex.width)

        self.floor_sprites = arcade.SpriteList()
        self.wall_sprites = arcade.SpriteList()
        self.goal_sprites = arcade.SpriteList()
        self.actor_sprites = arcade.SpriteList()

        s_floor = scale_for(self.tex_floor)
        s_wall = scale_for(self.tex_wall)
        s_goal = scale_for(self.tex_goal)
        s_actor = scale_for(self.tex_actor)

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
        self._hint = arcade.Text("Space: autoplay  |  R: reset  |  M: switch VI/PI", 10, 6, arcade.color.LIGHT_GRAY, 12)
        self._hint.text = "Space: autoplay  |  R: reset  |  M: switch VI/PI  |  L: toggle layout  |  N: new maze"

    def _update_overlay(self):
        x, y = self.env.state()
        mode = "VI" if self.method == "vi" else "PI"
        self._overlay.text = (
            f"mode={mode}  pos=({x},{y})  totalR={self.total_reward:+.1f}  autoplay={'ON' if self.autoplay else 'OFF'}"
        )

    def on_draw(self):
        self.clear()
        self.floor_sprites.draw()
        self.wall_sprites.draw()
        self.goal_sprites.draw()
        self.actor_sprites.draw()
        self._update_overlay()
        self._overlay.draw()
        self._hint.draw()

    def on_update(self, dt: float):
        if self.done or not self.autoplay:
            return
        self._accum += dt
        if self._accum >= self.viz.step_interval:
            self._accum = 0.0
            self._step_policy()

    def _move_sprite_to_state(self):
        cs = self.viz.cell_size
        x, y = self.env.state()
        self.actor.center_x = x * cs + cs / 2
        self.actor.center_y = y * cs + cs / 2

    # --- Deterministic next-state helper mirroring env rules (no mutation) ---
    def _next_from(self, s: Tuple[int, int], a: int) -> Tuple[int, int]:
        x, y = s
        nx, ny = x, y
        if a == UP:
            ny = min(self.env.rows - 2, y + 1)
        elif a == DOWN:
            ny = max(1, y - 1)
        elif a == LEFT:
            nx = max(1, x - 1)
        elif a == RIGHT:
            nx = min(self.env.cols - 2, x + 1)
        if (nx, ny) in getattr(self.env, "walls", set()):
            return (x, y)
        return (nx, ny)

    def _greedy_to_goal(self, s: Tuple[int, int]) -> int:
        gx, gy = self.env.cfg.goal
        x, y = s
        # Prefer axis with larger distance first, break ties by x then y
        candidates = []
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
            if self._next_from(s, a) != s:
                return a
        return UP

    def _current_policy(self) -> Dict[Tuple[int, int], int]:
        return self.pi_vi if self.method == "vi" else self.pi_pi

    def _step_policy(self):
        if self.done:
            return
        s = self.env.state()
        pi = self._current_policy() or {}
        # Choose action: policy -> greedy-to-goal fallback
        a = pi.get(s, None)
        if a is None:
            a = self._greedy_to_goal(s)
        # If policy action is blocked (no state change), pick first valid alternative
        if self._next_from(s, a) == s:
            for alt in (UP, DOWN, LEFT, RIGHT):
                if self._next_from(s, alt) != s:
                    a = alt
                    break
        s2, r, done, _info = self.env.step(a)
        self.total_reward += r
        self.done = done
        self._move_sprite_to_state()
        # Auto-restart on success (reaching goal)
        if self.done and self.env.state() == self.env.cfg.goal:
            # If current layout is a maze, generate a new one
            if self.env.cfg.layout == "maze":
                self.env.regenerate_walls()
            self.env.reset()
            self._rebuild_tiles()
            self._recompute_policies()
            self.total_reward = 0.0
            self.done = False
            self._accum = 0.0
            self.autoplay = True
            self._move_sprite_to_state()

    def _rebuild_tiles(self):
        # Clear existing tiles and build according to env.walls
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

    def _recompute_policies(self):
        gwcfg = GWConfig(
            cols=self.env.cols,
            rows=self.env.rows,
            start=self.env.cfg.start,
            goal=self.env.cfg.goal,
            step_reward=self.env.rew.step,
            goal_reward=self.env.rew.goal,
            collision_reward=self.env.rew.collision,
            gamma=self.env.cfg.gamma,
            walls=set(getattr(self.env, "walls", set())),
        )
        _, self.pi_vi = value_iteration(gwcfg)
        _, self.pi_pi = policy_iteration(gwcfg)

    def _step_manual(self, a: int):
        if self.done:
            return
        s2, r, done, _info = self.env.step(a)
        self.total_reward += r
        self.done = done
        self._move_sprite_to_state()
        # Auto-restart on success (reaching goal) even in manual mode
        if self.done and self.env.state() == self.env.cfg.goal:
            if self.env.cfg.layout == "maze":
                self.env.regenerate_walls()
            self.env.reset()
            self._rebuild_tiles()
            self._recompute_policies()
            self.total_reward = 0.0
            self.done = False
            self._accum = 0.0
            self.autoplay = True
            self._move_sprite_to_state()

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == pygkey.SPACE:
            self.autoplay = not self.autoplay
            return
        if symbol in (pygkey.R,):
            self.env.reset()
            self.total_reward = 0.0
            self.done = False
            self._accum = 0.0
            # keep autoplay on after resets
            self.autoplay = True
            self._move_sprite_to_state()
            return
        if symbol in (pygkey.M,):
            # Switch planning method
            self.method = "pi" if self.method == "vi" else "vi"
            return
        if symbol in (pygkey.L,):
            # Toggle layout and rebuild
            self.env.cfg.layout = "maze" if (self.env.cfg.layout != "maze") else "corridor"
            self.env.regenerate_walls()
            self.env.reset()
            self._rebuild_tiles()
            self._recompute_policies()
            self.total_reward = 0.0
            self.done = False
            self._accum = 0.0
            self.autoplay = True
            self._move_sprite_to_state()
            return
        if symbol in (pygkey.N,):
            # New maze (only meaningful when layout == 'maze')
            if self.env.cfg.layout == "maze":
                self.env.regenerate_walls()
                self.env.reset()
                self._rebuild_tiles()
                self._recompute_policies()
                self.total_reward = 0.0
                self.done = False
                self._accum = 0.0
                self.autoplay = True
                self._move_sprite_to_state()
            return
        # Manual steps
        if symbol == pygkey.UP or symbol == pygkey.W:
            self._step_manual(UP)
        elif symbol == pygkey.DOWN or symbol == pygkey.S:
            self._step_manual(DOWN)
        elif symbol == pygkey.LEFT or symbol == pygkey.A:
            self._step_manual(LEFT)
        elif symbol == pygkey.RIGHT or symbol == pygkey.D:
            self._step_manual(RIGHT)
