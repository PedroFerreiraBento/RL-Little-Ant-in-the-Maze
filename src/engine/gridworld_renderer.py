from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import arcade
from pyglet.window import key as pygkey

from src.envs.gridworld_v1 import GridworldV1, UP, DOWN, LEFT, RIGHT

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "images"


@dataclass
class VizCfg:
    cell_size: int = 32
    step_interval: float = 0.12  # seconds between auto-steps


class GridworldV1Window(arcade.Window):
    def __init__(self, env: GridworldV1, policy: Dict[Tuple[int, int], int], viz: VizCfg):
        width = env.cols * viz.cell_size
        height = env.rows * viz.cell_size
        super().__init__(width=width, height=height, title="Gridworld V1 (Planning)", resizable=False)

        arcade.set_background_color(arcade.color.BLACK)

        self.env = env
        self.pi = policy
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

        # Build static tiles: border walls and interior floors/walls, goal tile
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
        self._hint = arcade.Text("Space: toggle autoplay  |  R: reset", 10, 6, arcade.color.LIGHT_GRAY, 12)

    def _update_overlay(self):
        x, y = self.env.state()
        self._overlay.text = f"pos=({x},{y})  totalR={self.total_reward:+.1f}  autoplay={'ON' if self.autoplay else 'OFF'}"

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

    def _step_policy(self):
        if self.done:
            return
        s = self.env.state()
        a = self.pi.get(s, UP)
        s2, r, done, _info = self.env.step(a)
        self.total_reward += r
        self.done = done
        self._move_sprite_to_state()

    def _step_manual(self, a: int):
        if self.done:
            return
        s2, r, done, _info = self.env.step(a)
        self.total_reward += r
        self.done = done
        self._move_sprite_to_state()

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == pygkey.SPACE:
            self.autoplay = not self.autoplay
            return
        if symbol in (pygkey.R,):
            self.env.reset()
            self.total_reward = 0.0
            self.done = False
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
