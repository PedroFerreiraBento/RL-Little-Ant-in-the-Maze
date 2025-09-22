from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import random

import arcade
from pyglet.window import key as pygkey

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "images"

# Tile indices
WALL = 1
FLOOR = 0
EXIT = 2


@dataclass
class GridConfig:
    cols: int = 31
    rows: int = 21
    cell_size: int = 32


class MazeGenerator:
    """Depth-first backtracker maze generator on odd-sized grid.

    Grid uses odd dimensions so that walls occupy even indices and cells odd indices.
    """

    def __init__(self, cols: int, rows: int, rng_seed: Optional[int] = None):
        # enforce odd sizes for traditional maze carving
        self.cols = cols if cols % 2 == 1 else cols - 1
        self.rows = rows if rows % 2 == 1 else rows - 1
        self.rng = random.Random(rng_seed)

    def generate(self, start: Tuple[int, int] = (1, 1)) -> List[List[int]]:
        c, r = self.cols, self.rows
        grid = [[WALL for _ in range(c)] for _ in range(r)]

        # Helper to carve a cell to floor
        def carve(x: int, y: int):
            grid[y][x] = FLOOR

        # Directions: (dx, dy), two-step to jump to next cell; between is the wall to remove
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        sx, sy = start
        sx = max(1, min(self.cols - 2, sx))
        sy = max(1, min(self.rows - 2, sy))
        if sx % 2 == 0:
            sx -= 1
        if sy % 2 == 0:
            sy -= 1

        stack: List[Tuple[int, int]] = [(sx, sy)]
        carve(sx, sy)

        while stack:
            x, y = stack[-1]
            self.rng.shuffle(dirs)
            carved = False
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                wx, wy = x + dx // 2, y + dy // 2
                if 1 <= nx < c - 1 and 1 <= ny < r - 1 and grid[ny][nx] == WALL:
                    # carve wall and next cell
                    grid[wy][wx] = FLOOR
                    carve(nx, ny)
                    stack.append((nx, ny))
                    carved = True
                    break
            if not carved:
                stack.pop()
        return grid


class MazeWindow(arcade.Window):
    def __init__(self, cfg: GridConfig, start: Tuple[int, int], exit_pos: Optional[Tuple[int, int]], rng_seed: Optional[int],
                 random_exit: bool = True, min_actor_exit_distance: int = 8):
        super().__init__(width=cfg.cols * cfg.cell_size, height=cfg.rows * cfg.cell_size,
                         title="Little Ant in the Maze", resizable=False)
        arcade.set_background_color(arcade.color.BLACK)

        self.cfg = cfg
        self.start = start
        self.exit_pos: Tuple[int, int]
        self.rng = random.Random(rng_seed)

        # Generate maze
        gen = MazeGenerator(cfg.cols, cfg.rows, rng_seed=rng_seed)
        self.grid = gen.generate(start=start)

        # Build list of floor cells
        floor_cells = [(x, y) for y in range(cfg.rows) for x in range(cfg.cols) if self.grid[y][x] == FLOOR]

        # Helper to pick a random border floor cell
        def random_border_floor() -> Tuple[int, int]:
            border = [(x, y) for (x, y) in floor_cells if x in (1, cfg.cols - 2) or y in (1, cfg.rows - 2)]
            pool = border if border else floor_cells
            return self.rng.choice(pool)

        # Decide exit position
        if random_exit or not exit_pos:
            ex, ey = random_border_floor()
        else:
            ex, ey = exit_pos
            ex = max(1, min(cfg.cols - 2, ex))
            ey = max(1, min(cfg.rows - 2, ey))
            if self.grid[ey][ex] == WALL:
                self.grid[ey][ex] = FLOOR

        # Set exit tile
        self.grid[ey][ex] = EXIT
        self.exit_pos = (ex, ey)

        # Load textures
        self.tex_floor = arcade.load_texture(str(ASSETS / "floor.png"))
        self.tex_wall = arcade.load_texture(str(ASSETS / "wall.png"))
        self.tex_exit = arcade.load_texture(str(ASSETS / "exit.png"))
        # actor reserved for later
        # self.tex_actor = arcade.load_texture(str(ASSETS / "actor.png"))

        # Build sprite lists for tiles (robust across Arcade versions)
        cs = self.cfg.cell_size
        # Compute per-texture scale to match cell size
        def scale_for(tex: arcade.Texture) -> float:
            # Avoid division by zero; default to 1.0
            return cs / max(1, tex.width)

        self.floor_sprites = arcade.SpriteList()
        self.wall_sprites = arcade.SpriteList()
        self.exit_sprites = arcade.SpriteList()
        self.actor_sprites = arcade.SpriteList()

        s_floor = scale_for(self.tex_floor)
        s_wall = scale_for(self.tex_wall)
        s_exit = scale_for(self.tex_exit)

        for y in range(self.cfg.rows):
            for x in range(self.cfg.cols):
                tile = self.grid[y][x]
                center_x = x * cs + cs / 2
                center_y = y * cs + cs / 2
                if tile == WALL:
                    spr = arcade.Sprite()
                    spr.texture = self.tex_wall
                    spr.center_x = center_x
                    spr.center_y = center_y
                    spr.scale = s_wall
                    self.wall_sprites.append(spr)
                elif tile == FLOOR:
                    spr = arcade.Sprite()
                    spr.texture = self.tex_floor
                    spr.center_x = center_x
                    spr.center_y = center_y
                    spr.scale = s_floor
                    self.floor_sprites.append(spr)
                elif tile == EXIT:
                    spr = arcade.Sprite()
                    spr.texture = self.tex_exit
                    spr.center_x = center_x
                    spr.center_y = center_y
                    spr.scale = s_exit
                    self.exit_sprites.append(spr)

        # Place actor at a random floor cell with minimum Manhattan distance from exit
        ex, ey = self.exit_pos
        candidates = [(x, y) for (x, y) in floor_cells if abs(x - ex) + abs(y - ey) >= max(0, min_actor_exit_distance)]
        # If constraint too strict, relax progressively
        d = min_actor_exit_distance
        while not candidates and d > 0:
            d -= 1
            candidates = [(x, y) for (x, y) in floor_cells if abs(x - ex) + abs(y - ey) >= d]
        if not candidates:
            candidates = floor_cells
        ax, ay = self.rng.choice(candidates)
        # Track actor grid position
        self.actor_xy: Tuple[int, int] = (ax, ay)
        actor = arcade.Sprite()
        actor.texture = arcade.load_texture(str(ASSETS / "actor.png"))
        actor.center_x = ax * cs + cs / 2
        actor.center_y = ay * cs + cs / 2
        # Match cell size
        atex = actor.texture
        actor.scale = cs / max(1, atex.width)
        self.actor_sprites.append(actor)

    def on_draw(self):
        self.clear()
        # Draw order: floor -> walls -> exit on top
        if self.floor_sprites:
            self.floor_sprites.draw()
        if self.wall_sprites:
            self.wall_sprites.draw()
        if self.exit_sprites:
            self.exit_sprites.draw()
        if self.actor_sprites:
            self.actor_sprites.draw()

    # --- Movement mechanics ---
    def _can_walk(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.cfg.cols and 0 <= gy < self.cfg.rows and self.grid[gy][gx] != WALL

    def _move_actor(self, dx: int, dy: int) -> None:
        if not self.actor_sprites:
            return
        x, y = self.actor_xy
        nx, ny = x + dx, y + dy
        if self._can_walk(nx, ny):
            self.actor_xy = (nx, ny)
            cs = self.cfg.cell_size
            actor = self.actor_sprites[0]
            actor.center_x = nx * cs + cs / 2
            actor.center_y = ny * cs + cs / 2

    def on_key_press(self, symbol: int, modifiers: int):
        # Arrows
        if symbol == pygkey.UP:
            self._move_actor(0, 1)
        elif symbol == pygkey.DOWN:
            self._move_actor(0, -1)
        elif symbol == pygkey.LEFT:
            self._move_actor(-1, 0)
        elif symbol == pygkey.RIGHT:
            self._move_actor(1, 0)
        # WASD
        elif symbol == pygkey.W:
            self._move_actor(0, 1)
        elif symbol == pygkey.S:
            self._move_actor(0, -1)
        elif symbol == pygkey.A:
            self._move_actor(-1, 0)
        elif symbol == pygkey.D:
            self._move_actor(1, 0)


def run_from_config(config_path: Path):
    from ..utils.config import load_json

    data = load_json(config_path)
    grid = data.get("grid", {})
    cfg = GridConfig(
        cols=int(grid.get("cols", 31)),
        rows=int(grid.get("rows", 21)),
        cell_size=int(grid.get("cell_size", 32)),
    )
    rng_seed = data.get("rng_seed")
    start = tuple(data.get("start", [1, 1]))  # type: ignore
    exit_raw = data.get("exit")
    exit_pos = tuple(exit_raw) if isinstance(exit_raw, list) and len(exit_raw) == 2 else None  # type: ignore
    random_exit = bool(data.get("random_exit", True))
    min_actor_exit_distance = int(data.get("min_actor_exit_distance", 8))

    window = MazeWindow(
        cfg,
        start=start,
        exit_pos=exit_pos,
        rng_seed=rng_seed,
        random_exit=random_exit,
        min_actor_exit_distance=min_actor_exit_distance,
    )
    arcade.run()
