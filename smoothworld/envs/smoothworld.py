from __future__ import annotations

from typing import Literal, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import dataclasses


def _cells_to_array(cells, dtype=float):
    if not cells:
        return np.empty((0, 2), dtype=dtype)
    arr = np.array(list(cells), dtype=dtype)
    return arr.reshape(-1, 2)


@dataclasses.dataclass
class GridWorldEnvConfig:
    grid_size: int = 25
    death_threshold: float = -10.0
    goal_threshold: float = 20.0
    time_step: float = 0.25
    max_speed: float = 1.0

    @property
    def death_cells(self) -> set[tuple[int, int]]:
        return set()

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        return set()

    @property
    def wall_cells(self) -> set[tuple[int, int]]:
        """
        These cells there's no penalty, BUT the agent can't move into them.
        """
        return set()

    @property
    def initial_cells(self) -> set[tuple[int, int]]:
        """
        The cells that the agent can start in.
        """
        return {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}

    @property
    def cx(self) -> float:
        return self.grid_size / 2.0

    @property
    def cy(self) -> float:
        return self.grid_size / 2.0

    @property
    def cx_cell(self) -> int:
        return self.grid_size // 2

    @property
    def cy_cell(self) -> int:
        return self.grid_size // 2

    @property
    def center(self) -> np.ndarray:
        return np.array([self.grid_size / 2.0, self.grid_size / 2.0], float)

    @property
    def reward_map(self) -> np.ndarray:
        size = self.grid_size
        M = np.zeros((size, size), float)
        for x in range(size):
            for y in range(size):
                if (x, y) in self.death_cells:
                    M[y, x] = self.death_threshold
                elif (x, y) in self.goal_cells:
                    M[y, x] = self.goal_threshold
        return M


@dataclasses.dataclass
class ThreeGoalsConfig(GridWorldEnvConfig):
    custom_triangle_radius: int | None = None
    custom_goal_radius: int = 3

    @property
    def triangle_radius(self) -> float:
        if self.custom_triangle_radius is not None:
            return self.custom_triangle_radius
        return self.grid_size / 3

    @property
    def radius(self) -> int:
        if self.custom_goal_radius is not None:
            r = self.custom_goal_radius
        else:
            r = (self.grid_size // 2) - 2
        assert r % 2 == 1, "Radius must be odd"
        return r

    @property
    def death_cells(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if x in [self.cx_cell - 1, self.cx_cell, self.cx_cell + 1]
            and y
            in [
                self.cy_cell - 1,
                self.cy_cell,
                self.cy_cell + 1,
            ]
        }

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        centerpoints = [
            (12, 18),
            (7, 7),
            (17, 7),
        ]
        blocks = []
        for cx, cy in centerpoints:
            blocks.extend(
                (cx + x - self.radius // 2, cy + y - self.radius // 2)
                for x in range(self.radius)
                for y in range(self.radius)
            )
        return set(blocks)


class TwoWallsConfig(GridWorldEnvConfig):
    @property
    def death_cells(self) -> set[tuple[int, int]]:
        return set()

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        # Two goal rows: top/bottom borders (no need to build long repeated lists)
        goal_rows = (self.grid_size - 1, self.grid_size - 2, 0, 1)
        return {(x, y) for x in range(self.grid_size) for y in goal_rows}


class FourWallsConfig(GridWorldEnvConfig):
    @property
    def death_cells(self) -> set[tuple[int, int]]:
        return set()

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        """
        Four 5x5 goal blocks placed at the midpoints of each edge (top/bottom/left/right).
        """
        size = self.grid_size
        block = 3
        half = block // 2
        centers = [
            (self.cx_cell, size - 1 - half),  # top
            (self.cx_cell, half),             # bottom
            (half, self.cy_cell),             # left
            (size - 1 - half, self.cy_cell),  # right
        ]
        cells: set[tuple[int, int]] = set()
        for cx, cy in centers:
            for dx in range(-half, half + 1):
                for dy in range(-half, half + 1):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < size and 0 <= y < size:
                        cells.add((x, y))
        return cells


class TreeInTheMiddleConfig(GridWorldEnvConfig):
    @property
    def death_cells(self) -> set[tuple[int, int]]:
        cells = {
            (x, y)
            # for x in range(self.cx_cell + 2, self.cx_cell + 4)
            # for y in range(self.cy_cell - 5, self.cy_cell + 6)
            for x in range(self.cx_cell + 1, self.cx_cell + 6)
            for y in range(self.cy_cell - 2, self.cy_cell + 3)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size
        }
        return cells

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for x in [self.grid_size - 1, self.grid_size - 2] * self.grid_size
            for y in range(self.cx_cell - 2, self.cx_cell + 3)
        }


class TwoSlitsConfig(GridWorldEnvConfig):
    """
    A grid world with a long wall-like obstacle (depth cells), with two openings.
    The goals are at the right edge of the environment.
    """

    @property
    def wall_cells(self) -> set[tuple[int, int]]:
        death_cell_col = self.cx_cell + 3
        opening_rows = [
            *list(range(self.cy_cell - 8, self.cy_cell - 4)),
            *list(range(self.cy_cell + 5, self.cy_cell + 9)),
        ]
        return {
            (death_cell_col, row)
            for row in range(self.grid_size)
            if row not in opening_rows
        }

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        return {(self.grid_size - 1, y) for y in range(self.grid_size)}


class CShapeConfig(GridWorldEnvConfig):
    """
    A gridworld with a C (but flipped) shaped wall, where the reward is at the center of the C.
    """

    @property
    def wall_cells(self) -> set[tuple[int, int]]:
        wall_rows = [self.cy_cell - 5, self.cy_cell + 5]
        wall_cols = [self.cx_cell - 5, self.cx_cell + 5]
        cells = set()
        cells = (
            cells
            | {
                (x, y)
                for x in range(wall_rows[0], wall_rows[1] + 1)
                for y in [wall_cols[0], wall_cols[1]]
            }
            | {
                (x, y)
                for y in range(wall_rows[0], wall_rows[1] + 1)
                for x in [wall_cols[1]]
            }
        )
        return cells

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for x in range(self.cx_cell - 2, self.cx_cell + 3)
            for y in range(self.cy_cell - 2, self.cy_cell + 3)
        }

    # NOTE(cmk) I removed this for now, because it was having a hard time converging even with PPO.
    # @property
    # def initial_cells(self) -> set[tuple[int, int]]:
    #     return {
    #         (x, y)
    #         for x in range(self.cx_cell + 8, self.cx_cell + 15)
    #         for y in range(self.cy_cell - 2, self.cy_cell + 3)
    #     }


class GridWorldEnv(gym.Env):
    """
   A continuous 2D world with selectable reward map configurations.

    - The world is defined by the `mode` parameter, which selects a specific layout
      (e.g. 'three_goals', 'two_walls', etc.). Each mode defines the reward map, death zones,
      goal zones, and walls.
    - The agent state is its 2D position (float, not snapped to cells), normalized to [-1,1]^2 in observations.
    - The action space is continuous in [-1,1]^2, interpreted as velocity. Each step moves
      the agent by `velocity * time_step` (clipped by `max_speed`) before hitting world bounds
      or walls.
    - Episodes terminate on entering a death cell, a goal cell, or reaching the max number of steps.
    - The environment enforces walls by blocking movement into wall regions.

    Observation:
        - 2D position normalized to [-1,1].

    Action:
        - 2D continuous velocity in [-1,1], scaled by `max_speed` and `time_step`.

    Reward:
        - Defined by the reward map of the selected mode.
        - Typically sparse: high positive in goal cells, large negative in death cells.
    """

    config: GridWorldEnvConfig
    metadata = {"render_modes": ["matplotlib"], "render_fps": 4}

    def __init__(
        self,
        mode: (
            Literal[
                "three_goals",
                "two_walls",
                "four_walls",
                "tree_in_the_middle",
                "two_slits",
                "cshape",
            ]
            | None
        ) = "two_walls",
        config: Optional[GridWorldEnvConfig] = None,
        max_steps: int = 256,
    ):
        super().__init__()
        if config is None:
            if mode == "three_goals":
                config = ThreeGoalsConfig()
            elif mode == "two_walls":
                config = TwoWallsConfig()
            elif mode == "four_walls":
                config = FourWallsConfig()
            elif mode == "tree_in_the_middle":
                config = TreeInTheMiddleConfig()
            elif mode == "two_slits":
                config = TwoSlitsConfig()
            elif mode == "cshape":
                config = CShapeConfig()
            else:
                raise ValueError(f"Invalid config name: {mode}")

        self.grid_size = config.grid_size
        self.max_steps = max_steps
        self.center = config.center
        self.config = config
        self.dt = config.time_step
        self.max_speed = config.max_speed

        # build reward map
        self.reward_map = self.config.reward_map

        # spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # plotting setup
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.patch.set_facecolor("#f0f0f0")
        self.ax.set_facecolor("#f8f9fa")
        self.reset()

    def _point_in_cells(self, pos: np.ndarray, cells: set[tuple[int, int]]) -> bool:
        if not cells:
            return False
        x, y = float(pos[0]), float(pos[1])
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return False
        ix, iy = int(np.floor(x)), int(np.floor(y))
        return (ix, iy) in cells

    def is_in_wall(self, pos: np.ndarray) -> bool:
        return self._point_in_cells(pos, self.config.wall_cells)

    def is_in_goal(self, pos: np.ndarray) -> bool:
        return self._point_in_cells(pos, self.config.goal_cells)

    def is_in_death(self, pos: np.ndarray) -> bool:
        return self._point_in_cells(pos, self.config.death_cells)

    def _sample_start(self) -> np.ndarray:
        """Uniformly sample a free position (not in wall/death/goal) within allowed initial cells."""
        free_cells = [
            (x, y)
            for (x, y) in self.config.initial_cells
            if (x, y) not in self.config.wall_cells
            and (x, y) not in self.config.death_cells
            and (x, y) not in self.config.goal_cells
            and 0 <= x < self.grid_size
            and 0 <= y < self.grid_size
        ]
        if not free_cells:
            raise RuntimeError("No valid starting positions available.")
        cx, cy = free_cells[self.np_random.integers(len(free_cells))]
        offset = self.np_random.random(2)  # sample inside the cell
        return np.array([cx, cy], float) + offset

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self._sample_start()
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return ((self.pos - self.center) / (self.grid_size / 2.0)).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, float)
        velocity = np.clip(action, -1.0, 1.0) * self.max_speed
        new_pos = self.pos + velocity * self.dt
        new_pos = np.clip(new_pos, 0.0, self.grid_size - 1e-6)
        if self._point_in_cells(new_pos, self.config.wall_cells):
            new_pos = self.pos  # blocked by wall
    
        self.pos = new_pos
        grid_pos = np.clip(np.floor(self.pos).astype(int), 0, self.grid_size - 1)
        
        r = float(self.reward_map[grid_pos[1], grid_pos[0]])
        done = (r == self.config.death_threshold) or (r == self.config.goal_threshold)

        self.steps += 1
        return self._get_obs(), r, done, self.steps >= self.max_steps, {}


    def render(self, alpha: float = 0.6):
        size = self.grid_size
        fig, ax = plt.subplots(figsize=(6, 6))

        img = np.ones((size, size, 3), dtype=np.uint8) * 240
        wc = _cells_to_array(self.config.wall_cells, dtype=int)
        for x, y in wc:
            img[y, x] = [128, 128, 128]
        gc = _cells_to_array(self.config.goal_cells, dtype=int)
        for x, y in gc:
            img[y, x] = [42, 157, 143]
        ax.imshow(
            img,
            extent=(0, size, 0, size),
            origin="lower",
        )

        # highlight wall cells in gray
        if wc.size:
            ax.scatter(wc[:, 0] + 0.5, wc[:, 1] + 0.5,
                    marker='s', s=200,
                    color=np.array([128,128,128])/255.,
                    label='Wall')
        # highlight death cells in red
        dc = _cells_to_array(self.config.death_cells, dtype=float)
        if dc.size:
            ax.scatter(dc[:, 0] + 0.5, dc[:, 1] + 0.5,
                    marker='s', s=200,
                    color=np.array([229,57,70])/255.,
                    label='Death')
        # highlight goal cells in green
        gc = _cells_to_array(self.config.goal_cells, dtype=float)

        if gc.size:
            ax.scatter(gc[:, 0] + 0.5, gc[:, 1] + 0.5,
                    marker='s', s=200,
                    color=np.array([42,157,143])/255.,
                    label='Goal')

        ax.imshow(
            np.ones((size, size)).astype(np.float32) * 0,
            extent=(0, size, 0, size),
            origin="lower",
            cmap='Greys',
            vmin=0,
            vmax=1,
            alpha=alpha,
            zorder=2,
        )

        majors = np.arange(0, size + 1, 5)
        minors = np.arange(0, size + 1, 1)
        ax.set_xticks(majors)
        ax.set_yticks(majors)
        ax.set_xticks(minors, minor=True)
        ax.set_yticks(minors, minor=True)
        ax.grid(which="minor", color="#ddd", linestyle="-", linewidth=0.5)
        ax.grid(which="major", color="#bbb", linestyle="--", linewidth=1)

        return fig, ax

    def close(self):
        plt.ioff()
        plt.close(self.fig)
