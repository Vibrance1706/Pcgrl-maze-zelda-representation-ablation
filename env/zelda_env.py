import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.zelda_problem import ZeldaProblem, bfs


class ZeldaPCGRLEnv(gym.Env):

    def __init__(self, grid_size=10, representation="narrow", max_steps=200, problem_config=None):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.tile_values = 5

        self.problem_config = problem_config or {}

        # REPRESENTATION

        if representation == "narrow":
            self.rep = NarrowRepresentation(grid_size, self.tile_values)
        elif representation == "wide":
            self.rep = WideRepresentation(grid_size, self.tile_values)
        else:
            raise ValueError("Invalid representation")

        self.action_space = spaces.Discrete(self.rep.action_space_size())

        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=(grid_size, grid_size),
            dtype=np.int32
        )

        # Pass config to problem
        self.problem = ZeldaProblem(config=self.problem_config)

    # RESET

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        px, py = 0, 0
        dx, dy = self.grid_size - 1, self.grid_size - 1

        # Random key
        while True:
            kx = np.random.randint(0, self.grid_size)
            ky = np.random.randint(0, self.grid_size)
            if (kx, ky) != (px, py) and (kx, ky) != (dx, dy):
                break

        self.grid[px, py] = 2  # PLAYER
        self.grid[kx, ky] = 3  # KEY
        self.grid[dx, dy] = 4  # DOOR

        # WALL GENERATION (FOR ABLATION PURPOSES)

        if not self.problem_config.get("enforce_connectivity", True):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.grid[i, j] == 0:
                        self.grid[i, j] = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            while True:
                temp_grid = self.grid.copy()

                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if temp_grid[i, j] == 0:
                            temp_grid[i, j] = np.random.choice([0, 1], p=[0.7, 0.3])

                player = (px, py)
                key = (kx, ky)
                door = (dx, dy)

                d1 = bfs(temp_grid, player, key)
                d2 = bfs(temp_grid, key, door)

                if d1 > 0 and d2 > 0:
                    self.grid = temp_grid
                    break

        self.steps = 0
        self.rep.reset()

        return self.grid, {}

    # STEP
    def step(self, action):
        old_grid = self.grid.copy()

        self.grid = self.rep.apply_action(self.grid, action)

        reward = self.problem.compute_reward(old_grid, self.grid)
        reward = reward / 20.0

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.grid, reward, done, False, {}


# REPRESENTATIONS

class NarrowRepresentation:
    def __init__(self, grid_size, tile_values):
        self.grid_size = grid_size
        self.tile_values = tile_values
        self.current_index = 0

    def action_space_size(self):
        return self.tile_values

    def reset(self):
        self.current_index = 0

    def apply_action(self, grid, action):
        x = self.current_index // self.grid_size
        y = self.current_index % self.grid_size

        if grid[x, y] in [2, 3, 4]:
            self.current_index = (self.current_index + 1) % (self.grid_size * self.grid_size)
            return grid

        new_grid = grid.copy()

        if action in [2, 3, 4]:
            action = np.random.choice([0, 1])

        new_grid[x, y] = action

        self.current_index = (self.current_index + 1) % (self.grid_size * self.grid_size)

        return new_grid


class WideRepresentation:
    def __init__(self, grid_size, tile_values):
        self.grid_size = grid_size
        self.tile_values = tile_values

    def action_space_size(self):
        return self.grid_size * self.grid_size * self.tile_values

    def reset(self):
        pass

    def apply_action(self, grid, action):
        tile_idx = action // self.tile_values
        value = action % self.tile_values

        x = tile_idx // self.grid_size
        y = tile_idx % self.grid_size

        if grid[x, y] in [2, 3, 4]:
            return grid

        new_grid = grid.copy()

        if value in [2, 3, 4]:
            value = np.random.choice([0, 1])

        new_grid[x, y] = value

        return new_grid