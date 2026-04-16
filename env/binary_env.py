import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .problem import BinaryProblem
from .representation import NarrowRepresentation, WideRepresentation


class BinaryPCGRLEnv(gym.Env):
    def __init__(self, grid_size=10, representation="narrow", max_steps=200, problem_config=None):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        if problem_config is None:
            problem_config = {}

        self.problem = BinaryProblem(**problem_config)

        if representation == "narrow":
            self.rep = NarrowRepresentation(grid_size)
        elif representation == "wide":
            self.rep = WideRepresentation(grid_size)
        else:
            raise ValueError("Invalid representation")

        self.action_space = spaces.Discrete(self.rep.action_space_size())

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size, grid_size),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.random.randint(0, 2, (self.grid_size, self.grid_size))

    # ensure start and goal are always empty
        self.grid[0, 0] = 0
        self.grid[self.grid_size - 1, self.grid_size - 1] = 0

        self.steps = 0

        if hasattr(self.rep, "reset"):
            self.rep.reset()

        return self.grid, {}

    def step(self, action):
        old_grid = self.grid.copy()

        self.grid = self.rep.apply_action(self.grid, action)

    # enforce valid endpoints (prevents degenerate solutions)
        self.grid[0, 0] = 0
        self.grid[self.grid_size - 1, self.grid_size - 1] = 0

        reward = self.problem.compute_reward(old_grid, self.grid)

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.grid, reward, done, False, {}