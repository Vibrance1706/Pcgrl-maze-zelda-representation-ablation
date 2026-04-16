import numpy as np


class NarrowRepresentation:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.current_index = 0

    def action_space_size(self):
        return 2  # only value (0 or 1)

    def reset(self):
        self.current_index = 0

    def apply_action(self, grid, action):
        x = self.current_index // self.grid_size
        y = self.current_index % self.grid_size

        new_grid = grid.copy()
        new_grid[x, y] = action

        self.current_index = (self.current_index + 1) % (self.grid_size * self.grid_size)

        return new_grid


class WideRepresentation:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def action_space_size(self):
        return self.grid_size * self.grid_size * 2

    def reset(self):
        pass

    def apply_action(self, grid, action):
        tile_idx = action // 2
        value = action % 2

        x = tile_idx // self.grid_size
        y = tile_idx % self.grid_size

        new_grid = grid.copy()
        new_grid[x, y] = value

        return new_grid