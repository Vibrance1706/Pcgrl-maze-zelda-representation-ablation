import numpy as np
from collections import deque

EMPTY = 0
WALL = 1
PLAYER = 2
KEY = 3
DOOR = 4


def bfs(grid, start, goal):
    rows, cols = grid.shape
    queue = deque([(start, 0)])
    visited = set([start])

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while queue:
        (x, y), dist = queue.popleft()

        if (x, y) == goal:
            return dist

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                if (nx, ny) not in visited and grid[nx, ny] != WALL:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

    return -1


def bfs_path(grid, start, goal):
    rows, cols = grid.shape
    queue = deque([start])
    visited = set([start])
    parent = {}

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while queue:
        current = queue.popleft()

        if current == goal:
            # reconstruct path
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path

        x, y = current

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                if (nx, ny) not in visited and grid[nx, ny] != WALL:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = current
                    queue.append((nx, ny))

    return None


def find_tile(grid, tile_type):
    return np.argwhere(grid == tile_type)


class ZeldaProblem:

    def __init__(self, config=None):
        self.config = config or {}

    def compute_reward(self, old_grid, grid):

        reward = 0

        players = find_tile(grid, PLAYER)
        keys = find_tile(grid, KEY)
        doors = find_tile(grid, DOOR)

        if len(players) == 1:
            reward += 2
        else:
            reward -= 1

        if len(keys) == 1:
            reward += 2
        else:
            reward -= 1

        if len(doors) == 1:
            reward += 2
        else:
            reward -= 1

        if len(players) == 1 and len(keys) == 1 and len(doors) == 1:

            player = tuple(players[0])
            key = tuple(keys[0])
            door = tuple(doors[0])

            d1 = bfs(grid, player, key)
            d2 = bfs(grid, key, door)

            use_sequential = self.config.get("use_sequential", True)
            use_distance = self.config.get("use_distance_reward", True)

            if d1 >= 0:
                reward += 30
                if use_distance:
                    reward += 5 / (d1 + 1)

                if use_sequential:
                    if d2 >= 0:
                        reward += 30
                        if use_distance:
                            reward += 5 / (d2 + 1)
                    else:
                        reward -= 10
                        
                else:
                    if d2 >= 0:
                        reward += 30
                        if use_distance:
                            reward += 5 / (d2 + 1)
            else:
                reward -= 10

        empty_ratio = np.mean(grid == EMPTY)
        reward += -abs(empty_ratio - 0.4) * 2

        unique_tiles = len(np.unique(grid))
        reward += unique_tiles * 0.5

        if not np.array_equal(old_grid, grid):
            reward += 0.1

        return float(reward)

    def compute_success(self, grid):
        players = find_tile(grid, PLAYER)
        keys = find_tile(grid, KEY)
        doors = find_tile(grid, DOOR)

        if len(players) != 1 or len(keys) != 1 or len(doors) != 1:
            return 0

        player = tuple(players[0])
        key = tuple(keys[0])
        door = tuple(doors[0])

        d1 = bfs(grid, player, key)
        d2 = bfs(grid, key, door)

        if d1 >= 0 and d2 >= 0:
            return 1

        return 0