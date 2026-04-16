import numpy as np
from .utils import is_connected, shortest_path_length


# CONNECTIVITY 

def connectivity_score(grid):
    total_empty = np.sum(grid == 0)
    if total_empty == 0:
        return 0
    return 1 if is_connected(grid) else 0



# CONNECTIVITY RATIO

def connectivity_ratio(grid):
    rows, cols = grid.shape
    visited = set()

    max_component = 0
    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0 and (i, j) not in visited:
                stack = [(i, j)]
                size = 0

                while stack:
                    x, y = stack.pop()
                    if (x, y) in visited:
                        continue

                    visited.add((x, y))
                    size += 1

                    for dx, dy in directions:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if grid[nx, ny] == 0:
                                stack.append((nx, ny))

                max_component = max(max_component, size)

    total_empty = np.sum(grid == 0)
    if total_empty == 0:
        return 0

    return max_component / total_empty


# LOCAL STRUCTURE

def local_similarity(grid):
    score = 0
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            neighbors = []
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni, nj = i+dx, j+dy
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbors.append(grid[ni, nj])

            score += sum(1 for n in neighbors if n == grid[i, j])

    return score


# CORRIDOR SCORE

def corridor_score(grid):
    rows, cols = grid.shape
    score = 0

    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:
                neighbors = 0
                for dx, dy in directions:
                    ni, nj = i+dx, j+dy
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if grid[ni, nj] == 0:
                            neighbors += 1

                # corridor-like tiles (not open rooms)
                if neighbors <= 2:
                    score += 1

    return score



# REWARD FUNCTION

class BinaryProblem:

    def __init__(self,
                 use_connectivity=True,
                 use_path=True,
                 use_density=True,
                 use_corridor=True,
                 use_local=True,
                 use_exploration=True):

        self.use_connectivity = use_connectivity
        self.use_path = use_path
        self.use_density = use_density
        self.use_corridor = use_corridor
        self.use_local = use_local
        self.use_exploration = use_exploration

    def compute_reward(self, old_grid, new_grid):

        conn_old = connectivity_ratio(old_grid)
        conn_new = connectivity_ratio(new_grid)

        local_old = local_similarity(old_grid)
        local_new = local_similarity(new_grid)

        corridor_old = corridor_score(old_grid)
        corridor_new = corridor_score(new_grid)

        path_len = shortest_path_length(new_grid)

        empty_ratio = np.mean(new_grid == 0)

        reward = 0

        # 1. CONNECTIVITY

        if self.use_connectivity:
            reward += 15 * (conn_new - conn_old)

            if conn_new > 0.95:
                reward += 8
                
        # 2. PATH LENGTH

        if self.use_path:
            if path_len > 0:
                reward += 0.02 * path_len

            if path_len >= 15:
                reward += 5

        # 3. DENSITY

        if self.use_density:
            target = 0.5
            reward += -3 * abs(empty_ratio - target)

            if empty_ratio > 0.9:
                reward -= 5

            if empty_ratio < 0.1:
                reward -= 5

        # 4. CORRIDOR

        if self.use_corridor:
            reward += 0.25 * (corridor_new - corridor_old)

        # 5. LOCAL SMOOTHNESS

        if self.use_local:
            reward += 0.05 * (local_new - local_old)

        # 6. EXPLORATION

        if self.use_exploration:
            if not np.array_equal(old_grid, new_grid):
                reward += 0.01

        return float(reward)