import numpy as np
from collections import deque

def is_connected(grid):
    visited = set()
    rows, cols = grid.shape

    start = None
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:
                start = (i, j)
                break
        if start:
            break

    if not start:
        return False

    queue = deque([start])
    visited.add(start)

    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    empty_tiles = np.sum(grid == 0)
    return len(visited) == empty_tiles

def shortest_path_length(grid):
    from collections import deque

    rows, cols = grid.shape

    # Find a starting empty tile
    start = None
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:
                start = (i, j)
                break
        if start:
            break

    if not start:
        return -1  # No empty tiles

    visited = set([start])
    queue = deque([(start, 0)])

    max_dist = 0
    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    while queue:
        (x, y), dist = queue.popleft()
        max_dist = max(max_dist, dist)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

    return max_dist


def shortest_path_start_goal(grid):
    from collections import deque

    rows, cols = grid.shape
    start = (0, 0)
    goal = (rows - 1, cols - 1)

    if grid[start] != 0 or grid[goal] != 0:
        return -1

    visited = set([start])
    queue = deque([(start, 0)])

    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    while queue:
        (x, y), dist = queue.popleft()

        if (x, y) == goal:
            return dist

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

    return -1