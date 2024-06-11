import numpy as np
from collections import deque
def create_map(shelf_data, map_size):
    grid = np.zeros(map_size, dtype=int)
    for shelf in shelf_data.values():
        x = shelf['x']
        y = shelf['y']
        grid[x, y] = -1  # 书架位置标记为 -1
    return grid

def is_in_bounds(x, y, map_size):
    return 0 <= x < map_size[0] and 0 <= y < map_size[1]

def bfs_shortest_path(grid, start, end, map_size):
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右移动

    while queue:
        (current_x, current_y), distance = queue.popleft()

        if (current_x, current_y) == end:
            return distance

        for dx, dy in directions:
            next_x, next_y = current_x + dx, current_y + dy
            if is_in_bounds(next_x, next_y, map_size) and (next_x, next_y) not in visited and grid[next_x, next_y] != -1:
                visited.add((next_x, next_y))
                queue.append(((next_x, next_y), distance + 1))

    return float('inf')  # 如果没有路径，则返回无穷大

def calculate_shortest_paths(shelf_data, map_size, depot):
    grid = create_map(shelf_data, map_size)
    shelf_ids = list(shelf_data.keys())
    n = len(shelf_ids)
    adjacency_matrix = np.full((n + 1, n + 1), float('inf'))  # 包括原点

    for i in range(n):
        start = (shelf_data[shelf_ids[i]]['x'], shelf_data[shelf_ids[i]]['y'] - 1)
        adjacency_matrix[0, i + 1] = bfs_shortest_path(grid, depot, start, map_size)  # 原点到书架取书点
        adjacency_matrix[i + 1, 0] = adjacency_matrix[0, i + 1]  # 对称矩阵

    for i in range(n):
        for j in range(n):
            if i != j:
                start = (shelf_data[shelf_ids[i]]['x'], shelf_data[shelf_ids[i]]['y'] - 1)
                end = (shelf_data[shelf_ids[j]]['x'], shelf_data[shelf_ids[j]]['y'] - 1)
                adjacency_matrix[i + 1, j + 1] = bfs_shortest_path(grid, start, end, map_size)
            else:
                adjacency_matrix[i + 1, j + 1] = 0  # 自己到自己的距离为0

    return adjacency_matrix

def path(shelf_data, map_size, depot):
    # 创建最短路径
    adjacency_matrix = calculate_shortest_paths(shelf_data, map_size, depot)
    return adjacency_matrix

# # 示例用法
# shelf_data = {
#     1: {'x': 2, 'y': 3},
#     2: {'x': 5, 'y': 4},
#     3: {'x': 1, 'y': 7},
# }
#
# map_size = (10, 10)  # 地图大小

#
# adjacency_matrix = path(shelf_data, map_size, depot)
# print(adjacency_matrix)
