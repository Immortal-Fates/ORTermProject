import csv
import numpy as np
from collections import deque
import pulp
import pandas as pd

# 读取CSV文件
file_path = 'Q2\draw_map\map_code\map2.csv'
map_data = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        map_data.append([int(cell) for cell in row])

# 提取网格信息和位置
channels = set()
storage_boxes = []
picking_stations = []
obstacles = []
recycling_stations = []

for row in map_data:
    cell_type, x, y = row
    if cell_type == 1:
        channels.add((x, y))
    elif cell_type == 2:
        storage_boxes.append((x, y))
    elif cell_type == 4:
        obstacles.append((x, y))
    elif cell_type == 5:
        picking_stations.append((x, y))
    elif cell_type == 7:
        recycling_stations.append((x, y))

# # 打印提取的坐标信息，确保正确提取
# print("Channels:\n", channels)
# print("Storage Boxes:\n", storage_boxes)
# print("Picking Stations:\n", picking_stations)
# print("Obstacles:\n", obstacles)
# print("Recycling Stations:\n", recycling_stations)

# 定义方向向量用于BFS
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 单目标BFS函数
def bfs_single_target(start, target, grid):
    queue = deque([(start[0], start[1], 0)])
    visited = set()
    visited.add((start[0], start[1]))
    
    while queue:
        x, y, dist = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) == target:
                return dist + 1
            if (nx, ny) not in visited and (nx, ny) in grid:
                queue.append((nx, ny, dist + 1))
                visited.add((nx, ny))
                
    return int(float('inf'))  # 如果无法到达目标

# 构建二维数组Dqm和Dmr
Q, M, R = len(storage_boxes), len(picking_stations), len(recycling_stations)
Dqm = np.zeros((Q, M), dtype=int)
Dmr = np.zeros((M, R), dtype=int)

# 计算Dqm
for i, box in enumerate(storage_boxes):
    for j, station in enumerate(picking_stations):
        Dqm[i, j] = bfs_single_target(box, station, channels)

# 计算Dmr
for i, station in enumerate(picking_stations):
    for j, recycle in enumerate(recycling_stations):
        Dmr[i, j] = bfs_single_target(station, recycle, channels)

# # 打印结果
# print("Dqm:")
# print(Dqm)
# print("\nDmr:")
# print(Dmr)

# # 转置Dqm得到Dmq
Dmq = Dqm.T
# print("\nDmq:")
# print(Dmq)

print(Q,M,R)




###################################

I = 12  # Number of trays
H = 400  # Number of book types

# 初始化矩阵为0
Jiq = np.zeros((I, Q), dtype=int)

# 随机选择100个不重复的储位
selected_positions = np.random.choice(Q, I, replace=False)

# 分配托盘到储位
for i in range(I):
    storage_position = selected_positions[i]
    Jiq[i, storage_position] = 1

print(Jiq)


# Load the CSV file
file_path = 'Q2\solve\code_csv\orders_books_4.csv'  # Make sure the file path is correct
df = pd.read_csv(file_path)
# Number of book types
# Initialize the array Ch with zeros
Ch = [0] * (H + 1)  # +1 because book types start from 1 to 20
# Count the occurrences of each book type in the 'book' column
for book_type in df['book']:
    Ch[book_type] += 1
# Since Ch[0] is not used (as book types start from 1), we can ignore it
Ch = Ch[1:]

# Load the CSV file
file_path = 'Q2\solve\code_csv\pallets_books_4.csv'  # Make sure the file path is correct
df = pd.read_csv(file_path)
# Number of trays and book types

# Initialize the Oih array with zeros
Oih = np.zeros((I, H), dtype=int)

# Fill the Oih array with counts of each book type in each tray
for index, row in df.iterrows():
    book_type = row['book']
    tray = row['tuopan']
    Oih[tray - 1][book_type - 1] += 1

# Define the problem
prob = pulp.LpProblem("Library_Book_Picking", pulp.LpMinimize)

# Define decision variables
x = pulp.LpVariable.dicts("x", range(I), 0, 1, cat='Binary')
y = pulp.LpVariable.dicts("y", range(I), 0, 1, cat='Binary')
k = pulp.LpVariable.dicts("k", [(i, h) for i in range(I) for h in range(H)], 0, cat='Integer')
p = pulp.LpVariable.dicts("p", [(i, m) for i in range(I) for m in range(M)], 0, 1, cat='Binary')
g = pulp.LpVariable.dicts("g", [(i, q) for i in range(I) for q in range(Q)], 0, 1, cat='Binary')
f = pulp.LpVariable.dicts("f", [(i, r) for i in range(I) for r in range(R)], 0, 1, cat='Binary')
d_iqm = pulp.LpVariable.dicts("d_iqm", [(i, q, m) for i in range(I) for q in range(Q) for m in range(M)], 0, cat='Integer')
d_imq = pulp.LpVariable.dicts("d_imq", [(i, m, q) for i in range(I) for m in range(M) for q in range(Q)], 0, cat='Integer')
d_imr = pulp.LpVariable.dicts("d_imr", [(i, m, r) for i in range(I) for m in range(M) for r in range(R)], 0, cat='Integer')
z = pulp.LpVariable.dicts("z", range(Q), 0, 1, cat='Binary')

# Define the objective function
prob += (pulp.lpSum(d_iqm[(i, q, m)] for i in range(I) for q in range(Q) for m in range(M)) +
         pulp.lpSum(d_imq[(i, m, q)] for i in range(I) for m in range(M) for q in range(Q)) +
         pulp.lpSum(d_imr[(i, m, r)] for i in range(I) for m in range(M) for r in range(R)))

# Add constraints
MAX = 100000  # A large number

# Quantity constraints
for i in range(I):
    for h in range(H):
        prob += k[(i, h)] <= Oih[i][h]
        prob += k[(i, h)] >= 0
    prob += pulp.lpSum(k[(i, h)] for h in range(H)) + MAX * (1 - x[i]) >= 1
    prob += pulp.lpSum(k[(i, h)] for h in range(H)) - pulp.lpSum(Oih[i][h] for h in range(H)) * x[i] <= 0
    prob += y[i] + MAX * (pulp.lpSum(Oih[i][h] for h in range(H)) - pulp.lpSum(k[(i,h)] for h in range(H))) >= 1
    prob += y[i] * pulp.lpSum(Oih[i][h] for h in range(H)) <= pulp.lpSum(k[(i,h)] for h in range(H))
for h in range(H):
    prob += pulp.lpSum(k[(i, h)] for i in range(I)) == Ch[h]

# Matching constraints
for i in range(I):
    prob += pulp.lpSum(p[(i, m)] for m in range(M)) <= 1
    prob += pulp.lpSum(p[(i, m)] for m in range(M)) - x[i] == 0

for i in range(I):
    for m in range(M):
        for q in range(Q):
            prob += d_iqm[(i, q, m)] + MAX * (3 - x[i] - p[(i, m)] - Jiq[i][q]) >= Dqm[q][m]
            prob += d_iqm[(i, q, m)] - MAX * x[i] <= 0
#(3)
# Storage constraints
for i in range(I):
    prob += x[i] - y[i] >= 0
    prob += pulp.lpSum(g[(i, q)] for q in range(Q)) <= 1
    prob += y[i] + pulp.lpSum(g[(i, q)] for q in range(Q)) == x[i]
    prob += x[i] - pulp.lpSum(g[(i, q)] for q in range(Q)) >= 0
    prob += pulp.lpSum(g[(i, q)] for q in range(Q)) + pulp.lpSum(f[(i, r)] for r in range(R)) == x[i]
    prob += pulp.lpSum(d_imq[(i, m, q)] for m in range(M) for q in range(Q)) + pulp.lpSum(d_imr[(i, m, r)] for m in range(M) for r in range(R)) + MAX * (1 - x[i]) >= 1

for i in range(I):
    for q in range(Q):
        prob += g[(i, q)] - z[q] <= 0
        prob += z[q] + Jiq[i][q] - x[i] <= 1
        # prob += x[i] * Jiq[i][q] <= z[q]
        for m in range(M):
            prob += d_imq[(i, m, q)] - MAX * (x[i] - y[i]) <= 0
            prob += d_imq[(i, m, q)] + MAX * (4 - x[i] + y[i] - p[(i, m)] - g[(i, q)] - z[q]) >= Dmq[m][q]

for q in range(Q):
    prob += pulp.lpSum(g[(i, q)] for i in range(I)) <= 1
    # for i in range(I):
    #      prob += (Jiq[i][q] * x[i]) <= z[q]

#(4)
# Recycling constraints
for i in range(I):
    prob += pulp.lpSum(k[(i, h)] for h in range(H)) - pulp.lpSum(Oih[i][h] for h in range(H)) + MAX * (1 - y[i]) >= 0
    prob += pulp.lpSum(f[(i, r)] for r in range(R)) <= 1
    prob += pulp.lpSum(f[(i, r)] for r in range(R)) == y[i]
    prob += x[i] - pulp.lpSum(f[(i, r)] for r in range(R)) >= 0

for i in range(I):
    for m in range(M):
        for r in range(R):
            prob += d_imr[(i, m, r)] + MAX * (4 - x[i] - y[i] - p[(i, m)] - f[(i, r)]) >= Dmr[m][r]

# Solve the problem
prob.solve()


# Extract the solution
x_solution = [x[i].varValue for i in range(I)]
y_solution = [y[i].varValue for i in range(I)]
k_solution = [[k[(i, h)].varValue for h in range(H)] for i in range(I)]
p_solution = [[p[(i, m)].varValue for m in range(M)] for i in range(I)]
g_solution = [[g[(i, q)].varValue for q in range(Q)] for i in range(I)]
f_solution = [[f[(i, r)].varValue for r in range(R)] for i in range(I)]
d_iqm_solution = [[[d_iqm[(i, q, m)].varValue for m in range(M)] for q in range(Q)] for i in range(I)]
d_imq_solution = [[[d_imq[(i, m, q)].varValue for q in range(Q)] for m in range(M)] for i in range(I)]
d_imr_solution = [[[d_imr[(i, m, r)].varValue for r in range(R)] for m in range(M)] for i in range(I)]
z_solution = [z[q].varValue for q in range(Q)]

# # Print the solution
# print("x:", x_solution)
# print("y:", y_solution)
# print("k:", k_solution)
# print("p:", p_solution)
# print("g:", g_solution)
# print("f:", f_solution)
# print("d_iqm:", d_iqm_solution)
# print("d_imq:", d_imq_solution)
# print("d_imr:", d_imr_solution)
# print("z:", z_solution)

# Print the solution
print("x solution:")
for i in range(I):
    print(f"x[{i}] = {x_solution[i]}")

print("\ny solution:")
for i in range(I):
    print(f"y[{i}] = {y_solution[i]}")

# print("\nk solution:")
# for i in range(I):
#     for h in range(H):
#         print(f"k[{i}][{h}] = {k_solution[i][h]}")

# print("\np solution:")
# for i in range(I):
#     for m in range(M):
#         print(f"p[{i}][{m}] = {p_solution[i][m]}")

# print("\ng solution:")
# for i in range(I):
#     for q in range(Q):
#         print(f"g[{i}][{q}] = {g_solution[i][q]}")

# print("\nf solution:")
# for i in range(I):
#     for r in range(R):
#         print(f"f[{i}][{r}] = {f_solution[i][r]}")

# print("\nd_iqm solution:")
# for i in range(I):
#     for q in range(Q):
#         for m in range(M):
#             print(f"d_iqm[{i}][{q}][{m}] = {d_iqm_solution[i][q][m]}")

# print("\nd_imq solution:")
# for i in range(I):
#     for m in range(M):
#         for q in range(Q):
#             print(f"d_imq[{i}][{m}][{q}] = {d_imq_solution[i][m][q]}")

# print("\nd_imr solution:")
# for i in range(I):
#     for m in range(M):
#         for r in range(R):
#             print(f"d_imr[{i}][{m}][{r}] = {d_imr_solution[i][m][r]}")

print("\nd_iqm solution:")
for i in range(I):
    for q in range(Q):
        for m in range(M):
            if(d_iqm_solution[i][q][m] != 0):
                print(f"d_iqm[{i}][{q}][{m}] = {d_iqm_solution[i][q][m]}")

print("\nd_imq solution:")
for i in range(I):
    for m in range(M):
        for q in range(Q):
            if(d_imq_solution[i][m][q] != 0):
                print(f"d_imq[{i}][{m}][{q}] = {d_imq_solution[i][m][q]}")

print("\nd_imr solution:")
for i in range(I):
    for m in range(M):
        for r in range(R):
            if(d_imr_solution[i][m][r] != 0):
                print(f"d_imr[{i}][{m}][{r}] = {d_imr_solution[i][m][r]}")

# print("\nz solution:")
# for q in range(Q):
#     print(f"z[{q}] = {z_solution[q]}")
