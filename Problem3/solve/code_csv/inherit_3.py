import csv
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from collections import deque
import matplotlib.pyplot as plt

# Define the number of trays, book types, storage positions, picking stations, and recycling stations
I = 20  # Number of trays
H = 400  # Number of book types
Q = 154  # Number of storage positions
M = 8  # Number of picking stations
R = 2  # Number of recycling stations

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
                return dist + 1, [(start[0], start[1])] + [(nx, ny)]
            if (nx, ny) not in visited and (nx, ny) in grid:
                queue.append((nx, ny, dist + 1))
                visited.add((nx, ny))
                
    return int(float('inf')), []  # 如果无法到达目标

# 构建二维数组Dqm和Dmr
Q, M, R = len(storage_boxes), len(picking_stations), len(recycling_stations)
Dqm = np.zeros((Q, M), dtype=int)
Dmr = np.zeros((M, R), dtype=int)

# 计算Dqm
paths_qm = {}
for i, box in enumerate(storage_boxes):
    for j, station in enumerate(picking_stations):
        distance, path = bfs_single_target(box, station, channels)
        Dqm[i, j] = distance
        paths_qm[(i, j)] = path

# 计算Dmr
paths_mr = {}
for i, station in enumerate(picking_stations):
    for j, recycle in enumerate(recycling_stations):
        distance, path = bfs_single_target(station, recycle, channels)
        Dmr[i, j] = distance
        paths_mr[(i, j)] = path

# 转置Dqm得到Dmq
Dmq = Dqm.T

# Load the CSV files
file_path_orders = 'Q2\solve\code_csv\orders_books_3.csv'
file_path_pallets = 'Q2\solve\code_csv\pallets_books_3.csv'
df_orders = pd.read_csv(file_path_orders)
df_pallets = pd.read_csv(file_path_pallets)

# Initialize the array Ch with zeros
Ch = [0] * (H + 1)  # +1 because book types start from 1 to 20
for book_type in df_orders['book']:
    Ch[book_type] += 1
Ch = Ch[1:]  # Ignore Ch[0]

# Initialize the Oih array with zeros
Oih = np.zeros((I, H), dtype=int)
for index, row in df_pallets.iterrows():
    book_type = row['book']
    tray = row['tuopan']
    Oih[tray - 1][book_type - 1] += 1


# 初始化矩阵为0
Jiq = np.zeros((I, Q), dtype=int)

# 随机选择100个不重复的储位
selected_positions = np.random.choice(Q, I, replace=False)

# 分配托盘到储位
for i in range(I):
    storage_position = selected_positions[i]
    Jiq[i, storage_position] = 1

# Define the fitness function
def evaluate(individual):
    x = individual[:I]
    y = individual[I:2*I]
    k = np.array(individual[2*I:2*I+I*H]).reshape((I, H))
    p = np.array(individual[2*I+I*H:2*I+I*H+I*M]).reshape((I, M))
    g = np.array(individual[2*I+I*H+I*M:2*I+I*H+I*M+I*Q]).reshape((I, Q))
    f = np.array(individual[2*I+I*H+I*M+I*Q:2*I+I*H+I*M+I*Q+I*R]).reshape((I, R))
    d_iqm = np.array(individual[2*I+I*H+I*M+I*Q+I*R:2*I+I*H+I*M+I*Q+I*R+I*Q*M]).reshape((I, Q, M))
    d_imq = np.array(individual[2*I+I*H+I*M+I*Q+I*R+I*Q*M:2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q]).reshape((I, M, Q))
    d_imr = np.array(individual[2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q:2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q+I*M*R]).reshape((I, M, R))
    z = np.array(individual[2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q+I*M*R:2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q+I*M*R+Q]).reshape((Q,1))

    # Define constraints and penalties
    penalty = 0
    MAX = 100000

    for i in range(I):
        for h in range(H):
            if k[i, h] > Oih[i, h]:
                penalty += MAX
            if k[i, h] < 0:
                penalty += MAX

        if np.sum(k[i, :]) + MAX * (1 - x[i]) < 1:
            penalty += MAX
        if np.sum(k[i, :]) - np.sum(Oih[i, :]) * x[i] > 0:
            penalty += MAX
        if y[i] + MAX * (np.sum(Oih[i, :]) - np.sum(k[i, :])) < 1:
            penalty += MAX
        if y[i] * np.sum(Oih[i, :]) > np.sum(k[i, :]):
            penalty += MAX

    for h in range(H):
        if np.sum(k[:, h]) != Ch[h]:
            penalty += MAX

    for i in range(I):
        if np.sum(p[i, :]) > 1:
            penalty += MAX
        if np.sum(p[i, :]) - x[i] != 0:
            penalty += MAX

        for m in range(M):
            for q in range(Q):
                if d_iqm[i, q, m] + MAX * (3 - x[i] - p[i, m] - Jiq[i][q]) < Dqm[q][m]:
                    penalty += MAX
                if d_iqm[i, q, m] - MAX * x[i] > 0:
                    penalty += MAX

    for i in range(I):
        if x[i] - y[i] < 0:
            penalty += MAX
        if np.sum(g[i, :]) > 1:
            penalty += MAX
        if y[i] + np.sum(g[i, :]) != x[i]:
            penalty += MAX
        if x[i] - np.sum(g[i, :]) < 0:
            penalty += MAX
        if np.sum(g[i, :]) + np.sum(f[i, :]) != x[i]:
            penalty += MAX
        if np.sum(d_imq[i, :, :]) + np.sum(d_imr[i, :, :]) + MAX * (1 - x[i]) < 1:
            penalty += MAX

    for i in range(I):
        for q in range(Q):
            if g[i, q] - z[q] > 0:
                penalty += MAX
            if z[q] + Jiq[i][q] - x[i] > 1:
                penalty += MAX
            for m in range(M):
                if d_imq[i, m, q] - MAX * (x[i] - y[i]) > 0:
                    penalty += MAX
                if d_imq[i, m, q] + MAX * (4 - x[i] + y[i] - p[i, m] - g[i, q] - z[q]) < Dmq[m][q]:
                    penalty += MAX

    for q in range(Q):
        if np.sum(g[:, q]) > 1:
            penalty += MAX

    for i in range(I):
        if np.sum(k[i, :]) - np.sum(Oih[i, :]) + MAX * (1 - y[i]) < 0:
            penalty += MAX
        if np.sum(f[i, :]) > 1:
            penalty += MAX
        if np.sum(f[i, :]) != y[i]:
            penalty += MAX
        if x[i] - np.sum(f[i, :]) < 0:
            penalty += MAX

    for i in range(I):
        for m in range(M):
            for r in range(R):
                if d_imr[i, m, r] + MAX * (4 - x[i] - y[i] - p[i, m] - f[i, r]) < Dmr[m][r]:
                    penalty += MAX

    # Objective function
    objective = (np.sum(d_iqm) + np.sum(d_imq) + np.sum(d_imr))

    return objective + penalty,


# Define the fitness function
def evaluate2(individual):
    x = individual[:I]
    y = individual[I:2*I]
    k = np.array(individual[2*I:2*I+I*H]).reshape((I, H))
    p = np.array(individual[2*I+I*H:2*I+I*H+I*M]).reshape((I, M))
    g = np.array(individual[2*I+I*H+I*M:2*I+I*H+I*M+I*Q]).reshape((I, Q))
    f = np.array(individual[2*I+I*H+I*M+I*Q:2*I+I*H+I*M+I*Q+I*R]).reshape((I, R))
    d_iqm = np.array(individual[2*I+I*H+I*M+I*Q+I*R:2*I+I*H+I*M+I*Q+I*R+I*Q*M]).reshape((I, Q, M))
    d_imq = np.array(individual[2*I+I*H+I*M+I*Q+I*R+I*Q*M:2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q]).reshape((I, M, Q))
    d_imr = np.array(individual[2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q:2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q+I*M*R]).reshape((I, M, R))
    z = np.array(individual[2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q+I*M*R:2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q+I*M*R+Q]).reshape((Q,1))
    # Objective function
    objective = (np.sum(d_iqm) + np.sum(d_imq) + np.sum(d_imr))

    return objective,

# Create the toolbox with the right parameters
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("attr_int", random.randint, 0, 10000)

# Number of genes in the individual
n_genes = 2*I + I*H + I*M + I*Q + I*R + I*Q*M + I*M*Q + I*M*R + Q

# Create an individual
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_genes)

# Create a population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register("evaluate", evaluate)

# Register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# Register a mutation operator
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# Register the selection operator
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm parameters
population_size = 100
n_generations = 10
cxpb = 0.4  # Crossover probability
mutpb = 0.1  # Mutation probability

# Create the population
population = toolbox.population(n=population_size)

# Apply the genetic algorithm
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, stats=None, halloffame=None, verbose=True)

# Check if there are multiple solutions in the population after the specified generations
if len(population) > 1:
    print("Multiple solutions found:")
    min_fitness = float('inf')
    best_individual = None
    for ind in population:
        fitness = evaluate2(ind)
        if fitness[0] < min_fitness:
            min_fitness = fitness[0]
            best_individual = ind
    print(f"Best solution fitness: {min_fitness}")
else:
    # Extract the best individual
    best_individual = tools.selBest(population, 1)[0]

    # Decode the best individual to get the solutions
    x_solution = best_individual[:I]
    y_solution = best_individual[I:2*I]
    k_solution = np.array(best_individual[2*I:2*I+I*H]).reshape((I, H))
    p_solution = np.array(best_individual[2*I+I*H:2*I+I*H+I*M]).reshape((I, M))
    g_solution = np.array(best_individual[2*I+I*H+I*M:2*I+I*H+I*M+I*Q]).reshape((I, Q))
    f_solution = np.array(best_individual[2*I+I*H+I*M+I*Q:2*I+I*H+I*M+I*Q+I*R]).reshape((I, R))
    d_iqm_solution = np.array(best_individual[2*I+I*H+I*M+I*Q+I*R:2*I+I*H+I*M+I*Q+I*R+I*Q*M]).reshape((I, Q, M))
    d_imq_solution = np.array(best_individual[2*I+I*H+I*M+I*Q+I*R+I*Q*M:2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q]).reshape((I, M, Q))
    d_imr_solution = np.array(best_individual[2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q:2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q+I*M*R]).reshape((I, M, R))
    z_solution = np.array(best_individual[2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q+I*M*R:2*I+I*H+I*M+I*Q+I*R+I*Q*M+I*M*Q+I*M*R+Q]).reshape((Q,1))

    # Print the solution
    print("x solution:")
    for i in range(I):
        print(f"x[{i}] = {x_solution[i]}")

    print("\ny solution:")
    for i in range(I):
        print(f"y[{i}] = {y_solution[i]}")

    print("\nk solution:")
    for i in range(I):
        for h in range(H):
            print(f"k[{i}][{h}] = {k_solution[i][h]}")

    print("\np solution:")
    for i in range(I):
        for m in range(M):
            print(f"p[{i}][{m}] = {p_solution[i][m]}")

    print("\ng solution:")
    for i in range(I):
        for q in range(Q):
            print(f"g[{i}][{q}] = {g_solution[i][q]}")

    print("\nf solution:")
    for i in range(I):
        for r in range(R):
            print(f"f[{i}][{r}] = {f_solution[i][r]}")

    print("\nd_iqm solution:")
    for i in range(I):
        for q in range(Q):
            for m in range(M):
                print(f"d_iqm[{i}][{q}][{m}] = {d_iqm_solution[i][q][m]}")

    print("\nd_imq solution:")
    for i in range(I):
        for m in range(M):
            for q in range(Q):
                print(f"d_imq[{i}][{m}][{q}] = {d_imq_solution[i][m][q]}")

    print("\nd_imr solution:")
    for i in range(I):
        for m in range(M):
            for r in range(R):
                print(f"d_imr[{i}][{m}][{r}] = {d_imr_solution[i][m][r]}")

    print("\nz solution:")
    for q in range(Q):
        print(f"z[{q}] = {z_solution[q]}")

# Visualize the paths
def plot_paths(paths, storage_boxes, picking_stations, recycling_stations):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot channels
    for (x, y) in channels:
        ax.plot(x, y, 's', color='lightgray', markersize=10)

    # Plot storage boxes
    for (x, y) in storage_boxes:
        ax.plot(x, y, 's', color='blue', markersize=10)

    # Plot picking stations
    for (x, y) in picking_stations:
        ax.plot(x, y, 's', color='green', markersize=10)

    # Plot recycling stations
    for (x, y) in recycling_stations:
        ax.plot(x, y, 's', color='red', markersize=10)

    # Plot paths
    for path in paths.values():
        for i in range(len(path)- 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Paths Visualization')
    plt.grid(True)
    plt.show()

# Combine paths for visualization
combined_paths = {**paths_qm, **paths_mr}

# Plot the paths
plot_paths(combined_paths, storage_boxes, picking_stations, recycling_stations)


