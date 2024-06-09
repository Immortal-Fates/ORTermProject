import csv
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the number of trays, book types, storage positions, picking stations, and recycling stations
I = 5  # Number of trays
H = 20  # Number of book types
Q = 10  # Number of storage positions
M = 2  # Number of picking stations
R = 2  # Number of recycling stations

# Load the CSV files
file_path_orders = 'Q2/solve/code_csv/orders_books_1.csv'
file_path_pallets = 'Q2/solve/code_csv/pallets_books_1_2.csv'
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

# 随机选择100个不重复的储位
selected_positions = np.random.choice(Q, I, replace=False)

# 分配托盘到储位
Jiq = np.zeros((I, Q), dtype=int)
for i in range(I):
    storage_position = selected_positions[i]
    Jiq[i, storage_position] = 1

Dqm = [[4, 7],
       [5, 6],
       [3, 8],
       [4, 7],
       [5, 6],
       [6, 3],
       [7, 6],
       [9, 2],
       [7, 6],
       [8, 5]]

Dmq = [[4, 5, 3, 4, 5, 6, 7, 9, 7, 8],
       [7, 6, 8, 7, 6, 3, 6, 2, 6, 5]]

Dmr = [[8, 4],
       [3, 9]]

# 计算Dqm
paths_qm = {}
paths_qm = Dqm
paths_mr = Dmr

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
    weight_1 = 10
    weight_2 = 5
    weight_3 = 1

    for i in range(I):
        for h in range(H):
            if k[i, h] > Oih[i, h]:
                penalty += weight_1 * MAX
            if k[i, h] < 0:
                penalty += weight_1 * MAX

        if np.sum(k[i, :]) + MAX * (1 - x[i]) < 1:
            penalty += weight_2 * MAX
        if np.sum(k[i, :]) - np.sum(Oih[i, :]) * x[i] > 0:
            penalty += weight_2 * MAX
        if y[i] + MAX * (np.sum(Oih[i, :]) - np.sum(k[i, :])) < 1:
            penalty += weight_2 * MAX
        if y[i] * np.sum(Oih[i, :]) > np.sum(k[i, :]):
            penalty += weight_2 * MAX

    for h in range(H):
        if np.sum(k[:, h]) != Ch[h]:
            penalty += weight_3 * MAX

    for i in range(I):
        if np.sum(p[i, :]) > 1:
            penalty += weight_3 * MAX
        if np.sum(p[i, :]) - x[i] != 0:
            penalty += weight_3 * MAX

        for m in range(M):
            for q in range(Q):
                if d_iqm[i, q, m] + MAX * (3 - x[i] - p[i, m] - Jiq[i][q]) < Dqm[q][m]:
                    penalty += weight_3 * MAX
                if d_iqm[i, q, m] - MAX * x[i] > 0:
                    penalty += weight_3 * MAX

    for i in range(I):
        if x[i] - y[i] < 0:
            penalty += weight_1 * MAX
        if np.sum(g[i, :]) > 1:
            penalty += weight_1 * MAX
        if y[i] + np.sum(g[i, :]) != x[i]:
            penalty += weight_1 * MAX
        if x[i] - np.sum(g[i, :]) < 0:
            penalty += weight_1 * MAX
        if np.sum(g[i, :]) + np.sum(f[i, :]) != x[i]:
            penalty += weight_1 * MAX
        if np.sum(d_imq[i, :, :]) + np.sum(d_imr[i, :, :]) + MAX * (1 - x[i]) < 1:
            penalty += weight_1 * MAX

    for i in range(I):
        for q in range(Q):
            if g[i, q] - z[q] > 0:
                penalty += weight_2 * MAX
            if z[q] + Jiq[i][q] - x[i] > 1:
                penalty += weight_2 * MAX
            for m in range(M):
                if d_imq[i, m, q] - MAX * (x[i] - y[i]) > 0:
                    penalty += weight_2 * MAX
                if d_imq[i, m, q] + MAX * (4 - x[i] + y[i] - p[i, m] - g[i, q] - z[q]) < Dmq[m][q]:
                    penalty += weight_2 * MAX

    for q in range(Q):
        if np.sum(g[:, q]) > 1:
            penalty += weight_3 * MAX

    for i in range(I):
        if np.sum(k[i, :]) - np.sum(Oih[i, :]) + MAX * (1 - y[i]) < 0:
            penalty += weight_3 * MAX
        if np.sum(f[i, :]) > 1:
            penalty += weight_3 * MAX
        if np.sum(f[i, :]) != y[i]:
            penalty += weight_3 * MAX
        if x[i] - np.sum(f[i, :]) < 0:
            penalty += weight_3 * MAX

    for i in range(I):
        for m in range(M):
            for r in range(R):
                if d_imr[i, m, r] + MAX * (4 - x[i] - y[i] - p[i, m] - f[i, r]) < Dmr[m][r]:
                    penalty += weight_3 * MAX

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
toolbox.register("attr_int", random.randint, 0, 100000)

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
population_size = 1000
n_generations = 20
cxpb = 0.5  # Crossover probability
mutpb = 0.1  # Mutation probability

def run_ga():
    # Create the population
    population = toolbox.population(n=population_size)

    # Apply the genetic algorithm
    for gen in range(n_generations):
        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=1, stats=None, halloffame=None, verbose=False)
    
    # Extract the best individual
    best_individual = tools.selBest(population, 1)[0]
    best_fitness = evaluate2(best_individual)[0]
    
    return best_fitness

# Run the GA 10 times and collect results
results = [run_ga() for _ in range(10)]

# Calculate mean, standard deviation, best, and worst performances
mean_result = np.mean(results)
std_dev_result = np.std(results)
best_result = np.min(results)
worst_result = np.max(results)

print(f"Mean: {mean_result}")
print(f"Standard Deviation: {std_dev_result}")
print(f"Best Result: {best_result}")
print(f"Worst Result: {worst_result}")

# Plot performance distribution
plt.figure()
plt.hist(results, bins=10, color='blue', alpha=0.7)
plt.title('Performance Distribution')
plt.xlabel('Fitness Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
