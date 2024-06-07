import random
import numpy as np
from  preprocess import *

# 数据预处理

def read_matrix_csv(filename):
  with open(filename, 'r') as file:
    reader = csv.reader(file)
    # 读取第一行，获取矩阵的大小
    matrix_size = int(next(reader)[0])
    # 读取剩余的行，获取矩阵的数据
    matrix = list(reader)
  # 将数据转换为浮点数
  matrix = [[float(cell) for cell in row] for row in matrix]
  # 将列表转换为numpy数组
  matrix = np.array(matrix)
  return matrix_size, matrix

# 读取CSV文件
matrix_size, matrix = read_matrix_csv('..\src\distance.csv')
class GeneticAlgorithm:
    def __init__(self, num_customers, num_vehicles, capacity, demands, distance_matrix, population_size, num_generations, mutation_rate):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.demands = demands
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.population = []

    def initialize_population(self):
        for _ in range(self.population_size):
            individual = np.random.permutation(self.num_customers).tolist()
            self.population.append(individual)

    def fitness(self, individual):
        total_distance = 0
        vehicle_load = 0
        current_route = [0]  # Start from the depot
        for customer in individual:
            if vehicle_load + self.demands[customer] > self.capacity:
                total_distance += self.calculate_route_distance(current_route)
                current_route = [0]
                vehicle_load = 0
            current_route.append(customer)
            vehicle_load += self.demands[customer]
        current_route.append(0)  # Return to depot
        total_distance += self.calculate_route_distance(current_route)
        return total_distance

    def calculate_route_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i]][route[i + 1]]
        return distance

    def selection(self):
        fitnesses = [self.fitness(individual) for individual in self.population]
        probabilities = [1 / fitness for fitness in fitnesses]
        total_fitness = sum(probabilities)
        probabilities = [prob / total_fitness for prob in probabilities]
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)
        selected_population = [self.population[i] for i in selected_indices]
        return selected_population

    def crossover(self, parent1, parent2):
        size = min(len(parent1), len(parent2))
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        pointer = 0
        for element in parent2:
            if element not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = element
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    def run(self):
        self.initialize_population()
        for generation in range(self.num_generations):
            selected_population = self.selection()
            next_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % self.population_size]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                self.mutate(child1)
                self.mutate(child2)
                next_population.extend([child1, child2])
            self.population = next_population[:self.population_size]
            best_individual = min(self.population, key=self.fitness)
            best_distance = self.fitness(best_individual)
            print(f'Generation {generation+1}, Best distance: {best_distance}')
        best_individual = min(self.population, key=self.fitness)
        return best_individual, self.fitness(best_individual)

num_customers = len(shelf_data)
num_vehicles = 1
capacity = 5
demands = shelf_tasks
distance_matrix = matrix
population_size = 20
num_generations = 100
mutation_rate = 0.1

ga = GeneticAlgorithm(num_customers, num_vehicles, capacity, demands, distance_matrix, population_size, num_generations, mutation_rate)
best_route, best_distance = ga.run()
print(f'Best route: {best_route}, Best distance: {best_distance}')
