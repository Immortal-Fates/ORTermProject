import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import random
import matplotlib.pyplot as plt
from preprocess import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 'SimHei' 字体
# 读取数据
# 尝试使用不同的分隔符（如逗号、分号等）来读取文件
# shelf_df = pd.read_csv('../src/shelf.csv')
# distance_df = pd.read_csv('../src/distance.csv', header=None)

# 提取书架信息
shelf_id = list(shelf_data.keys())
# shelf_coords = shelf_data[['x', 'y']].values
N = len(shelf_id)  # 书架数量
M = 4  # 运输小车数量
Q = 5  # 运输小车容量

def read_matrix_csv(filename):
    import csv
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        matrix_size = int(next(reader)[0])  # 读取第一行，获取矩阵的大小
        matrix = list(reader)
    matrix = [[float(cell) for cell in row] for row in matrix]  # 将数据转换为浮点数
    matrix = np.array(matrix)  # 将列表转换为numpy数组
    return matrix_size, matrix

# 读取CSV文件
matrix_size, matrix = read_matrix_csv('..\src\distance.csv')
# 初始化距离矩阵
distance_matrix = matrix

# 初始化需求
# demand = shelf_tasks
demand = np.array(list(shelf_tasks.values()))

# 定义 SDVRP 模型
class SDVRP:
    def __init__(self, distance_matrix, demand, capacity, num_vehicles=4):
        self.distance_matrix = distance_matrix
        self.demand = demand
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.num_nodes = len(distance_matrix)
        self.best_solution = None
        self.best_distance = float('inf')
        self.distance_history = []  # 记录每次迭代后的最佳距离

    def calculate_distance(self, route):
        """计算路线总距离"""
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i], route[i + 1]]
        return distance

    def evaluate_solution(self, solution):
        """评估解的质量，计算总距离"""
        total_distance = 0
        for route in solution:
            total_distance += self.calculate_distance(route)
        return total_distance

    def generate_initial_solution(self):
        """生成初始解，每个书架分配给不同的车辆"""
        solution = []
        remaining_demand = self.demand.copy()
        
        while np.sum(remaining_demand) > 0:
            route = [0]  # 从原点开始
            remaining_capacity = self.capacity

            while remaining_capacity > 0 and np.sum(remaining_demand) > 0:
                available_shelves = np.where(remaining_demand > 0)[0]  # 找到可访问的书架
                if len(available_shelves) == 0:
                    break
                shelf_index = random.choice(available_shelves)
                if remaining_capacity >= remaining_demand[shelf_index]:
                    route.append(shelf_index + 1)
                    remaining_capacity -= remaining_demand[shelf_index]
                    remaining_demand[shelf_index] = 0
                else:  # 需求量大于剩余容量，需要拆分
                    route.append(shelf_index + 1)
                    remaining_demand[shelf_index] -= remaining_capacity
                    remaining_capacity = 0  # 剩余容量为0

            route.append(0)  # 返回原点
            solution.append(route)

        return solution

    def split_routes(self, routes, num_vehicles):
        """根据路径长度平均分配给小车"""
        route_distances = [self.calculate_distance(route) for route in routes]
        sorted_routes = sorted(zip(route_distances, routes), key=lambda x: x[0], reverse=True)
        
        vehicle_routes = [[] for _ in range(num_vehicles)]
        vehicle_loads = [0] * num_vehicles
        
        for dist, route in sorted_routes:
            min_vehicle_index = vehicle_loads.index(min(vehicle_loads))
            vehicle_routes[min_vehicle_index].append(route)
            vehicle_loads[min_vehicle_index] += dist
        
        return vehicle_routes

    def solve(self, max_iterations=2000):
        """使用贪婪算法求解 SDVRP"""
        initial_solution = self.generate_initial_solution()
        initial_distance = self.evaluate_solution(initial_solution)
        
        best_solution = initial_solution
        best_distance = initial_distance
        self.distance_history.append(best_distance)

        for _ in range(max_iterations):
            current_solution = self.generate_initial_solution()
            current_distance = self.evaluate_solution(current_solution)
            if current_distance < best_distance:
                best_solution = current_solution
                best_distance = current_distance
            self.distance_history.append(best_distance)

        vehicle_solutions = self.split_routes(best_solution, self.num_vehicles)
        self.best_solution = vehicle_solutions
        self.best_distance = best_distance
        return self.best_solution, self.best_distance


# 实例化 SDVRP 模型
sdvrp = SDVRP(distance_matrix, demand, Q, M)
best_solution, best_distance = sdvrp.solve()

# 打印结果
for i, vehicle_routes in enumerate(best_solution):
    print(f"小车 {i+1} 的最佳路线:")
    for route in vehicle_routes:
        print(f"  路线: {route}")
print("最佳距离:", best_distance)

# 绘制收敛曲线
plt.plot(sdvrp.distance_history)
plt.xlabel('Iteration')
plt.ylabel('Total Distance')
plt.title('Convergence Curve')
plt.savefig('../result/greedy_convergence_curve.png')
plt.show()

# 假设我们有一个原点坐标 (0, 0) 和书架的坐标
origin = np.array([0, 0])
shelf_coords = np.array(list(shelf_data.values()))
shelf_coords_list = [(shelf['x'], shelf['y']) for shelf in shelf_coords]
print(shelf_coords_list)
# 绘制四辆小车的路径
colors = ['r', 'g', 'b', 'c']
markers = ['o', 's', 'D', 'P']

# 创建4个子图
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()  # 将2x2的子图展平为1维数组，方便遍历

for i, vehicle_routes in enumerate(best_solution):
    ax = axs[i]
    for route in vehicle_routes:
        route_coords = [origin] + [shelf_coords_list[shelf_id - 1] for shelf_id in route[1:-1]] + [origin]
        route_coords = np.vstack(route_coords)  # 使用 vstack 来堆叠坐标
        ax.plot(route_coords[:, 0], route_coords[:, 1], color=colors[i], marker=markers[i], label=f'小车 {i+1}' if route == vehicle_routes[0] else "")
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'小车 {i+1} 的路径')
        ax.legend()

plt.tight_layout()
plt.savefig('../result/greedy_vehicle_routes.png')
plt.show()
