import copy

from matplotlib import pyplot as plt

from preprocess import *
from preprocess_calc_distance import *
from preprocess_map_pic import plot_map_with_route


class Ant:
    def __init__(self, num_customers, num_vehicles, capacity, demands):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.routes = [[] for _ in range(num_vehicles)]
        self.route_num = 1
        self.route = [[] for _ in range(num_vehicles)]
        self.loads = [0] * num_vehicles
        self.demands = demands
        self.visited = set()

    def add_route(self):
        self.route.append([])

    def add_customer_to_route(self, route_num, customer):
        while len(self.route) <= route_num:
            self.add_route()
        self.route[route_num].append(customer)

    def visit_customer(self, vehicle, customer):
        self.routes[vehicle].append(customer)
        self.add_customer_to_route(self.route_num, customer)
        demand = self.demands[customer]  # 调整索引以适应需求数组
        if (self.capacity - self.loads[vehicle] >= demand):  # 如果车够空，全都装掉
            self.loads[vehicle] += demand
            self.demands[customer] = 0
        else:  # 如果不空，装满为止
            self.demands[customer] -= (self.capacity - self.loads[vehicle])
            self.loads[vehicle] = self.capacity
        self.visited.add(customer)

    def can_visit(self, vehicle, customer):
        return (self.loads[vehicle] < self.capacity)

    def return_to_depot(self, vehicle):
        self.routes[vehicle].append(0)
        self.add_customer_to_route(self.route_num, 0)
        self.route_num += 1
        self.loads[vehicle] = 0


class ACO:
    def __init__(self, num_customers, num_vehicles, capacity, demands, distance_matrix, num_ants, num_iterations,
                 alpha=1, beta=5, evaporation_rate=0.3):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.demands = demands
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_matrix = np.ones((num_customers + 1, num_customers + 1))
        self.best_routes = None
        self.best_distance = 10000000
        self.dis = []

    def plot_best_distances(self): # 绘制收敛曲线
        plt.plot(range(1, self.num_iterations + 1), self.dis)
        plt.xlabel('Iteration')
        plt.ylabel('Best Distance')
        plt.title('Best Distance vs Iteration')
        plt.grid(True)
        plt.savefig('../result/three_eva_rate_aco_convergence.png')
        plt.show()


    def run(self):
        for iteration in range(self.num_iterations):
            # print(iteration)
            ants = [Ant(self.num_customers, self.num_vehicles, self.capacity, copy.deepcopy(demands)) for _ in
                    range(self.num_ants)]
            for ant in ants:
                self.construct_solution(ant)
                distance = self.calculate_total_distance(ant.routes)
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_routes = ant.routes
            self.dis.append(self.best_distance)
            self.update_pheromones(ants)
            # print(f'Iteration {iteration + 1}/{self.num_iterations}, Best distance: {self.best_distance}')

        return self.best_routes, self.best_distance

    def construct_solution(self, ant):
        for vehicle in range(self.num_vehicles):
            current_customer = 0  # 从仓库开始
            # print(ant.demands.values())
            while any(idemand > 0 for idemand in ant.demands.values()):
                # print(ant.demands.values())
                next_customer = self.select_next_customer(ant, current_customer)
                # print(current_customer,next_customer)
                if next_customer is not None:
                    if ant.can_visit(vehicle, next_customer):
                        ant.visit_customer(vehicle, next_customer)
                        current_customer = next_customer
                    else:  # 返回仓库卸货
                        ant.return_to_depot(vehicle)
                        current_customer = 0  # 返回仓库
                else:
                    ant.return_to_depot(vehicle)  # 确保车辆最终返回仓库
                    current_customer = 0
            # print(ant.routes)
            # break  # 没有更多客户可以访问

    def select_next_customer(self, ant, current_customer):
        probabilities = []
        for customer in range(0, self.num_customers + 1):
            if ant.demands.get(customer, 0) == 0 or customer == current_customer:
                continue
            pheromone = self.pheromone_matrix[current_customer][customer] ** self.alpha
            visibility = (1.0 / self.distance_matrix[current_customer][customer]) ** self.beta
            probabilities.append((pheromone * visibility, customer))

        if not probabilities:
            return None

        total_prob = sum(prob for prob, customer in probabilities)
        if total_prob == 0:
            return None

        probabilities = [(prob / total_prob, customer) for prob, customer in probabilities]
        rand = np.random.random()
        cumulative_prob = 0.0

        for prob, customer in probabilities:
            cumulative_prob += prob
            if rand <= cumulative_prob:
                return customer

        return None

    def calculate_total_distance(self, routes): # 根据路径列表计算最短路径的长度
        total_distance = 0
        for route in routes:
            if route:
                distance = 0
                # print("route", route)
                for i in range(len(route) - 1):
                    distance += self.distance_matrix[route[i]][route[i + 1]]
                distance += self.distance_matrix[route[-1]][0]  # 返回仓库
                distance += self.distance_matrix[0][route[0]]
                total_distance += distance
        return total_distance

    def update_pheromones(self, ants):
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        for ant in ants:
            distance = self.calculate_total_distance(ant.routes)
            for route in ant.routes:
                for i in range(len(route) - 1):
                    self.pheromone_matrix[route[i]][route[i + 1]] += 1.0 / distance
                if route:
                    self.pheromone_matrix[route[-1]][route[0]] += 1.0 / distance  # 返回仓库


# 数据预处理

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

# 示例输入数据
num_customers = len(shelf_data)  # 书架数量
num_vehicles = 4  # 运输小车数量
capacity = 5  # 小车容量
demands = shelf_tasks
distance_matrix = matrix  # 距离矩阵
num_ants = 5  # 蚂蚁数量
num_iterations = 100  # 迭代次数

# 运行ACO算法
# aco = ACO(num_customers, num_vehicles, capacity, demands, distance_matrix, num_ants, num_iterations)
# best_routes, best_distance = aco.run()
# aco.plot_best_distances() # 利用aco内部函数进行迭代曲线的绘制
# print(f'Best routes: {best_routes}, Best distance: {best_distance}')
# plot_map_with_route(best_routes) # 使用 preprocess_map_pic 文件中的函数进行图像绘制



## 灵敏度分析

# 参数设置
alphas = np.arange(0.5, 3.5, 1.0)
betas = np.arange(0.5, 5.0, 1.0)
evaporation_rates = np.arange(0.1, 0.8, 0.2)
num_ants = 5
num_iterations = 100

results = []

# 运行算法
for alpha in alphas:
    for beta in betas:
        for evaporation_rate in evaporation_rates:
            aco = ACO(num_customers, num_vehicles, capacity, demands, distance_matrix, num_ants, num_iterations,
                      alpha, beta, evaporation_rate)
            best_routes, best_distance = aco.run()
            results.append((alpha, beta, evaporation_rate, best_distance))
            print(f'Alpha: {alpha}, Beta: {beta}, Evaporation Rate: {evaporation_rate}, Best Distance: {best_distance}')

# 分析结果
results = np.array(results)

# 创建3D图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 散点图的数据
sc = ax.scatter(results[:, 0], results[:, 1], results[:, 2], c=results[:, 3], cmap='coolwarm')

# 添加颜色条
cbar = plt.colorbar(sc)
cbar.set_label('Best Distance')

# 设置轴标签
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('Evaporation Rate')

# 设置标题
plt.title('3D ACO Performance Analysis')

plt.show()


