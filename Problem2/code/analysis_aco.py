from preprocess import *
from preprocess_calc_distance import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

# 假设您有一个ACO类定义好了
from calc_aco import *

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

# 参数范围
# alphas = np.arange(0.5, 5.5, 0.5)
# betas = np.arange(0.5, 5.5, 0.5)
# evaporation_rates = np.arange(0.1, 1.1, 0.1)
alphas = 1
betas = 5
evaporation_rates = 0.5

def run_aco():
    # 设置ACO算法的参数
    aco = ACO(num_customers, num_vehicles, capacity, demands, distance_matrix, num_ants, num_iterations)
    best_routes, best_distance = aco.run()
    return (best_distance)

# 运行算法50次收集数据
results = [run_aco() for _ in range(50)]

# 计算统计数据
mean_result = np.mean(results)
std_deviation = np.std(results)
best_result = min(results)
worst_result = max(results)

# 输出统计数据
print(f"平均成本: {mean_result}")
print(f"标准偏差: {std_deviation}")
print(f"最佳成本: {best_result}")
print(f"最差成本: {worst_result}")

# 绘制结果直方图
plt.hist(results, bins=10, color='blue', alpha=0.7)
plt.title('ACO Algorithm Performance Distribution')
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.show()

