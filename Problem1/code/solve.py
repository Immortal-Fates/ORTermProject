import math
import csv
import matplotlib.pyplot as plt

# 坐标系
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

# 旅行商问题
def tsp(cities):
    start_city = cities[0]
    current_city = start_city
    path = [current_city]
    unvisited_cities = cities[1:]

    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda city: current_city.distance(city))
        path.append(nearest_city)
        unvisited_cities.remove(nearest_city)
        current_city = nearest_city

    # 返回到起点
    path.append(start_city)
    return path

# 从CSV文件读取城市坐标
def read_cities_from_csv(filename="../src/position.csv"):
    cities = []
    with open(filename, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            x, y = int(row[0]), int(row[1])
            cities.append(City(x, y))
    return cities

# 计算路径长度
def calculate_path_length(path):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += path[i].distance(path[i + 1])
    return total_distance

# 绘制运动轨迹路线
def plot_path(path):
    x = [city.x for city in path]
    y = [city.y for city in path]
    plt.plot(x, y, marker='o', linestyle='-', markersize=5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Car Path")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 从CSV文件读取城市坐标
    cities = read_cities_from_csv()

    # 使用贪婪算法求解旅行商问题
    best_path = tsp(cities)

    # 打印结果
    print("城市坐标：")
    for city in cities:
        print(f"({city.x}, {city.y})")
    print("\n最佳路径：")
    for city in best_path:
        print(f"({city.x}, {city.y})", end=" ")
    print("\n路径长度：", calculate_path_length(best_path))

    # 绘制运动轨迹路线
    plot_path(best_path)