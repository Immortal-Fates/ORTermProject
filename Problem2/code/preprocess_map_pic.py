import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_shelf_positions(file_path):
    """
    读取shelf.csv文件，并将书架坐标存储在字典中
    :param file_path: shelf.csv文件的路径
    :return: 书架坐标字典，键为shelf_id，值为(x, y)坐标元组
    """
    shelf_positions = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过文件头
        for row in reader:
            shelf_id, x, y = map(int, row)
            shelf_positions[shelf_id] = (x, y)
    return shelf_positions

def draw_map(ax, shelf_positions):
    """
    绘制基础地图，包括书架和起点
    :param ax: Matplotlib的轴对象
    :param shelf_positions: 书架坐标字典
    """
    # 创建一个21x19的地图
    for x in range(21):
        for y in range(19):
            ax.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='none'))

    # 绘制绿色的格子表示书架
    for (x, y) in shelf_positions.values():
        ax.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='green'))

    # 将原点(0, 0)绘制为红色
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='red'))

def plot_route(ax, shelf_positions, best_route, start_pos=(0, 0)):
    """
    根据小车路径绘制轨迹
    :param ax: Matplotlib的轴对象
    :param shelf_positions: 书架坐标字典
    :param best_route: 小车访问书架的顺序，书架id列表
    :param start_pos: 起点坐标，默认为(0, 0)
    """
    # 定义颜色列表
    colors = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']
    color_index = 0

    current_pos = start_pos
    color = colors[color_index % len(colors)]  # 初始颜色

    for route in best_route:
        for shelf_id in route:
            if shelf_id == 0:
                # 返回原点
                ax.plot([current_pos[0] + 0.5, start_pos[0] + 0.5],
                        [current_pos[1] + 0.5, start_pos[1] + 0.5],
                        color=color, marker='o')
                current_pos = start_pos
                # 更换颜色
                color_index += 1
                color = colors[color_index % len(colors)]
            else:
                # 获取书架坐标
                x1, y1 = shelf_positions[shelf_id]
                ax.plot([current_pos[0] + 0.5, x1 + 0.5],
                        [current_pos[1] + 0.5, y1 + 0.5],
                        color=color, marker='o')
                current_pos = (x1, y1)

def plot_map_with_route(best_route, shelf_file_path='../src/shelf.csv', output_path='../result/map_with_route.png'):
    """
    主函数，读取书架数据，绘制地图和小车轨迹，并保存图片
    :param best_route: 小车访问书架的顺序，书架id列表
    :param shelf_file_path: shelf.csv文件的路径
    :param output_path: 输出图片的路径
    """
    # 读取书架数据
    shelf_positions = read_shelf_positions(shelf_file_path)

    # 创建一个21x19的地图
    fig, ax = plt.subplots(figsize=(10, 9))

    # 绘制基础地图
    draw_map(ax, shelf_positions)

    # 绘制小车轨迹
    plot_route(ax, shelf_positions, best_route)

    # 设置坐标轴范围和标签
    plt.xlim(0, 20)
    plt.ylim(0, 18)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Map with Car Route')

    # 显示地图
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

    print(f"地图图片已保存为 {output_path} 文件。")

# 示例调用
if __name__ == "__main__":
    best_route = [[3, 0, 3, 2, 0, 1, 0, 2, 0, 2, 1, 0, 1, 0, 1]]
    plot_map_with_route(best_route)
