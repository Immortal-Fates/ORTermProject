import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 定义节点类型的颜色
node_colors = {
    "1": "lightgray",  # 通道
    "2": "green",      # 储位节点
    "3": "black",      # 障碍物
    "4": "red",        # 挑拣位置
    "5": "yellow",     # 空盘回收
    "wall": "lightgray" # 墙壁
}

# 从CSV文件读取数据
df = pd.read_csv('Q2/draw_map/map_code/map1.csv')

def draw():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 创建一个字典来存储每个坐标点的颜色和高度
    grid_colors_heights = {}
    for node in nodes:
        node_type = node["TYPE"]
        x, y = node["X"], node["Y"]
        color = node_colors.get(str(node_type), "gray")
        height = 2 if node_type == 2 else (0 if node_type == 1 else 1)
        grid_colors_heights[(x, y)] = (color, height)

    # 绘制实心立方体
    def draw_solid_cube(ax, position, color, height):
        x, y, z = position
        if height > 0:
            vertices = [
                [x, y, z],
                [x+1, y, z],
                [x+1, y+1, z],
                [x, y+1, z],
                [x, y, z+height],
                [x+1, y, z+height],
                [x+1, y+1, z+height],
                [x, y+1, z+height]
            ]
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
                [vertices[3], vertices[0], vertices[4], vertices[7]],  # 左面
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                [vertices[4], vertices[5], vertices[6], vertices[7]]   # 顶面
            ]
            poly3d = [[list(p) for p in face] for face in faces]
            ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors='k', alpha=.75))

    # 绘制外围墙
    wall_height = 3
    for x in range(-1, 9):  # 增加外层
        draw_solid_cube(ax, (x, -1, 0), node_colors["wall"], wall_height)
        draw_solid_cube(ax, (x, 6, 0), node_colors["wall"], wall_height)
    for y in range(0, 6):  # 增加外层
        draw_solid_cube(ax, (-1, y, 0), node_colors["wall"], wall_height)
        draw_solid_cube(ax, (8, y, 0), node_colors["wall"], wall_height)

    # 绘制整个网格背景
    for x in range(8):
        for y in range(6):
            color, height = grid_colors_heights.get((x, y), ("white", 1))  # 默认背景颜色为白色，高度为1
            draw_solid_cube(ax, (x, y, 0), color, height)

    # 设置坐标范围
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 6)
    ax.set_zlim(0, 4)
    ax.set_xticks(range(-1, 9))
    ax.set_yticks(range(-1, 7))
    ax.set_zticks(range(5))

    # 设置坐标比例一致
    ax.set_box_aspect([9, 7, 4])  # 使得x, y, z轴比例一致

    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示绘图
    plt.show()

# 创建样本数据
nodes = []
for _, row in df.iterrows():
    node = {
        "TYPE": row['TYPE'],
        "X": row['X'],
        "Y": row['Y'],
    }
    nodes.append(node)

draw()
