import csv
import pandas as pd
import matplotlib.pyplot as plt

# 定义节点类型的颜色
node_colors = {
    "1": "lightgray",#通道
    "2": "green",#储位节点
    "3": "black",#障碍物
    "4": "red",#挑拣位置
    "5": "yellow"#空盘回收
}

# 从CSV文件读取数据
df = pd.read_csv('./map1.csv')


def draw():
    # 创建绘图
    fig, ax = plt.subplots(figsize=(4, 3))

    # 跟踪已经添加到图例中的类型
    legend_labels = set()

    # 创建一个字典来存储每个坐标点的颜色
    grid_colors = {}
    for node in nodes:
        node_type = node["TYPE"]
        x, y = node["X"], node["Y"]
        color = node_colors.get(str(node_type), "gray")
        grid_colors[(x, y)] = color

    # 绘制整个网格背景
    for x in range(9):
        for y in range(7):
            color = grid_colors.get((x, y), "white")  # 默认背景颜色为白色
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))

    # 绘制节点和连通性
    for node in nodes:
        node_type = node["TYPE"]
        x, y = node["X"], node["Y"]
        color = node_colors.get(str(node_type), "gray")

        # 仅在类型未添加到图例时添加标签
        if node_type not in legend_labels:
            ax.scatter(x + 0.5, y + 0.5, c=color, label=node_type)
            legend_labels.add(node_type)
        else:
            ax.scatter(x + 0.5, y + 0.5, c=color)

    # 设置图例
    #ax.legend()

    # 设置坐标范围
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.set_xticks(range(9))
    ax.set_yticks(range(7))
    ax.grid(True)

    # 显示绘图
    plt.show()




# 创建样本数据
nodes = []
for _, row in df.iterrows():
    # print(_,row)
    node = {
        "TYPE": row['TYPE'],
        "X": row['X'],
        "Y": row['Y'],
    }
    nodes.append(node)
draw()