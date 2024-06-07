import pandas as pd
import matplotlib.pyplot as plt

# 定义节点类型的颜色
node_colors = {
    "1": "lightgray",
    "2": "green",
    "3": "blue",
    "4": "black",
    "5": "red",
    "6": "orange",
    "7": "yellow"
}

# 从CSV文件读取数据
df = pd.read_csv('./map2.csv')

# 打印前几行，查看数据
print(df.head())

print(df.columns)


# 解析NEIGHBORS字段为列表的元组
def parse_neighbors(neighbors_str):
    if pd.isna(neighbors_str) or neighbors_str == '':
        return []
    neighbors = []
    for neighbor in neighbors_str.split(';'):
        x, y = map(int, neighbor.split(':'))
        neighbors.append((x, y))
    return neighbors
def draw():
    # 创建绘图
    fig, ax = plt.subplots(figsize=(16, 11))

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
    for x in range(33):
        for y in range(23):
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


    # # 设置图例
    # ax.legend()

    # 设置坐标范围
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 22)
    ax.set_xticks(range(33))
    ax.set_yticks(range(23))
    ax.grid(True)

    # 显示绘图
    plt.show()




# 创建样本数据
nodes = []
for _, row in df.iterrows():
    print(_,row)
    node = {
        "TYPE": row['TYPE'],
        "X": row['X'],
        "Y": row['Y'],
    }
    nodes.append(node)
draw()


