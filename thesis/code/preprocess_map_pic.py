import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 创建一个21x19的地图
fig, ax = plt.subplots(figsize=(10, 9))

# 绘制每一个格子
for x in range(21):
    for y in range(19):
        ax.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='none'))

# 读取shelf.csv文件并绘制绿色的格子
with open('../src/shelf.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过文件头
    for row in reader:
        shelf_id, x, y = map(int, row)
        ax.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='green'))

# 将原点(0, 0)绘制为红色
ax.add_patch(patches.Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='red'))

# 设置坐标轴范围和标签
plt.xlim(0, 20)
plt.ylim(0, 18)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Map')

# 显示地图
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.savefig('../result/map_image.png')
plt.show()

print("地图图片已保存为 map_image.png 文件。")