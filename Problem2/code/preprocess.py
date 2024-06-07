import csv
import numpy as np
from collections import deque
from preprocess_calc_distance import *
def read_shelf_csv(filename):
    shelf_data = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过文件头
        for row in reader:
            shelf_id = int(row[0])
            x = int(row[1])
            y = int(row[2])
            shelf_data[shelf_id] = {'x': x, 'y': y, 'tasks': 0}
    return shelf_data

# 读取书籍列表
def read_book_csv(filename):
    book_dict = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过文件头
        for row in reader:
            book_id, book_quantity, shelf_id = map(int, row)
            book_dict[book_id] = shelf_id
    return book_dict

# 读取选出的书籍列表
def read_selected_books_csv(filename):
    selected_books = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过文件头
        for row in reader:
            order_id, book_id = map(int, row)
            selected_books.append((order_id, book_id))
    return selected_books

depot = (0, 0)  # 原点位置
shelf_data = read_shelf_csv('../src/shelf.csv')
map_size = (40,40)  # 地图大小，根据实际情况调整
adjacency_matrix = path(shelf_data, map_size,depot)
with open('../src/distance.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入书架个数
    n = adjacency_matrix.shape[0]
    writer.writerow([n])
    # 写入 adjacency_matrix 矩阵
    writer.writerows(adjacency_matrix)



# 读取书籍列表
book_dict = read_book_csv('../src/book.csv')

# 读取选出的书籍列表
selected_books = read_selected_books_csv('../src/selected_books.csv')

# 统计每个书架上要取的书籍数量
for order_id, book_id in selected_books:
    shelf_id = book_dict.get(book_id)
    if shelf_id is not None and shelf_id in shelf_data:
        shelf_data[shelf_id]['tasks'] += 1





# 除去所有书架上书籍数量为0的书架
shelf_data = {shelf_id: shelf_info for shelf_id, shelf_info in shelf_data.items() if shelf_info['tasks'] > 0}

# 初始化一个空字典来存储书架信息
shelf_tasks = {}
shelf_coords = {}
# 按照书架顺序输出每个书架上要取的书籍数量
for shelf_id in sorted(shelf_data.keys()):
    shelf_info = shelf_data[shelf_id]
    # print(f"书架 {shelf_id} (坐标: ({shelf_info['x']}, {shelf_info['y']})) 上要取的书籍数量为: {shelf_info['tasks']}")
    # 将书架信息添加到字典中
    shelf_tasks[shelf_id] = shelf_info['tasks']
    # 将书架的x,y坐标添加到shelf_coords字典中
    shelf_coords[shelf_id] = (shelf_info['x'], shelf_info['y'])

shelf_coords = list(shelf_coords.values())
print(shelf_coords)
