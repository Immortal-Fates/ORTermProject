import pandas as pd
import random

# 读取CSV文件
df = pd.read_csv('../src/book.csv')

# 创建一个空的列表来存储选中的书籍
selected_books = []

# 创建一个新的DataFrame来存储抽取结果
result_df = pd.DataFrame(columns=['order_id', 'book_id'])

# 记录当前的订单ID
order_id = 1

# 随机抽取500本书
total_books_needed = 500
while total_books_needed > 0:
    # 从所有有剩余书籍的记录中随机选取一条
    available_books = df[df['book_quantity'] > 0]
    if available_books.empty:
        raise ValueError("书籍数量不足，无法抽取500本书")

    random_row = available_books.sample(n=1)
    book_id = random_row['book_id'].values[0]
    book_quantity = random_row['book_quantity'].values[0]

    # 随机确定从这条记录中抽取的书籍数量
    num_books_to_take = min(book_quantity, total_books_needed, random.randint(1, 5))

    for _ in range(num_books_to_take):
        selected_books.append((order_id, book_id))
        order_id += 1

    # 更新原始数据中的书籍数量
    df.loc[df['book_id'] == book_id, 'book_quantity'] -= num_books_to_take

    # 更新需要抽取的书籍总量
    total_books_needed -= num_books_to_take

# 创建结果DataFrame
result_df = pd.DataFrame(selected_books, columns=['order_id', 'book_id'])

# 将结果写入新的CSV文件
result_df.to_csv('../src/selected_books.csv', index=False)

print("500本书已经成功随机抽取并保存到 selected_books.csv 文件中。")
