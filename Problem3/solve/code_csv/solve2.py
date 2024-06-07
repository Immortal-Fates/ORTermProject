import pulp
import pandas as pd
import numpy as np

# Define the number of trays, book types, storage positions, picking stations, and recycling stations
I = 5  # Number of trays
H = 20  # Number of book types
Q = 10  # Number of storage positions
M = 2  # Number of picking stations
R = 2  # Number of recycling stations

# Load the CSV file
file_path = 'Q2\solve\code_csv\orders_books_2.csv'  # Make sure the file path is correct
df = pd.read_csv(file_path)
# Number of book types
# Initialize the array Ch with zeros
Ch = [0] * (H + 1)  # +1 because book types start from 1 to 20
# Count the occurrences of each book type in the 'book' column
for book_type in df['book']:
    Ch[book_type] += 1
# Since Ch[0] is not used (as book types start from 1), we can ignore it
Ch = Ch[1:]

# Load the CSV file
file_path = 'Q2\solve\code_csv\pallets_books_1_2.csv'  # Make sure the file path is correct
df = pd.read_csv(file_path)
# Number of trays and book types

# Initialize the Oih array with zeros
Oih = np.zeros((I, H), dtype=int)

# Fill the Oih array with counts of each book type in each tray
for index, row in df.iterrows():
    book_type = row['book']
    tray = row['tuopan']
    Oih[tray - 1][book_type - 1] += 1

Jiq = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

# Jiq = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]


Dqm = [[4, 7],
       [5, 6],
       [3, 8],
       [4, 7],
       [5, 6],
       [6, 3],
       [7, 6],
       [9, 2],
       [7, 6],
       [8, 5]]

Dmq = [[4, 5, 3, 4, 5, 6, 7, 9, 7, 8],
       [7, 6, 8, 7, 6, 3, 6, 2, 6, 5]]

Dmr = [[8, 4],
       [3, 9]]

# Define the problem
prob = pulp.LpProblem("Library_Book_Picking", pulp.LpMinimize)

# Define decision variables
x = pulp.LpVariable.dicts("x", range(I), 0, 1, cat='Binary')
y = pulp.LpVariable.dicts("y", range(I), 0, 1, cat='Binary')
k = pulp.LpVariable.dicts("k", [(i, h) for i in range(I) for h in range(H)], 0, cat='Integer')
p = pulp.LpVariable.dicts("p", [(i, m) for i in range(I) for m in range(M)], 0, 1, cat='Binary')
g = pulp.LpVariable.dicts("g", [(i, q) for i in range(I) for q in range(Q)], 0, 1, cat='Binary')
f = pulp.LpVariable.dicts("f", [(i, r) for i in range(I) for r in range(R)], 0, 1, cat='Binary')
d_iqm = pulp.LpVariable.dicts("d_iqm", [(i, q, m) for i in range(I) for q in range(Q) for m in range(M)], 0, cat='Integer')
d_imq = pulp.LpVariable.dicts("d_imq", [(i, m, q) for i in range(I) for m in range(M) for q in range(Q)], 0, cat='Integer')
d_imr = pulp.LpVariable.dicts("d_imr", [(i, m, r) for i in range(I) for m in range(M) for r in range(R)], 0, cat='Integer')
z = pulp.LpVariable.dicts("z", range(Q), 0, 1, cat='Binary')

# Define the objective function
prob += (pulp.lpSum(d_iqm[(i, q, m)] for i in range(I) for q in range(Q) for m in range(M)) +
         pulp.lpSum(d_imq[(i, m, q)] for i in range(I) for m in range(M) for q in range(Q)) +
         pulp.lpSum(d_imr[(i, m, r)] for i in range(I) for m in range(M) for r in range(R)))

# Add constraints
MAX = 100000  # A large number

#(1)
# Quantity constraints
for i in range(I):
    for h in range(H):
        prob += k[(i, h)] <= Oih[i][h]
        prob += k[(i, h)] >= 0
    prob += pulp.lpSum(k[(i, h)] for h in range(H)) + MAX * (1 - x[i]) >= 1
    prob += pulp.lpSum(k[(i, h)] for h in range(H)) - pulp.lpSum(Oih[i][h] for h in range(H)) * x[i] <= 0
    prob += y[i] + MAX * (pulp.lpSum(Oih[i][h] for h in range(H)) - pulp.lpSum(k[(i,h)] for h in range(H))) >= 1
    prob += y[i] * pulp.lpSum(Oih[i][h] for h in range(H)) <= pulp.lpSum(k[(i,h)] for h in range(H))
for h in range(H):
    prob += pulp.lpSum(k[(i, h)] for i in range(I)) == Ch[h]

#(2)
# Matching constraints
for i in range(I):
    prob += pulp.lpSum(p[(i, m)] for m in range(M)) <= 1
    prob += pulp.lpSum(p[(i, m)] for m in range(M)) - x[i] == 0

for i in range(I):
    for m in range(M):
        for q in range(Q):
            prob += d_iqm[(i, q, m)] + MAX * (3 - x[i] - p[(i, m)] - Jiq[i][q]) >= Dqm[q][m]
            prob += d_iqm[(i, q, m)] - MAX * x[i] <= 0

#(3)
# Storage constraints
for i in range(I):
    prob += x[i] - y[i] >= 0
    prob += pulp.lpSum(g[(i, q)] for q in range(Q)) <= 1
    prob += y[i] + pulp.lpSum(g[(i, q)] for q in range(Q)) == x[i]
    prob += x[i] - pulp.lpSum(g[(i, q)] for q in range(Q)) >= 0
    prob += pulp.lpSum(g[(i, q)] for q in range(Q)) + pulp.lpSum(f[(i, r)] for r in range(R)) == x[i]
    prob += pulp.lpSum(d_imq[(i, m, q)] for m in range(M) for q in range(Q)) + pulp.lpSum(d_imr[(i, m, r)] for m in range(M) for r in range(R)) + MAX * (1 - x[i]) >= 1

for i in range(I):
    for q in range(Q):
        prob += g[(i, q)] - z[q] <= 0
        prob += z[q] + Jiq[i][q] - x[i] <= 1
        # prob += x[i] * Jiq[i][q] <= z[q]
        for m in range(M):
            prob += d_imq[(i, m, q)] - MAX * (x[i] - y[i]) <= 0
            prob += d_imq[(i, m, q)] + MAX * (4 - x[i] + y[i] - p[(i, m)] - g[(i, q)] - z[q]) >= Dmq[m][q]

for q in range(Q):
    prob += pulp.lpSum(g[(i, q)] for i in range(I)) <= 1
    # for i in range(I):
    #      prob += (Jiq[i][q] * x[i]) <= z[q]

#(4)
# Recycling constraints
for i in range(I):
    prob += pulp.lpSum(k[(i, h)] for h in range(H)) - pulp.lpSum(Oih[i][h] for h in range(H)) + MAX * (1 - y[i]) >= 0
    prob += pulp.lpSum(f[(i, r)] for r in range(R)) <= 1
    prob += pulp.lpSum(f[(i, r)] for r in range(R)) == y[i]
    prob += x[i] - pulp.lpSum(f[(i, r)] for r in range(R)) >= 0

for i in range(I):
    for m in range(M):
        for r in range(R):
            prob += d_imr[(i, m, r)] + MAX * (4 - x[i] - y[i] - p[(i, m)] - f[(i, r)]) >= Dmr[m][r]

# Solve the problem
prob.solve()

# Extract the solution
x_solution = [x[i].varValue for i in range(I)]
y_solution = [y[i].varValue for i in range(I)]
k_solution = [[k[(i, h)].varValue for h in range(H)] for i in range(I)]
p_solution = [[p[(i, m)].varValue for m in range(M)] for i in range(I)]
g_solution = [[g[(i, q)].varValue for q in range(Q)] for i in range(I)]
f_solution = [[f[(i, r)].varValue for r in range(R)] for i in range(I)]
d_iqm_solution = [[[d_iqm[(i, q, m)].varValue for m in range(M)] for q in range(Q)] for i in range(I)]
d_imq_solution = [[[d_imq[(i, m, q)].varValue for q in range(Q)] for m in range(M)] for i in range(I)]
d_imr_solution = [[[d_imr[(i, m, r)].varValue for r in range(R)] for m in range(M)] for i in range(I)]
z_solution = [z[q].varValue for q in range(Q)]

# # Print the solution
# print("x:", x_solution)
# print("y:", y_solution)
# print("k:", k_solution)
# print("p:", p_solution)
# print("g:", g_solution)
# print("f:", f_solution)
# print("d_iqm:", d_iqm_solution)
# print("d_imq:", d_imq_solution)
# print("d_imr:", d_imr_solution)
# print("z:", z_solution)

# # Print the solution
# print("x solution:")
# for i in range(I):
#     print(f"x[{i}] = {x_solution[i]}")

# print("\ny solution:")
# for i in range(I):
#     print(f"y[{i}] = {y_solution[i]}")

# print("\nk solution:")
# for i in range(I):
#     for h in range(H):
#         print(f"k[{i}][{h}] = {k_solution[i][h]}")

# print("\np solution:")
# for i in range(I):
#     for m in range(M):
#         print(f"p[{i}][{m}] = {p_solution[i][m]}")

# print("\ng solution:")
# for i in range(I):
#     for q in range(Q):
#         print(f"g[{i}][{q}] = {g_solution[i][q]}")

# print("\nf solution:")
# for i in range(I):
#     for r in range(R):
#         print(f"f[{i}][{r}] = {f_solution[i][r]}")

print("\nd_iqm solution:")
for i in range(I):
    for q in range(Q):
        for m in range(M):
            if(d_iqm_solution[i][q][m] != 0):
                print(f"d_iqm[{i}][{q}][{m}] = {d_iqm_solution[i][q][m]}")

print("\nd_imq solution:")
for i in range(I):
    for m in range(M):
        for q in range(Q):
            if(d_imq_solution[i][m][q] != 0):
                print(f"d_imq[{i}][{m}][{q}] = {d_imq_solution[i][m][q]}")

print("\nd_imr solution:")
for i in range(I):
    for m in range(M):
        for r in range(R):
            if(d_imr_solution[i][m][r] != 0):
                print(f"d_imr[{i}][{m}][{r}] = {d_imr_solution[i][m][r]}")

# print("\nz solution:")
# for q in range(Q):
#     print(f"z[{q}] = {z_solution[q]}")
print(Oih)