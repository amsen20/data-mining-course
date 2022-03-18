import random
from csv import *
from math import *
import matplotlib.pyplot as plt

rows = []

with open("Dataset1.csv", "r") as f:
    csv_f = reader(f, delimiter=',')
    for row in csv_f:
        if row[0] == 'X':
            continue
        rows.append(tuple(map(float, row)))

K = int(input("Enter k: "))


def initialize_centers(x, k):
    return random.sample(x, k)


def find_closest_centers(x, centers):
    idx = {i: set() for i in range(len(centers))}

    def dis(A, B):
        ax, ay = A
        bx, by = B
        return sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    for i, pt in enumerate(x):
        mn_dis, mn_cen = inf, -1
        for j, cen in enumerate(centers):
            d = dis(pt, cen)
            if d < mn_dis:
                mn_dis = d
                mn_cen = j
        idx[mn_cen].add(i)
    return idx


def compute_means(x, idx, k):
    centers = []
    for i in idx:
        pt = (0, 0)
        for j in idx[i]:
            pt = (pt[0] + x[j][0], pt[1] + x[j][1])
        pt = (pt[0] / len(idx[i]), pt[1] / len(idx[i]))
        centers.append(pt)
    return centers


cs = initialize_centers(rows, K)
cs_history = {i: [cs[i]] for i in range(K)}
for _ in range(15):
    idx = find_closest_centers(rows, cs)
    cs = compute_means(rows, idx, K)
    for i in range(K):
        cs_history[i].append(cs[i])

idx = find_closest_centers(rows, cs)
for i in idx:
    ls = idx[i]
    plt.scatter([rows[i][0] for i in ls], [rows[i][1] for i in ls], label=i)
plt.legend()
for k, history in cs_history.items():
    plt.plot([c[0] for c in history], [c[1] for c in history], marker='x', color='black')
plt.show()
