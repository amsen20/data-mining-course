import random

from categorizer import *

goal = "disease"

prep(3)
calc_categorizers()
train = get_cat_data("Dataset/Dataset3.csv")
test = get_cat_data("Dataset/Dataset3_Unknown.csv")
all_rows = list(range(len(train[list(train)[0]])))
random.shuffle(all_rows)
n = len(all_rows)*8//10

classes = set()
for cls in train[goal]:
    classes.add(cls)

cls_p = {cls: 0 for cls in classes}
for row in all_rows[:n]:
    cls_p[train[goal][row]] += 1

for cls in cls_p:
    cls_p[cls] /= n

cache = {}


def calc(col, val, cls):
    obj = (col, val, cls)
    if obj in cache:
        return cache[obj]

    tot = 0
    cnt = 0
    for row in all_rows[:n]:
        if train[goal][row] != cls:
            continue
        cnt += 1
        if train[col][row] == val:
            tot += 1
    ret = 0 if cnt == 0 else tot/cnt
    cache[obj] = ret
    return ret


def choose(data):
    probs = {cls: cls_p[cls] for cls in classes}
    for col in data:
        if col == goal:
            continue
        for cls in classes:
            probs[cls] *= calc(col, data[col], cls)
    ans = list(classes)
    ans.sort(key=lambda x: probs[x], reverse=True)
    return ans[0]


crr = 0
tot = 0
for row in all_rows[n:]:
    cdata = {}
    for col in train:
        cdata[col] = train[col][row]
    pred = choose(cdata)
    out = train[goal][row]
    tot += 1
    if pred == out:
        crr += 1

print(f"acc: {crr/tot}")


with open("NB.txt", 'w') as f:
    for row in range(len(test[list(test)[0]])):
        cdata = {}
        for col in train:
            if col != goal:
                cdata[col] = test[col][row]
        cpred = choose(cdata)
        cpred = 1 if cpred == 'B' else 0
        f.write(str(cpred) + "\n")
