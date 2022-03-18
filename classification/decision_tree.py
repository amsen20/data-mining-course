import random
import graphviz

from categorizer import *
from math import *

prep(1)
calc_categorizers()
train = get_cat_data("Dataset/Dataset1.csv")


def gini(state):
    tot = sum([state[st] for st in state])
    return 1 - sum([(state[st]/tot)**2 for st in state])


def entropy(state):
    tot = sum([state[st] for st in state])
    ps = [state[st]/tot for st in state]

    def f(x):
        if x < 1e-10:
            return 0
        return -log2(x)*x
    return sum([f(p) for p in ps])


criterion = input("Enter criterion(g/e): ")
if criterion[0] == 'g':
    imp_func = gini
else:
    imp_func = entropy

attrs = set(train)
goal = "income"
attrs.remove(goal)


class Node:
    def __init__(self):
        self.par = -1
        self.edge_label = ""
        self.id = 0
        self.h = 0
        self.av_attrs = set()
        self.answer = None
        self.asked_attr = None
        self.rows = []
        self.childs = {}


def get_state(rows):
    state = {}
    for row in rows:
        cur = train[goal][row]
        if cur not in state:
            state[cur] = 0
        state[cur] += 1
    return state


def get_divs(attr, rows):
    divs = {}
    for row in rows:
        cur = train[attr][row]
        if cur not in divs:
            divs[cur] = []
        divs[cur].append(row)
    return divs


def get_answer(rows):
    ret = []
    for _ in range(5):
        it = random.randint(0, len(rows) - 1)
        ret.append(train[goal][rows[it]])
    return max(set(ret), key=ret.count)


all_rows = list(range(len(train[list(train)[0]])))
random.shuffle(all_rows)
n = len(all_rows)*8//10
root = Node()
root.av_attrs = attrs
root.rows = all_rows[:n]

# visualization:
dot = graphviz.Digraph(comment="Decision Tree")

q = [root]
root.id = 0
ind = 0
while q:
    v = q.pop(0)
    # Set answer to node
    v.answer = get_answer(v.rows)
    if v.h >= 3 or len(v.rows) < 10:
        dot.node(str(v.id), v.answer)
        dot.edge(str(v.par), str(v.id), v.edge_label)
        continue

    state = get_state(v.rows)
    if imp_func(state) == 0:
        dot.node(str(v.id), v.answer)
        dot.edge(str(v.par), str(v.id), v.edge_label)
        continue

    mn_attr = None
    mn_val = inf

    for attr in v.av_attrs:
        divs = get_divs(attr, v.rows)
        imp = 0
        for (_, div) in divs.items():
            cur = imp_func(get_state(div))
            cur = cur * len(div) / len(v.rows)
            imp += cur

        if imp < mn_val:
            mn_val = imp
            mn_attr = attr

    if not mn_attr:
        dot.node(str(v.id), v.answer)
        dot.edge(str(v.par), str(v.id), v.edge_label)
        continue

    v.asked_attr = mn_attr
    divs = get_divs(mn_attr, v.rows)
    for (val, div) in divs.items():
        ind += 1
        u = Node()

        u.id = ind
        u.par = v.id
        u.edge_label = val
        u.h = v.h + 1

        u.rows = div
        u.av_attrs = set(v.av_attrs)
        u.av_attrs.remove(mn_attr)
        v.childs[val] = u
        q.append(u)

    dot.node(str(v.id), v.asked_attr)
    dot.edge(str(v.par), str(v.id), v.edge_label)


def get_out(data):
    it = root
    while True:
        if not it.asked_attr or data[it.asked_attr] not in it.childs:
            return it.answer

        it = it.childs[data[it.asked_attr]]


test_rows = all_rows[n:]
pred = []
out = []
for row in test_rows:
    cur_data = {}
    for col in train:
        cur_data[col] = train[col][row]
    pred.append(get_out(cur_data))
    out.append(train[goal][row])
print(f"acc: {sum([p == o for (p, o) in zip(pred, out)])/len(pred)}")

test = get_cat_data("Dataset/Dataset1_Unknown.csv")
with open("DT.txt", 'w') as f:
    cnt = {}
    for row in range(len(test[list(test)[0]])):
        cur_data = {}
        for col in test:
            cur_data[col] = test[col][row]
        cpred = get_out(cur_data)
        f.write(cpred + "\n")
        if cpred not in cnt:
            cnt[cpred] = 0
        cnt[cpred] += 1

print("output for unknown data: ")
print(cnt)

# For visualization
# dot.render(directory='', view=True)
