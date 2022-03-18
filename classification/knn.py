from categorizer import *
import random

goal = "poisonous"

prep(2)
calc_categorizers()
train = get_cat_data("Dataset/Dataset2.csv")
all_rows = list(range(len(train[list(train)[0]])))
random.shuffle(all_rows)
n = len(all_rows)*8//10
train_rows = all_rows[:n]

k = int(input("Enter k: "))


def get_answer(rows):
    ret = []
    for _ in range(5):
        it = random.randint(0, len(rows) - 1)
        ret.append(train[goal][rows[it]])
    return max(set(ret), key=ret.count)


crr = 0
tot = 0


def gen_dis(row, U):
    def dis(crow):
        ret = 0
        for col in train:
            if col != goal and train[col][crow] != U[col][row]:
                ret += 1
        return ret
    return dis


for row in all_rows[:n]:
    ls = list(all_rows[:n])
    ls.sort(key=gen_dis(row, train))
    pred = get_answer(ls[:k])
    out = train[goal][row]
    tot += 1
    if pred == out:
        crr += 1

print(f"acc: {crr/tot}")
test = get_cat_data("Dataset/Dataset2_Unknown.csv")
cnt = {}
pred = []
for row in range(len(test[list(test)[0]])):
    ls = list(all_rows[:n])
    ls.sort(key=gen_dis(row, test))
    pred.append(get_answer(ls[:k]))

with open(f"KNN{k}.txt", 'w') as f:
    for pr in pred:
        f.write(pr + "\n")
