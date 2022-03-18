import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_excel('dataset2.xlsx')
n = len(df['A'])
data = []
for i in range(n):
    cur = {
        'vec': np.array([df['A'][i], df['B'][i], df['C'][i], df['D'][i]]),
        'label': df['class'][i]
    }
    data.append(cur)


def pca(r):
    u = np.zeros(4)
    D = []
    for i in range(n):
        v = data[i]['vec'].copy()
        D.append(v)
        u += v
    u /= n
    D_prime = []
    for i in range(n):
        v_prime = D[i].copy()
        v_prime -= u
        D_prime.append(v_prime)
    sigma = np.zeros((4, 4))
    for vp in D_prime:
        sigma += np.matmul(vp.reshape(4, 1), vp.reshape(1, 4))
    sigma /= n
    w, v = LA.eig(sigma)
    vs = [(w[i], v[i]) for i in range(4)]

    vs.sort(key=lambda x: x[0], reverse=True)
    vs = vs[:r]
    new_data = []
    for i in range(n):
        cur = []
        for _, u in vs:
            cur.append(u.T @ D[i])
        new_data.append({
            'vec': np.array(cur),
            'label': data[i]['label']
        })
    return new_data


def svm(data):
    X = []
    y = []
    for i in range(n):
        X.append(data[i]['vec'])
        y.append(data[i]['label'])
    X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    acc = 0
    guess = clf.predict(X_test)
    for i in range(len(X_test)):
        if guess[i] == y_test[i]:
            acc += 1

    return acc/len(X_test)


print(f"Regular data acc: {svm(data)}")
new_data = pca(2)
print(f"Dimension reduced to 2 (using PCA) data's acc: {svm(new_data)}")
labels_to_pts = {}
for it in new_data:
    clabel = it['label']
    if clabel not in labels_to_pts:
        labels_to_pts[clabel] = []
    labels_to_pts[clabel].append(it['vec'])
for i in labels_to_pts:
    ls = labels_to_pts[i]
    plt.scatter([it[0] for it in ls], [it[1] for it in ls], label=i)
plt.legend()
plt.show()
