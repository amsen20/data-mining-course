import random

import numpy as np
import matplotlib.pyplot as plt
from consts import *

dtest = False


def load_train_data():
    data = np.load('data.npz')
    x1 = data['x1']
    x2 = data['x2']
    y = data['y']
    x1_test = data['x1_test']
    x2_test = data['x2_test']
    y_test = data['y_test']
    return x1, x2, y, x1_test, x2_test, y_test


def show_errs(errs, save=False):
    x = list(range(NUMBER_OF_ITERATIONS))

    plt.plot(x, errs)
    plt.xlabel("iteration number")
    plt.ylabel("error")
    plt.title("Gradient descent")
    if save:
        plt.savefig('errs.png')
    plt.show()


def show_reg(x1, x2, y, y_pred, save=False):
    ax = plt.axes(projection='3d')
    plt.xlabel = 'x1'
    plt.ylabel = 'x2'
    plt.zlabel = 'y'
    ax.scatter(x1, x2, y, c='r')
    ax.plot_trisurf(x1, x2, y_pred, linewidth=0.2, antialiased=True)
    if save:
        plt.savefig('reg.png')
    plt.show()


def calc_err(X, beta, y):
    err = np.matmul(X, beta) - y
    return np.matmul(err, err.transpose())


def get_X(x1, x2):
    return np.array([[1]*x1.shape[0], x1, x2]).transpose()


def show_graphs(x1, x2, y, x1_test, x2_test, y_test, beta, errs):
    X = get_X(x1, x2)
    X_test = get_X(x1_test, x2_test)
    show_errs(errs, save=True)

    if dtest:
        y_pred = np.matmul(X_test, beta)
        show_reg(x1_test, x2_test, y_test, y_pred, save=True)
    else:
        y_pred = np.matmul(X, beta)
        show_reg(x1, x2, y, y_pred, save=True)


def gradient_descent_reg(x1, x2, y, x1_test, x2_test, y_test):
    X = get_X(x1, x2)
    X_test = get_X(x1_test, x2_test)

    beta = np.array([1, 1, 1], dtype=np.float64)
    errs = []
    for i in range(NUMBER_OF_ITERATIONS):
        if dtest:
            errs.append(calc_err(X_test, beta, y_test))
        else:
            errs.append(calc_err(X, beta, y))

        gradient = np.matmul(np.matmul(X.transpose(), X), beta) - np.matmul(X.transpose(), y)
        beta -= LEARNING_RATE * gradient/np.linalg.norm(gradient)

    show_graphs(x1, x2, y, x1_test, x2_test, y_test, beta, errs)

    return beta


def stochastic_gradient_descent_reg(x1, x2, y, x1_test, x2_test, y_test):
    X = get_X(x1, x2)
    X_test = get_X(x1_test, x2_test)

    beta = np.array([1, 1, 1], dtype=np.float64)
    errs = []
    for i in range(NUMBER_OF_ITERATIONS):
        if dtest:
            errs.append(calc_err(X_test, beta, y_test))
        else:
            errs.append(calc_err(X, beta, y))

        ind = random.randint(0, y.shape[0]-1)
        d = [0, 0, 0]
        diff = beta[0] + beta[1] * x1[ind] + beta[2] * x2[ind] - y[ind]
        d[0] = 2 * diff
        d[1] = 2 * x1[ind] * diff
        d[2] = 2 * x2[ind] * diff
        d = np.array(d)
        beta -= LEARNING_RATE * d / np.linalg.norm(d)

    show_graphs(x1, x2, y, x1_test, x2_test, y_test, beta, errs)

    return beta


if __name__ == '__main__':
    data = load_train_data()
    choice = input("Stochastic gradient descent(s)/Gradient descent(g)? ")
    test_or_train = input("Draw on test set(test) or train set(train)? ")
    if test_or_train == 'test':
        dtest = True
    else:
        dtest = False
    if choice[0] == 'g':
        gradient_descent_reg(*data)
    else:
        stochastic_gradient_descent_reg(*data)
