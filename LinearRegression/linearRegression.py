# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(2)

# Doc Du lieu
datasetFile = './datasets/salary_data.csv';

def readFile(datasetFile):
    x = []
    y = []
    line_count = 0
    with open(datasetFile, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if (line_count == 0):
                line_count += 1
            else:
                x.append(row[0])
                y.append(row[1])

    X = np.array([x], dtype=float).T
    Y = np.array([y], dtype=int).T

    return (X, Y)

X, y = readFile(datasetFile)
# Tinh Xbar
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

# Tinh dao ham
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

# Tinh gia tri ham so
def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2;

# Kiem tra do chinh xac dao ham
def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
    return g

def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))

# Train model
def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(500):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it)

# init w
w_init = np.array([[2], [1]])

(w1, it1) = myGD(w_init, grad, 0.05)
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))
