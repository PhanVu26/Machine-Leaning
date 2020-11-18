import numpy as np
import csv
import matplotlib.pyplot as plt


def prod(w,X):
    return np.dot(w.T, X)


def sigmoid(s):
    return 1/(1 + np.exp(-s))


def my_logistic_sigmoid_regression(X, Y, w_init, eta, epsilon = 1e-3, M = 10000):
    w = [w_init]
    print("W",w)
    print("w[-1]",w[-1])
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < M:
        # mix data
        mix_id = np.random.permutation(N)
        print("Mix_", mix_id)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            print("Xi", xi);
            yi = Y[i]
            print("yi", yi)
            zi = sigmoid(np.dot(w[-1].T, xi))
            print("zi", zi)
            w_new = w[-1] + eta*(yi - zi)*xi
            print("w-new", w_new)
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                print("====20====")
                if np.linalg.norm(w_new - w[-check_w_after]) < epsilon:
                    return w
            w.append(w_new)
            print("W after append", w)
    return w


def display(w, X, y):
    X0 = X[1, np.where(y == 0)][0]
    y0 = y[np.where(y == 0)]
    X1 = X[1, np.where(y == 1)][0]
    y1 = y[np.where(y == 1)]
    plt.plot(X0, y0, 'ro', markersize=8)
    plt.plot(X1, y1, 'bs', markersize=8)
    xx = np.linspace(0, 6, 1000)
    w0 = w[-1][0][0]
    w1 = w[-1][1][0]
    threshold = -w0 / w1
    yy = sigmoid(w0 + w1 * xx)
    plt.axis([-2, 8, -1, 2])
    plt.plot(xx, yy, 'g-', linewidth=2)
    plt.plot(threshold, .5, 'y^', markersize=8)
    plt.xlabel('studying hours')
    plt.ylabel('predicted probability of pass')
    plt.show()


if __name__ == '__main__':

    # Doc file
    x = []
    y = []
    with open("datasets/logisticDataset.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x.append(row[0])
            y.append(row[1])
        X = np.array([x], dtype=float)
        Y = np.array(y, dtype=int)
    print("X", X)
    print("Y", Y)
    # data:
    X1 = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                  2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
    y1 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    # extended data
    print("X1: ",X1)
    print("y1: ", y1)
    Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    print("Xbar : ", Xbar)
    epsilon = .05
    d = Xbar.shape[0]
    print("d",d)
    w_init = np.random.randn(d, 1)
    print("W_init", w_init)
    w = my_logistic_sigmoid_regression(Xbar, Y, w_init, epsilon)
    print("w sau khi tim duoc",w[-1].T)
    print("Ket qua du doan" ,sigmoid(np.dot(w[-1].T,Xbar)))
    display(w, Xbar, Y)