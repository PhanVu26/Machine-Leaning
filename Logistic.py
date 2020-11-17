import numpy as np
import matplotlib.pyplot as plt


def prod(w,X):
    return np.dot(w.T, X)


def sigmoid(s):
    return 1/(1 + np.exp(-s))


def my_logistic_sigmoid_regression(X, y, w_init, eta, epsilon = 1e-3, M = 10000):
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
            yi = y[i]
            print("yi", yi)
            zi = sigmoid(np.dot(w[-1].T, xi))
            print("zi", zi)
            w_new = w[-1] + eta*(yi - zi)*xi
            print("w-new", w_new)
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < epsilon:
                    return w
            w.append(w_new)
    return w


if __name__ == '__main__':
    print("ok")
    # data:
    X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                   2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    # extended data
    #print("X: ",X)
    Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    print("Xbar : ", Xbar)
    epsilon = .05
    d = Xbar.shape[0]
    print("d",d)
    w_init = np.random.randn(d, 1)
    print("W_init", w_init)
    w = my_logistic_sigmoid_regression(Xbar, y, w_init, epsilon)
    print("RS",w[-1].T)
    print("RS2" ,sigmoid(np.dot(w[-1].T,Xbar)))