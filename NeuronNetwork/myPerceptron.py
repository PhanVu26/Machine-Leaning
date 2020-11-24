import numpy as np


def pred(X, w):
    return np.sign(np.dot(w.T, X))


def hasConverged(X, w, y):
    return np.array_equal(pred(X, w), y)


def my_perceptron(X, y, w_init):
    w = [w_init]
    miss_points = []
    N = X.shape[1]
    d = X.shape[0]
    while True:
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if(pred(xi, w[-1]) != yi):
                miss_points.append(mix_id[i])
                w_new = w[-1] + xi * yi
                w.append(w_new)
        if(hasConverged(X, w[-1], y)) :
            break;
    return (w, miss_points);
if __name__ == '__main__':
    means = [[2, 2], [4, 2]]
    cov = [[.3, .2], [.2, .3]]
    N = 10
    X0 = np.random.multivariate_normal(means[0], cov, N).T
    print("X0", X0)
    X1 = np.random.multivariate_normal(means[1], cov, N).T
    print("X1", X1)
    X = np.concatenate((X0, X1), axis=1)
    print("X", X)
    print(X.shape)
    y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)
    print("y", y)
    # Xbar
    Xbar = np.concatenate((np.ones((1, 2 * N)), X), axis=0)
    print("XBar", Xbar)
    d = Xbar.shape[0]
    w_init = np.random.randn(d, 1)
    print("w_init", w_init)
    (w, m) = my_perceptron(Xbar, y, w_init)
    print("wwwww", w[-1].T)
    # print(len(w[-1]))
