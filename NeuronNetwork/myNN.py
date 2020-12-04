import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def add_bias(a):
    return np.insert(a, 0, 1, axis=1)


def feedforward(x):
    z = []
    a = [add_bias(x)]
    for i in range(1, nL):
        z_i = np.dot(a[i - 1], w[i - 1].T)
        a_i = sigmoid(z_i)
        if i < nL - 1:
            a_i = add_bias(a_i)
        z.append(z_i)
        a.append(a_i)
    # print('f')
    return z, a


def backprop(x, y):
    w_grad = [np.zeros(_w.shape) for _w in w]
    z, a = feedforward(x)

    # loss = 1.0 * (y^ - y) ** 2 / 2
    d_a = 2 * (a[-1] - y)

    r = None
    for l in range(1, nL):
        i = -l
        d_z = d_sigmoid(z[i])
        if i < -1:
            r = np.dot(r, w[i + 1][:, 1:]) * d_z
        if i == -1:
            r = d_a * d_z
        w_grad[i] = np.dot(a[i - 1].T, r)
        # print('l')

    # print('b')
    return w_grad


def predict(x):
    z, a = feedforward(x)
    return z, a

def train(inputs, outputs, w):
    loop = 10000
    eta = 0.02
    print("W", w)
    # Train
    while loop > 0:
        w_grad = backprop(inputs, outputs)
        w = [W - eta * W_grad.T for W, W_grad in zip(w, w_grad)]
        loop -= 1

    return w

def load_data(file):
    df = pd.read_csv(file)
    N, d = df.values.shape  # N: số điểm dữ liệu, d: số đặc trưng
    X = (df.values[:, 0:d - 1])
    Y = (df.values[:, d - 1:d])
    print(X)
    return X, Y


if __name__ == '__main__':

    # doc du lieu
    file = "../datasets/nn.csv"
    X, Y = load_data(file)
    d = X.shape[1]+1

    # dinh nghia layers
    layers = [d - 1,4, 2]
    nL = len(layers)

    # init weights
    w = [np.random.randn(l2, l1 + 1) for l2, l1 in zip(layers[1:], layers[:-1])]

    # Train du lieu
    W = train(X, Y, w)

    print('Final weights: \n', W)

    Xtest = np.array([[8,0.1]])
    print(Xtest)
    print('Evaluate:')
    z, a = predict(np.array(Xtest))
    print("X_test", Xtest)
    print('Predict:', a)