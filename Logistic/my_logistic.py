import numpy as np
import csv
import matplotlib.pyplot as plt

class Logistic():

    def readFile(self, file):
        # Doc file
        x = []
        y = []
        with open(file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x.append(row[0])
                y.append(row[1])
            X = np.array([x], dtype=float)
            Y = np.array(y, dtype=int)
        return (X, Y)


    def prod(self, w, X):
        return np.dot(self, w.T, X)


    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))


    def my_logistic_sigmoid_regression(self, X, Y, w_init, eta, epsilon=1e-3, M=10000):
        w = [w_init]
        print("W", w)
        print("w[-1]", w[-1])
        N = X.shape[1]
        d = X.shape[0]
        count = 0
        check_w_after = 20
        while count < M:
            # mix data
            mix_id = np.random.permutation(N)
            # print("Mix_", mix_id)
            for i in mix_id:
                xi = X[:, i].reshape(d, 1)
                # print("Xi", xi);
                yi = Y[i]
                # print("yi", yi)
                zi = self.sigmoid(np.dot(w[-1].T, xi))
                # print("zi", zi)
                w_new = w[-1] + eta * (yi - zi) * xi
                # print("w-new", w_new)
                count += 1
                # stopping criteria
                if count % check_w_after == 0:
                    # print("====20====")
                    if np.linalg.norm(w_new - w[-check_w_after]) < epsilon:
                        return w
                w.append(w_new)
                # print("W after append", w)
        return w

    def display(self, w, X, y):
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
        yy = self.sigmoid(w0 + w1 * xx)
        plt.axis([-2, 8, -1, 2])
        plt.plot(xx, yy, 'g-', linewidth=2)
        plt.plot(threshold, .5, 'y^', markersize=8)
        plt.xlabel('studying hours')
        plt.ylabel('predicted probability of pass')
        plt.show()

