import numpy as np
import csv
import matplotlib.pyplot as plt

class Logistic():

    def readFile(self, file):
        # Doc file
        x0 = []
        x1 = []
        y = []
        line_count = 0
        with open(file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if(line_count == 0):
                    line_count += 1
                else:
                    x0.append(row[0])
                    x1.append(row[1])
                    y.append(row[2])
            X = np.array(np.concatenate(([x0], [x1]), axis=0), dtype=float)
            Y = np.array(y, dtype=int)
        return (X, Y)


    def prod(self, w, X):
        return np.dot(self, w.T, X)


    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))


    def train(self, X, Y, w_init, eta, epsilon=1e-3, M=10000):
        w = [w_init]
        N = X.shape[1]
        d = X.shape[0]
        count = 0
        check_w_after = 20
        while count < M:
            # mix data
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = X[:, i].reshape(d, 1)
                yi = Y[i]
                zi = self.sigmoid(np.dot(w[-1].T, xi))
                w_new = w[-1] + eta * (yi - zi) * xi
                count += 1
                # stopping criteria
                if count % check_w_after == 0:
                    if np.linalg.norm(w_new - w[-check_w_after]) < epsilon:
                        return w
                w.append(w_new)
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


    def predict(self, X_test, w):
        thredsold = 0.5
        accuracy  = self.sigmoid(np.dot(w[-1].T, X_test))
        if (accuracy > thredsold):
            pred = 1
        else:
            pred = 0
        return pred, accuracy[0][0]