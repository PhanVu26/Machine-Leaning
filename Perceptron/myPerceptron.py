import numpy as np
import matplotlib.pyplot as plt

class Perceptron():

    def display_data(self, X, Y, y):
        plt.plot(X, Y, 'ro')
        plt.axis([0, 5, 0, 6])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def display_result(self, X, y, x0, y0):
        # Drawing the fitting line
        plt.plot(X.T, y.T, 'ro')  # data
        plt.plot(x0, y0)  # the fitting line
        plt.axis([0, 6, 0, 6])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def pred(sefl, w, x):
        return np.sign(np.dot(w.T, x))

    def has_converged(sefl, X, y, w):
        return np.array_equal(sefl.pred(w, X), y)

    def init_w(self, Xbar):
        return  np.random.randn(Xbar.shape[0], 1)

    def loadData(self):
        means = [[2, 2], [4, 2]]
        cov = [[.3, .2], [.2, .3]]
        N = 10
        X0 = np.random.multivariate_normal(means[0], cov, N).T
        print(X0)
        X1 = np.random.multivariate_normal(means[1], cov, N).T
        X = np.concatenate((X0, X1), axis=1)
        y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)
        self.display_data(X[0], X[1], y)
        Xbar = np.concatenate((np.ones((1, 2 * N)), X), axis=0)
        return Xbar, X, y

    def my_perceptron(sefl, X, y, w_init):
        w = [w_init]
        N = X.shape[1]
        d = X.shape[0]
        mis_points = []
        while True:
            # mix data
            mix_id = np.random.permutation(N)
            print("mix_id", mix_id)
            for i in range(N):
                xi = X[:, mix_id[i]].reshape(d, 1)
                print("xi", xi)
                yi = y[0, mix_id[i]]
                print("yi", yi)
                print("KQ", sefl.pred(w[-1], xi))
                if sefl.pred(w[-1], xi)[0] != yi:
                    mis_points.append(mix_id[i])
                    w_new = w[-1] + yi * xi  # Cap nhat lai w cho tap cac diem bi miss
                    print("w_new", w_new)
                    w.append(w_new)
            if sefl.has_converged(X, y, w[-1]):
                break
        return (w, mis_points)