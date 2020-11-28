import numpy as np
import matplotlib.pyplot as plt
import csv

class Perceptron():

    def display_data(self, X, Y):
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

    def loadData(self, file):
        # Doc file
        x0 = []
        x1 = []
        y = []
        with open(file, 'r') as file:
            reader = csv.reader(file)
            line_count = 0;
            for row in reader:
                if(line_count == 0):
                    line_count += 1
                else:
                    x0.append(row[0])
                    x1.append(row[1])
                    y.append(row[2])
            X = np.array(np.concatenate(([x0], [x1]), axis=0), dtype=float)
            Y = np.array([y], dtype=int)
        return X, Y

    def my_perceptron(sefl, X, y, w_init):
        w = [w_init]
        N = X.shape[1]
        d = X.shape[0]
        mis_points = []
        while True:
            # mix data
            mix_id = np.random.permutation(N)
            #print("mix_id", mix_id)
            for i in range(N):
                xi = X[:, mix_id[i]].reshape(d, 1)
                #print("xi", xi)
                yi = y[0, mix_id[i]]
                #print("yi", yi)
                #print("KQ", sefl.pred(w[-1], xi))
                if sefl.pred(w[-1], xi)[0] != yi:
                    mis_points.append(mix_id[i])
                    w_new = w[-1] + yi * xi  # Cap nhat lai w cho tap cac diem bi miss
                    #print("w_new", w_new)
                    w.append(w_new)
            if sefl.has_converged(X, y, w[-1]):
                break
        return (w, mis_points)