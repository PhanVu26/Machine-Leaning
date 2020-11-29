import numpy as np
import matplotlib.pyplot as plt
import csv

class K_means():

    def __init__(self, K):
        self.K = K

    def load_data(self, K, file):
        # # group size
        # # create random data
        # from sklearn.datasets import make_blobs
        # X, _ = make_blobs(n_samples=500,
        #                   n_features=2,
        #                   centers=K,
        #                   cluster_std=0.5,
        #                   shuffle=True,
        #                   random_state=0)
        # Doc file
        x0 = []
        x1 = []
        with open(file, 'r') as file:
            reader = csv.reader(file)
            line_count = 0;
            for row in reader:
                if(line_count == 0):
                    line_count += 1;
                else:
                    x0.append(row[0])
                    x1.append(row[1])
            X0 = np.array([x0], dtype=float).T
            X1 = np.array([x1], dtype=float).T
            X = np.array(np.concatenate((X0, X1), axis=1), dtype=float)
        return X

    def show_data(self, X):
        # show data
        plt.scatter(X[:, 0], X[:, 1], c='red', marker='o', s=10)
        plt.grid()
        plt.show()

    def plot_result(sefl, X, y, centers, k, title):
        for i in range(k):
            plt.scatter(X[y == i, 0],
                        X[y == i, 1],
                        s=50,
                        label='cluster ' + str(i + 1))
        plt.scatter(centers[:, 0],
                    centers[:, 1],
                    s=250,
                    marker='*',
                    c='red',
                    label='centroids')
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()

    # 1. init center points
    def init_centers(self, X, k):
        return X[np.random.choice(X.shape[0], k, replace=False)]

    # 2.  grouping
    def group_data(self, X, centers):
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            d = X[i] - centers
            # print("d", d);
            d = np.linalg.norm(d, axis=1)
            y[i] = np.argmin(d)

        return y

    # 3. Update center points
    def update_centers(self, X, y, k):
        centers = np.zeros((k, X.shape[1]))

        for i in range(k):
            X_i = X[y == i, :]
            centers[i] = np.mean(X_i, axis=0)
        return centers

    # kmeans algorithm
    def kmeans(self, X, k):
        centers = self.init_centers(X, k)
        print(centers)
        y = []
        iter = 0
        while True:
            # save pre-loop groups
            y_old = y
            # grouping
            y = self.group_data(X, centers)
            # break while loop if groups are not changed
            if np.array_equal(y, y_old):
                break
            #  update centers
            centers = self.update_centers(X, y, k)
            # plot current state
            iter += 1
            # plot_result(X, y, centers, k, 'iter: ' + str(iter))
        return (centers, y)

