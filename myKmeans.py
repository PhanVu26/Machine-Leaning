import numpy as np
import matplotlib.pyplot as plt

def plot_result(X, y, centers, k, title):
    for i in range(K):
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



def init_centers(X, K):
    return X[np.random.choice(X.shape[0], K, replace=False)]


def group_data(X, centers):
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        d = X[i] - centers
        d = np.linalg.norm(d, axis=1)
        y[i] = np.argmin(d)

    return y
def update_centers(X, y, K):
    centers = np.zeros((K, X.shape[1]))

    for i in range(K):
        X_i = X[y == i, :]
        centers[i] = np.mean(X_i, axis=0)

    return centers

def kmeans(X, K):
    centers = init_centers(X, K)
    y = []
    iter = 0;
    while True:
        y_old = y
        y =  group_data(X, centers)
        if(np.array_equal(y, y_old)):
            break;
        centers = update_centers(X, y, K)
        iter += 1
        plot_result(X, y, centers, K, 'iter: ' + str(iter))
    return (centers, y)
if __name__ == '__main__':
    K = 6
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=1500,
                      n_features=2,
                      centers=K,
                      cluster_std=0.5,
                      shuffle=True,
                      random_state=0)
    print("X shape", X.shape)
    # show data
    plt.scatter(X[:, 0], X[:, 1], c='red', marker='o', s=10)
    plt.grid()
    plt.show()
    # run k-means
    print("X[1]", X[1])
    centers, y = kmeans(X, K)
    # plot result
    plot_result(X, y, centers, K, 'Final')

