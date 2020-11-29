from Logistic.my_logistic import Logistic
import numpy as np

logistic = Logistic()

file = "../datasets/logisticDataset.csv"

# Load Du lieu
X, Y = logistic.readFile(file)

# Tinh Xbar
Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

# learning rate
eta = .05

# Init w
w_init = np.random.randn(Xbar.shape[0], 1)

# Tinh w
w = logistic.my_logistic_sigmoid_regression(Xbar, Y, w_init, eta)

print("w sau khi tim duoc", w[-1].T)
print("Ket qua du doan", logistic.sigmoid(np.dot(w[-1].T, Xbar)))
logistic.display(w, Xbar, Y)