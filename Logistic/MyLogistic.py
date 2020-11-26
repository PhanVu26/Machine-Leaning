from Logistic.my_logistic import Logistic
import numpy as np

logistic = Logistic()

X, Y = logistic.readFile()

eta = .05
d = X.shape[0]
print("d", d)
w_init = np.random.randn(d, 1)
# print("W_init", w_init)
w = logistic.my_logistic_sigmoid_regression(X, Y, w_init, eta)

print("w sau khi tim duoc", w[-1].T)
print("Ket qua du doan", logistic.sigmoid(np.dot(w[-1].T, X)))
logistic.display(w, X, Y)