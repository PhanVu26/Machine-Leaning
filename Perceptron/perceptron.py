from Perceptron.myPerceptron import  Perceptron
import numpy as np


perceptron = Perceptron()

# Load du lieu
file = "../datasets/perceptron.csv"
X, y = perceptron.loadData(file)

# Hien thi du lieu
perceptron.display_data(X[0], X[1])

# Tinh Xbar (d rows, N columns)
Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

# init w
w = perceptron.init_w(Xbar)

# Thuc hien thuat toan Perceptron
(w, m) = perceptron.my_perceptron(Xbar, y, w)
# w( ma tran d rows, 1 column )
#
# x0 = np.linspace(1, 2, 1000)
# x1 = np.linspace(2, 4, 1000)
# y0 = w[-1][0][0] + x0 * w[-1][1][0] + x1 * w[-1][2][0];
# perceptron.display_result(Xbar, y, x0, y0)
print("Ket qua w sau khi tim duoc: ", w[-1].T)