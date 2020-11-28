from Perceptron.myPerceptron import  Perceptron
import numpy as np


perceptron = Perceptron()

# Load du lieu
file = "../datasets/perceptron.csv"
X, y = perceptron.loadData(file)

# Hien thi du lieu
perceptron.display_data(X[0], X[1], y)

# Tinh Xbar (d rows, N columns)
Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

# init w
w = perceptron.init_w(Xbar)

# Thuc hien thuat toan Perceptron tinh w( w la ma tran d rows, 1 column )
(w, m) = perceptron.my_perceptron(Xbar, y, w)

# Tim phuong trinh duong phan chia 2 lop
x0, y0 = perceptron.findBound(w)

perceptron.display_result(X[0], X[1], y, x0, y0)
print("Ket qua w sau khi tim duoc: ", w[-1].T)