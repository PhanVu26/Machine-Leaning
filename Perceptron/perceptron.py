from Perceptron.myPerceptron import  Perceptron
import numpy as np


perceptron = Perceptron()

# Load du lieu
Xbar, X, y = perceptron.loadData()

# init w
w = perceptron.init_w(Xbar)

# Thuc hien thuat toan Perceptron
(w, m) = perceptron.my_perceptron(Xbar, y, w)

print("Ket qua w sau khi tim duoc: ", w[-1].T)