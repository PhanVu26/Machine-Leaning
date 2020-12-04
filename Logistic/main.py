from Logistic.my_logistic import Logistic
import numpy as np

logistic = Logistic()

# d: so dac trung, N: so diem du lieu
# X_train(ma tran d x N), Y_train(array(N))

# Doc Du lieu
file = "./exam.csv"
X_train, Y_train = logistic.readFile(file)

# Tinh Xbar
Xbar = np.concatenate((np.ones((1, X_train.shape[1])), X_train), axis=0)

# learning rate
eta = .005

# Init w
w_init = np.random.randn(Xbar.shape[0], 1)

# Train du lieu, tim ma tran w
w = logistic.train(Xbar, Y_train, w_init, eta)

print("w sau khi tim duoc", w[-1].T)
print("Ket qua du doan tap train", logistic.sigmoid(np.dot(w[-1].T, Xbar)))

# Test du lieu moi
Xtest = np.array([[1], [7], [0.5]])
pred, accuracy = logistic.predict(Xtest, w)
print("Ket qua du doan thuoc lop: ", pred)
print("Do chinh xac accuracy = ", accuracy)

