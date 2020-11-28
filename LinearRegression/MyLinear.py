from LinearRegression.Linear import  LinearRegression
import numpy as np

linearRegress = LinearRegression()

# Doc Du lieu
datasetFile = '../datasets/salary_data.csv';
X, Y = linearRegress.readFile(datasetFile)

# hien thi du lieu vao
linearRegress.display_data(X, Y)

# Tinh Xbar
one = np.ones((X.shape[0], 1), dtype=float)
XBar = np.concatenate((one, X), axis=1)

#Xbar (Ma tran N rows, d column)
# Y (ma tran N rows)

# Tinh w(ma tran d rows)
W = linearRegress.linearRegress(XBar, Y)
w_0 = W[0][0]
w_1 = W[1][0]
x0 = np.linspace(0, 10, 2)
y0 = w_0 + w_1*x0

linearRegress.display_result(X, Y, x0, y0)

# Du doan
x_test = [150, 162, 176, 168, 155]
y_test = []
for x in x_test:
    y_test.append(w_0 + w_1 * x)
print("X test: ", x_test)
print("Y_test: ", y_test)




