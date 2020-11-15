import csv
import numpy as np

if __name__ == '__main__':
    x = []
    y = []
    with open('LinearRegressionDataset.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x.append(row[0])
            y.append(row[1])

    X = np.array([x], dtype=int).T
    Y = np.array([y], dtype=int).T
    d = X.shape[1]
    print("So dac trung: ", d)

    one = np.ones((X.shape[0], 1), dtype=int)

    XBar = np.concatenate((one, X), axis=1)

    a = np.dot(XBar.T, XBar)
    b = np.dot(XBar.T, Y)
    B = np.dot(np.linalg.pinv(a), b)

    print("B: ", B)
    x_test = [150, 162, 176, 168, 155]
    y_test = []
    for x in x_test:
        y_test.append(B[0][0] + B[1][0]*x)
    print("X test: ", x_test)
    print("Y_test: ", y_test)


