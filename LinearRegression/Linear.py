import numpy as np
import csv
import matplotlib.pyplot as plt

class LinearRegression():
    def display_data(self, X, y):
        plt.plot(X, y, 'ro')
        plt.axis([140, 190, 45, 75])
        plt.xlabel('Height (cm)')
        plt.ylabel('Weight (kg)')
        plt.show()

    def display_result(self,X, y, x0, y0):
        # Drawing the fitting line
        plt.plot(X.T, y.T, 'ro')  # data
        plt.plot(x0, y0)  # the fitting line
        plt.axis([140, 190, 45, 75])
        plt.xlabel('Height (cm)')
        plt.ylabel('Weight (kg)')
        plt.show()


    def linearRegress(self, XBar, Y):
       a = np.dot(XBar.T, XBar)
       b = np.dot(XBar.T, Y)
       B = np.dot(np.linalg.pinv(a), b)
       return B;

    def readFile(self ):
       x = []
       y = []
       with open('../datasets/LinearRegressionDataset.csv', 'r') as file:
           reader = csv.reader(file)
           for row in reader:
               x.append(row[0])
               y.append(row[1])

       X = np.array([x], dtype=int).T
       Y = np.array([y], dtype=int).T

       return (X,Y)