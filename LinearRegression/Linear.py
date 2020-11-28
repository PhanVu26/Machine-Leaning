import numpy as np
import csv
import matplotlib.pyplot as plt

class LinearRegression():

    def display_data(self, X, y):
        plt.plot(X, y, 'ro')
        plt.axis([0, 11, 30000, 122000])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def display_result(self,X, y, x0, y0):
        # Drawing the fitting line
        plt.plot(X.T, y.T, 'ro')  # data
        plt.plot(x0, y0)  # the fitting line
        plt.axis([0, 11, 30000, 122000])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


    def linearRegress(self, XBar, Y):
       a = np.dot(XBar.T, XBar)
       b = np.dot(XBar.T, Y)
       B = np.dot(np.linalg.pinv(a), b)
       return B;

    def readFile(self, datasetFile ):
       x = []
       y = []
       line_count = 0
       with open(datasetFile, 'r') as file:
           reader = csv.reader(file)
           for row in reader:
               print(row[0])
               if(line_count == 0):
                   line_count +=1
               else:
                x.append(row[0])
                y.append(row[1])

       X = np.array([x], dtype=float).T
       Y = np.array([y], dtype=int).T

       return (X,Y)