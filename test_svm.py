import scipy
import numpy as np
import pandas as pd
from learner import learner
from hw2_resources.plotBoundary import *

# basic training data
# X = np.array([[2,2],[2,3],[0,-1], [-3, -2]])
# Y = np.array([1, 1, -1, -1]).reshape((4,1))

data = pd.read_table('hw2_resources/data/data4_train.csv', sep = ' ', header = None)
# data = data[1:100]

X = np.array(data[[0,1]])
Y = np.array(data[[2]])

svm = learner()
svm.set_data(X,Y)

def gaussian_kernel(x, y, s = 1):
    return scipy.exp(-(np.linalg.norm(x - y))**2 / (s**2))

def linear_kernel(x,y):
    return np.dot(x,y)

svm.set_kernel_function(gaussian_kernel)
svm.make_kernel_matrix()
np.linalg.matrix_rank(svm.K)

svm.train(1)
plotDecisionBoundary(X, Y, scoreFn = svm.predict, values = (0))
svm.training_error()