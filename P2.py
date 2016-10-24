
# coding: utf-8

# In[1]:

import scipy
import numpy as np
import pandas as pd
from learner import learner
from hw2_resources.plotBoundary import *
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

# kernels we're using
def gaussian_kernel(x, y, s = 1):
    return scipy.exp(-(np.linalg.norm(x - y))**2 / (s**2))

def linear_kernel(x,y):
    return np.dot(x,y)


# # Toy Problem

# In[3]:

X = np.array([[2,2],[2,3],[0,-1], [-3, -2]])
Y = np.array([1, 1, -1, -1]).reshape((4,1))

svm = learner()
svm.set_data(X,Y)
svm.set_kernel_function(linear_kernel)
svm.make_kernel_matrix()
constraints = svm.train(1)
plotDecisionBoundary(X, Y, scoreFn = svm.predict, values = (-0.5, 0, 0.5))
plt.gca().scatter(svm.support_vectors[:,0],
            svm.support_vectors[:,1], 
            s = 200,
            c = 'red',
            alpha = .2) # ugly but whatevs


# In[4]:

for constraint in constraints:
    print constraint, constraints[constraint]


# # Linear Kernel for Each Data Set

# In[5]:

def data_path(i, type):
    return 'hw2_resources/data/data' + str(i) + '_' + type + '.csv'

def plot_data(i, X, Y, values, ax, title = None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/20., (y_max-y_min)/20.)
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))
    zz = array([svm.predict(x) for x in c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)

    CS = ax.contour(xx, yy, zz, values, colors = 'black', linestyles = 'dashed', linewidths = 2)
    ax.clabel(CS, fontsize=9, inline=1)
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
    ax.set_title(title)
#     ax.axis('tight')

def read_data(path):
    data = pd.read_table(path, sep = ' ', header = None)
    X = np.array(data[[0,1]])
    Y = np.array(data[[2]])
    return((X,Y))


# In[ ]:

fig = plt.figure()
C = 1

for i in [1, 2, 3, 4]:

    X,Y = read_data(data_path(i, 'train'))

    svm = learner()
    svm.set_data(X,Y)
    svm.set_kernel_function(linear_kernel)
    svm.make_kernel_matrix()
    svm.train(C)
    
    X_val, Y_val = read_data(data_path(i, 'validate'))
    
    training_error = svm.training_error()
    validation_error = svm.test_error(X_val, Y_val)
    
    ax = fig.add_subplot(2,2,i)
    plot_data(i, X, Y, (0), ax, 'Train:' + str(training_error) + ' Test: ' + str(validation_error))
plt.tight_layout()


# # Gaussian Kernel: Messing with $C$

# In[ ]:

table = pd.DataFrame({'C' : [],
                      'Training Error': [],
                      'Validation Error': [],
                      'Margin': [],
                      'Support Vectors': []})

Cs = [10 ** p for p in range(-2, 3)]
fig = plt.figure(figsize = (10, 3))

i = 4
for j in range(len(Cs)):
    X,Y = read_data(data_path(i, 'train'))
    svm = learner()
    svm.set_data(X,Y)
    svm.set_kernel_function(gaussian_kernel)
    svm.make_kernel_matrix()
    svm.train(Cs[j])
    
    X_val, Y_val = read_data(data_path(i, 'validate'))
    
    training_error = svm.training_error()
    validation_error = svm.test_error(X_val, Y_val)
    margin = svm.get_margin()
    n_supports = len(svm.get_supports())
    
    ax = fig.add_subplot(1,len(Cs),j+1)
    plot_data(j+1, X, Y, (0), ax, r'$C = $' + str(Cs[j]))
    
    add_to_table = pd.DataFrame({'C' : [Cs[j]],
                      'Training Error': [training_error],
                      'Validation Error': [validation_error],
                                'Margin' : margin,
                                'Support Vectors' : n_supports})
    table = table.append(add_to_table)
plt.tight_layout()


# In[ ]:

table # summarize the results

