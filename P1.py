import math
import numpy as np
import matplotlib
import time
import random
import matplotlib.pyplot as plt
#from plotBoundary import *
import sys
sys.path.append('Lib')
import Functions
import GradientDescent
import RegularizedFunctions
import ValidateParams
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 18})
    # PART 1:
    # =======
    ##Plot the classification
    Lambda=0
    train = pl.loadtxt('hw2_resources\data\data1_train.csv')
    X_ = train[:, 0:2]
    X = np.concatenate((np.ones((len(X_), 1)), np.array(X_)), axis=1)
    Y_ = train[:, 2:3]
    Y = (Y_ + 1) / 2
    StepSize = 0.01
    Error = 0.001
    GradPath, Opt, GradNorm = GradientDescent.GradientDescent(Functions.RegularizedNLL,
                                                                             Functions.GradientRegularizedNLL,
                                                                             [0, 0,0], StepSize, Error,
                                                                             0, 0, 100000, X, Y,Lambda)
    h = .02  # step size in the mesh


    x_min, x_max = X_[:, 0].min() - .5, X_[:, 0].max() + .5
    y_min, y_max = X_[:, 1].min() - .5, X_[:, 1].max() + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mashMat=np.concatenate((np.ones((len(np.c_[xx.ravel(), yy.ravel()]), 1)), np.c_[xx.ravel(), yy.ravel()]),axis=1)
    Z=np.zeros((len(mashMat)))
    for x in range(0,len(mashMat)):
        Z[x] = Functions.predict(GradPath[-1],mashMat[x])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X_[:, 0], X_[:, 1], c=Y_, s=40)

    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

    # # print "Gaussian - convergence of the gradient"
    # xes = range(0, len(GradNormGauss))
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.plot(xes, GradNormGauss)
    # plt.xlabel('Iteration')
    # plt.ylabel('Second Norm of the Gradient')
    # plt.title('Gradient Norm Convergence rate - Gaussian')
    # plt.savefig('Gradient Norm Convergence rate - Gaussian.png')
    # # print "Gaussian - convergence - different points"
    # ConvergenceRate = np.zeros((10, 10))
    # for x in range(0, 10):
    #     for y in range(0, 10):
    #         GaussGradPath, GaussOpt, GradNorm = GradientDescent.GradientDescent(Functions.GaussianFunc,
    #                                                                             Functions.GradientGaussianFunc,
    #                                                                             [x, y], StepSize,
    #                                                                             Error, 0, 0, 100000, gaussMean,
    #                                                                             gaussCov)
    #         ConvergenceRate[x, y] = len(GradNorm)
    # plt.figure(figsize=(15, 10))
    # plt.subplot()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Convergence for Different Starting Points - Gaussian')
    # plt.pcolor(ConvergenceRate)
    # plt.axis([0, 10, 0, 10])
    # plt.colorbar()
    # plt.savefig('Convergence for Different Starting Points - Gaussian.png')

    # PART 2:
    # =======
    train = pl.loadtxt('hw2_resources\data\data1_train.csv')
    X = train[:, 0:2]
    Y = train[:, 2:3]
    h = .02
    logreg = linear_model.LogisticRegression(penalty='l2',C=1)
    logreg.fit(X, np.ravel(Y))
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(2)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=Y,s=40)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()

    # PART 3:
    # =======
    train = pl.loadtxt('hw2_resources\data\data1_train.csv')
    XTrain = train[:, 0:2]
    YTrain = train[:, 2:3]
    validate = pl.loadtxt('hw2_resources\data\data1_train.csv')
    XVal = validate[:, 0:2]
    YVal = validate[:, 2:3]


    Lambdas=np.arange(0.001,5,0.05)
    InterLASSO,WLASSO, LambdaLASSO=ValidateParams.Validate(XTrain, YTrain, XVal, YVal,Lambdas,
                                                                   RegularizedFunctions.LASSO)
    InterRIDGE,WRIDGE, LambdaRIDGE = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas,
                                                                     RegularizedFunctions.Ridge)

