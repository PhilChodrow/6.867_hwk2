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
    # StepSizes = range(-10, -5)
    # Errors = range(-10, -5)
    # for data in range(4, 5):
    #     for Lambda in range(1, 2):
    #         train = pl.loadtxt('hw2_resources\data\data' + str(data) + '_train.csv')
    #         X_ = train[:, 0:2]
    #         X = np.concatenate((np.ones((len(X_), 1)), np.array(X_)), axis=1)
    #         Y_ = train[:, 2:3]
    #         Y = (Y_ + 1) / 2
    #         StepSize = 0.01
    #         Error = 0.01
    #         GradPath, Opt, GradNorm = GradientDescent.GradientDescent(Functions.RegularizedNLL,
    #                                                                   Functions.GradientRegularizedNLL,
    #                                                                   [0, 0,0], StepSize, Error,
    #                                                                   0, 0, 10000000, X, Y,Lambda)
    #         h = .02
    #         x_min, x_max = X_[:, 0].min() - .5, X_[:, 0].max() + .5
    #         y_min, y_max = X_[:, 1].min() - .5, X_[:, 1].max() + .5
    #         xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #         mashMat=np.concatenate((np.ones((len(np.c_[xx.ravel(), yy.ravel()]), 1)), np.c_[xx.ravel(), yy.ravel()]),axis=1)
    #         Z=np.zeros((len(mashMat)))
    #         for x in range(0,len(mashMat)):
    #             Z[x] = Functions.predict(GradPath[-1],mashMat[x])
    #         Z = Z.reshape(xx.shape)
    #         plt.figure(1)
    #         plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    #         plt.scatter(X_[:, 0], X_[:, 1], c=Y_, s=40)
    #         plt.xlabel('X1')
    #         plt.ylabel('X2')
    #         plt.xlim(xx.min(), xx.max())
    #         plt.ylim(yy.min(), yy.max())
    #         plt.xticks(())
    #         plt.yticks(())
    #         plt.savefig('GradientDescent separator lambda ='+str(Lambda)+' - Data Set'+str(data)+'.png')
    #
    #
    #     ConvergenceRate = np.zeros((len(Errors), len(StepSizes)))
    #     for Error in range(0, len(Errors)):
    #         for StepSize in range(0, len(StepSizes)):
    #             print Error, ",", StepSize, ",", time.time()
    #             GradPath, Opt, GradNorm = GradientDescent.GradientDescent(Functions.RegularizedNLL,
    #                                                                       Functions.GradientRegularizedNLL,
    #                                                                       [0, 0, 0], math.exp(StepSizes[StepSize]),
    #                                                                       math.exp(Errors[Error]),
    #                                                                       0, 0, 100000, X, Y, 1)
    #
    #             ConvergenceRate[Error, StepSize] = len(GradNorm)
    #     plt.figure(figsize=(15, 10))
    #     plt.subplot()
    #     plt.xlabel('ln(Precision Level)')
    #     plt.ylabel('ln(Learning Rate)')
    #     plt.xticks(range(0, len(StepSizes)), StepSizes)
    #     plt.yticks(range(0, len(Errors)), Errors)
    #     plt.pcolor(ConvergenceRate)
    #     plt.colorbar()
    #     plt.savefig('Convergence as a Function of the Precision Level and the Learning Rate - Data Set'+str(data)+'.png')

    # # PART 2:
    # # =======
    # for data in range(1, 5):
    #     for Lambda in range(0, 2):
    #         train = pl.loadtxt('hw2_resources\data\data' + str(data) + '_train.csv')
    # train = pl.loadtxt('hw2_resources\data\data1_train.csv')
    # X = train[:, 0:2]
    # Y = train[:, 2:3]
    # h = .02
    # logreg = linear_model.LogisticRegression(penalty='l2',C=1)
    # logreg.fit(X, np.ravel(Y))
    # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # plt.figure(2)
    # plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    # plt.scatter(X[:, 0], X[:, 1], c=Y,s=40)
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()
    #
    # # PART 3:
    # # =======
    train = pl.loadtxt('hw2_resources\data\data1_train.csv')
    XTrain = train[:, 0:2]
    YTrain = train[:, 2:3]
    validate = pl.loadtxt('hw2_resources\data\data1_validate.csv')
    XVal = validate[:, 0:2]
    YVal = validate[:, 2:3]

    Lambdas=np.arange(0.001,5,0.05)
    InterLASSO,WLASSO, LambdaLASSO,RSQ=ValidateParams.Validate(XTrain, YTrain, XVal, YVal,Lambdas,
                                                                   RegularizedFunctions.LASSO)
    print "LASSO:"
    print RSQ
    print LambdaLASSO
    h = .02
    logreg = linear_model.LogisticRegression(penalty='l1',C=LambdaLASSO)
    logreg.fit(XTrain, np.ravel(YTrain))
    x_min, x_max = XTrain[:, 0].min() - .5, XTrain[:, 0].max() + .5
    y_min, y_max = XTrain[:, 1].min() - .5, XTrain[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(2)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(XTrain[:, 0], XTrain[:, 1], c=YTrain,s=40)
    plt.scatter(XVal[:, 0], XVal[:, 1], c=YVal, s=50, marker="*")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.savefig('Optimal LASSO eparator - Data Set1.png')

    InterRIDGE,WRIDGE, LambdaRIDGE,RSQ = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas,
                                                                     RegularizedFunctions.Ridge)
    print "RIDGE:"
    print RSQ
    print LambdaRIDGE
    h = .02
    logreg = linear_model.LogisticRegression(penalty='l2',C=LambdaRIDGE)
    logreg.fit(XTrain, np.ravel(YTrain))
    x_min, x_max = XTrain[:, 0].min() - .5, XTrain[:, 0].max() + .5
    y_min, y_max = XTrain[:, 1].min() - .5, XTrain[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(2)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(XVal[:, 0], XVal[:, 1], c=YVal,s=50,marker="*")
    plt.scatter(XTrain[:, 0], XTrain[:, 1], c=YTrain,s=40)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.savefig('Optimal Ridge eparator - Data Set1.png')

