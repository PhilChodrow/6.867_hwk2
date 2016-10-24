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
from learner import learner
#import P2

def GetData(FilesSet1,FilesSet2):
    Xes1=np.empty((0,784), int)
    for f in FilesSet1:
        Xes1 = np.append(Xes1,np.array(pl.loadtxt(f)),axis=0)
    Xes2 = np.empty((0,784), int)
    for f in FilesSet2:
        Xes2 = np.append(Xes2, np.array(pl.loadtxt(f)),axis=0)

    NSet1=len(Xes1)
    NSet2 = len(Xes2)
    TrainSet1 = round(NSet1 / 2)
    ValSet1 = round(NSet1 / 4) + TrainSet1

    TrainSet2 = round(NSet2 / 2)
    ValSet2 = round(NSet2 / 4) + TrainSet2

    XTrain = np.concatenate((Xes2[1:TrainSet2], Xes1[1:TrainSet1]))
    YTrain = np.concatenate((np.ones((TrainSet2-1,1)),(-1)*np.ones((TrainSet1-1,1))))

    XVal=np.concatenate((Xes2[TrainSet2+1:ValSet2], Xes1[TrainSet1+1:ValSet1]))
    Yval=np.concatenate((np.ones((ValSet2-TrainSet2-1,1)),(-1)*np.ones((ValSet1-TrainSet1-1,1))))

    XTest=np.concatenate((Xes2[ValSet2+1:NSet2], Xes1[ValSet1+1:NSet1]))
    YTest=np.concatenate((np.ones((NSet2-ValSet2-1,1)),(-1)*np.ones((NSet1-ValSet1-1,1))))

    return XTrain,YTrain,XVal,Yval,XTest,YTest

if __name__ == '__main__':
    #define x,y train val and test sets
    FilesSet1=['hw2_resources\data\mnist_digit_7.csv']
    FilesSet2 = ['hw2_resources\data\mnist_digit_1.csv']
    XTrain, YTrain, XVal, YVal, XTest, YTest=GetData(FilesSet1, FilesSet2)


    # #l1 norm:
    # print "L1 Norm"
    # Lambdas = np.arange(0.001, 5, 0.05)
    # InterLASSO, WLASSO, LambdaLASSO = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas,
    #                                                           RegularizedFunctions.LASSO)
    # #get test error
    # print "L2 Norm"
    # # l2 norm:
    # Lambdas = np.arange(0.001, 5, 0.05)
    # InterLASSO, WLASSO, LambdaLASSO = ValidateParams.Validate(XTrain, YTrain, XVal, YVal, Lambdas,
    #                                                           RegularizedFunctions.Ridge)
    # # get test error
    #

    # print "SVM -Gaussian"
    # # Gaussian RBF SVM:
    # BestC = 0
    # BestRSQ = float('Inf')
    # for c in range(1,5):
    #     print c
    #     for gamma in range(1,3):
    #         print gamma
    #         svm = learner()
    #         svm.set_data(XTrain, YTrain)
    #         svm.set_kernel('rbf')
    #         svm.make_kernel_matrix(gamma=gamma)
    #         svm.train(c)
    #         RQS = svm.test_error(XVal, YVal)
    #         if RQS<BestRSQ:
    #             BestC=c
    #             BestRSQ=RQS
    #
    # # get test error

    print "SVM -Pegasus"
    # Pegasus:
    Bestgamma = 0
    Bestmax_epochs=0
    BestRSQ = float('Inf')
    for gamma in range(1,5):
        for max_epochs in range (100,102):
            svm = learner()
            svm.set_data(XTrain, YTrain)
            svm.set_kernel('rbf')
            svm.make_kernel_matrix(gamma=gamma)
            svm.train_pegasos_kernelized(1, max_epochs)
            RQS = svm.test_error(XVal, YVal)
            if RQS<BestRSQ:
                Bestgamma=gamma
                Bestmax_epochs=max_epochs
                BestRSQ=RQS
    # get test error

