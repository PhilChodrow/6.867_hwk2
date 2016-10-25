import numpy as np
import sys
sys.path.append('../')



def Validate(DataSetXTrain, DataSetYTrain, DataSetXVal, DataSetYVal,Lamdas,RgularizedFunction):
    BestRSQ = float('Inf')
    BestLambda = 0
    for Lambda in Lamdas:
        Inter,W,RSQ = RgularizedFunction(DataSetXTrain, DataSetYTrain, DataSetXVal, DataSetYVal, Lambda)
        print "Lambda:",Lambda,"RSQ:", RSQ
        if RSQ < BestRSQ:
            BestLambda = Lambda
            BestRSQ = RSQ
    Inter, W, RSQ = RgularizedFunction(DataSetXTrain, DataSetYTrain, DataSetXVal, DataSetYVal, BestLambda)
    return Inter, W, BestLambda,RSQ
