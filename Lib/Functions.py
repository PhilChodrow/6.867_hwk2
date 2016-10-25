import math
import numpy as np
import numpy.linalg as linalg
def GaussianFunc(x,gaussMean,gaussCov):
    NPgaussMean =       np.array(gaussMean)
    NPgaussCov  =       np.array(gaussCov)
    NPx         =       np.array(x)
    det         =       linalg.det(NPgaussCov)
    XlessU      =       NPx-NPgaussMean
    f           =       -(1/math.sqrt(math.pow((2*math.pi), 2)*det)) * \
                        math.exp(-0.5*(np.dot(np.dot(XlessU.T, linalg.inv(NPgaussCov)), XlessU)))
    return f

def GradientGaussianFunc(x, gaussMean, gaussCov):
    NPgaussMean =       np.array(gaussMean)
    NPgaussCov  =       np.array(gaussCov)
    NPx         =       np.array(x)
    XlessU      =       NPx-NPgaussMean
    f           =       -GaussianFunc(x, gaussMean, gaussCov)*np.dot(linalg.inv(NPgaussCov), XlessU)
    return f
def QuadraticBowlFunc(x, quadBowlA, quadBowlb):
    NPquadBowlA =       np.array(quadBowlA)
    NPquadBowlb =       np.array(quadBowlb)
    NPx         =       np.array(x)
    f           =       0.5*(np.dot(np.dot(NPx.T, NPquadBowlA), NPx))-np.dot(NPx.T, NPquadBowlb)
    return f

def GradientQuadraticBowlFunc(x, quadBowlA, quadBowlb):
    NPquadBowlA =       np.array(quadBowlA)
    NPquadBowlb =       np.array(quadBowlb)
    NPx         =       np.array(x)
    f           =       np.dot(NPquadBowlA, NPx)-NPquadBowlb
    return f

def SSE(theta, X, Y):
    return np.dot((np.dot(np.array(X), np.array(theta))-np.array(Y)).T,
                  (np.dot(np.array(X), np.array(theta))-np.array(Y)))

def GradientSSE(theta, X, Y):
    return 2*np.dot((np.dot(np.array(X), np.array(theta))-np.array(Y)), np.array(X))

def GradientSSEPoint(theta_t, x, y):
    return 2*(np.dot(np.array(x).T, np.array(theta_t))-y)*np.array(x)


def RegularizedNLL(W,X,Y,Lambda):
    func=0
    for i in range(1,len(Y)):
        a = np.array(W).dot(np.array(X[i]))
        func -= Y[i]*math.log(sigmoid(a))
        func -= (1-Y[i])*math.log(1-sigmoid(a))
    func += Lambda*np.array(W).dot(np.array(W))
    return func


def GradientRegularizedNLL(W,X,Y,Lambda):
    func=np.zeros(len(W))
    for i in range(1,len(Y)):
        func+= (sigmoid(np.array(W).dot(np.array(X[i])))-Y[i])*X[i]
    func += 2*Lambda*np.array(W)
    return func



def sigmoid(a):
    #print a
    sig= 1/(1+math.exp(-a))
    return sig

def predict(w,x):
    return np.sign(np.dot(np.array(w).T,np.array(x)))

def gaussian_kernel(x, y, s = 1):
    return math.exp(-(np.linalg.norm(x - y))**2 / (s**2))

def linear_kernel(x,y):
    return np.dot(x,y)
