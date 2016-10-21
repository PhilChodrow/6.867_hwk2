from sklearn import linear_model
import numpy as np
import numpy.linalg as linalg


def LASSO(DataSetXTrain, DataSetYTrain, DataSetXVal, DataSetYVal, Lambda):
    clf = linear_model.LogisticRegression(penalty='l1',C=Lambda)
    # DataSet should look like:[[0, 0], [1, 1], [2, 2]], [0, 1, 2]
    clf.fit(DataSetXTrain, np.ravel(DataSetYTrain))
    Err= clf.predict(DataSetXVal)-np.array(DataSetYVal).T
    RSQ = np.dot(np.array(Err), np.array(Err).T)
    return clf.intercept_,clf.coef_, RSQ


def Ridge(DataSetXTrain, DataSetYTrain, DataSetXVal, DataSetYVal, Lambda):
    clf = linear_model.LogisticRegression(penalty='l2',C=Lambda)
    # DataSet should look like:[[0, 0], [1, 1], [2, 2]], [0, 1, 2]
    clf.fit(DataSetXTrain, np.ravel(DataSetYTrain))
    Err= clf.predict(DataSetXVal)-np.array(DataSetYVal).T
    RSQ = np.dot(np.array(Err), np.array(Err).T)
    return clf.intercept_,clf.coef_, RSQ

