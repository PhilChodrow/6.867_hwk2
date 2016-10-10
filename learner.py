import numpy as np
from cvxopt import solvers, matrix

class learner:

    def __init__(self):

        self.beta            = None
        self.X               = None
        self.Y               = None
        self.K               = None
        self.kernel_func     = None
        self.support_vectors = None
        self.support_betas   = None
        self.bias            = 0

    def set_data(self,X,Y):

        self.X = X
        self.Y = Y

    def set_kernel_function(self, kernel_func):
        self.kernel_func = kernel_func

    def make_kernel_matrix(self,  **kwargs):
        self.K = np.fromfunction(np.vectorize(
            lambda i, j: self.kernel_func(self.X[i],
                                          self.X[j]), **kwargs),
                        (self.X.shape[0], self.X.shape[0]),
                        dtype='int64')

    def make_gram_matrix(self):
        return np.dot(self.Y, self.Y.T) * self.K

    def train(self, C):

        P = self.make_gram_matrix()
        q = - np.ones((len(P),1))
        A = self.Y.T
        b = np.zeros((1,1))
        G = np.concatenate((np.eye(len(P)),
                            -np.eye(len(P))),
                           axis = 0)
        h = np.concatenate((C * np.ones((len(P),1)),
                            -np.zeros((len(P),1))),
                           axis = 0)

        P = matrix(P, tc = 'd')
        q = matrix(q, tc = 'd')
        A = matrix(A, tc = 'd')
        b = matrix(b, tc = 'd')
        G = matrix(G, tc = 'd')
        h = matrix(h, tc = 'd')

        solvers.options['show_progress'] = False
        alpha = solvers.qp(P = P, q = q, G = G, h = h, A = A, b = b, )['x']

        self.beta = (alpha * self.Y).T[0]
        self.set_supports()
        return {'P' : P, 'q' : q, 'A' : A, 'b' : b, 'G' : G, 'h' : h}

    def set_supports(self):

        supports = np.abs(self.beta) > 0.0000001
        self.support_vectors = self.X[supports,:]
        self.support_betas   = self.beta.T[supports]
        self.set_bias()

    def predict(self, x):

        f = np.vectorize(lambda i: self.kernel_func(x, self.support_vectors[i]))
        return np.dot(self.support_betas,
                      f(np.arange(self.support_vectors.shape[0]))) + self.bias

    def classify(self, x):
        return np.sign(self.predict(x))

    def set_bias(self):
        # http://cs229.stanford.edu/notes/cs229-notes3.pdf

        predictor   = np.vectorize(lambda i: self.predict(self.X[i]))
        preds = predictor(np.arange(self.X.shape[0]))
        self.bias   = -(np.min(preds[self.Y.T[0] == 1]) + np.max(preds[self.Y.T[0] == -1])) / 2.0

    def training_error(self):
        return self.test_error(self.X, self.Y)

    def test_error(self, X_test, Y_test):
        classifier = np.vectorize(lambda i: self.classify(X_test[i]))
        pred_classes = classifier(np.arange(Y_test.shape[0]))
        return np.mean(np.abs(Y_test.T[0] - pred_classes) > 0)

    def get_margin(self):
        return 1.0 / np.dot(self.beta, self.beta)

    def get_supports(self):
        return self.support_vectors
