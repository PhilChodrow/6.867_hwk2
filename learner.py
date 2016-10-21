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
        self.w               = 0 

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
        self.w = np.dot(self.beta, self.X)
        return {'P' : P, 'q' : q, 'A' : A, 'b' : b, 'G' : G, 'h' : h}

    def set_supports(self, kind = 'kernel'):

        supports = np.abs(self.beta) > 0.001
        self.support_vectors = self.X[supports,:]
        self.support_betas   = self.beta.T[supports]
        self.set_bias(kind)

    def predict(self, x, kind = 'kernel'):
 
        if(kind == 'kernel'):
            f = np.vectorize(lambda i: self.kernel_func(x, self.support_vectors[i]))
            return np.dot(self.support_betas,
                          f(np.arange(self.support_vectors.shape[0]))) + self.bias
        elif(kind == 'pegasos'):
            return np.dot(x, self.w) + self.bias


    def classify(self, x, kind = 'kernel'):
        return np.sign(self.predict(x, kind))

    def set_bias(self, kind = 'kernel'):
        # http://cs229.stanford.edu/notes/cs229-notes3.pdf

        predictor   = np.vectorize(lambda i: self.predict(self.X[i], kind))
        preds = predictor(np.arange(self.X.shape[0]))
        self.bias   = -(np.min(preds[self.Y.T[0] == 1]) + np.max(preds[self.Y.T[0] == -1])) / 2.0

    def training_error(self, kind = 'kernel'):
        return self.test_error(self.X, self.Y, kind)

    def test_error(self, X_test, Y_test, kind = 'kernel'):
        classifier = np.vectorize(lambda i: self.classify(X_test[i], kind))
        pred_classes = classifier(np.arange(Y_test.shape[0]))
        return np.mean(np.abs(Y_test.T[0] - pred_classes) > 0)

    def get_margin(self):
        return 1.0 / np.dot(self.w, self.w)

    def get_supports(self):
        return self.support_vectors

    def train_pegasos_linear(self, gamma, max_epochs):
        t    = 0
        w = np.zeros(self.X.shape[1])

        for j in range(max_epochs):
            for i in np.random.permutation(np.arange(self.X.shape[0])):
                t += 1
                eta = 1.0 / (t*gamma)
                first_term = (1 - eta * gamma) * w
                if self.Y[i] * (np.dot(w, self.X[i])) < 1:
                    w = first_term + eta * self.Y[i]*self.X[i]
                else:
                    w = first_term
        self.w = w
        self.set_bias(kind = 'pegasos')

    def train_pegasos_kernelized(self, gamma, max_epochs):
        t = 0
        beta = np.zeros(self.X.shape[0])
        for j in range(max_epochs):
            for i in np.random.permutation(np.arange(self.X.shape[0])):
                t += 1
                eta = 1.0 / (t*gamma)
                first_term = (1-eta * gamma) * beta[i]
                if(self.Y[i] * np.dot(beta, self.K[i]) < 1.0):
                    beta[i] = first_term + eta * self.Y[i]
                else:
                    beta[i] = first_term
        self.beta = beta
        self.w = np.dot(self.beta, self.X)
        self.set_supports(kind = 'kernel')
