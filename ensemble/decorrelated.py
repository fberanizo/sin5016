# -*- coding: utf-8 -*-

import numpy, time
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

class DecorrelatedEnsemble(BaseEstimator, RegressorMixin):
    """Class that implements a emsemble learning via negative correlation"""
    def __init__(self, n_estimators=4, max_epochs=100):
        self.n_estimators = n_estimators
        self.learning_rate = 0.01
        self.max_epochs = max_epochs

    def fit(self, X, y):
        """Trains the emsemble and returns the trained model"""
        self.samples_size = X.shape[0]
        self.n_input = X.shape[1]
        remaining_epochs = self.max_epochs
        epsilon = 0.01
        error = 1

        # Initialize weights for each estimator
        self.WEst = [numpy.random.rand(1, self.n_input) for x in xrange(self.n_estimators)]

        # Initialize weights for ensemble
        self.WEns = numpy.random.rand(1, self.n_estimators)

        # Repeats until error is small enough or max epochs is reached
        while error > epsilon and remaining_epochs > 0:
            # For each input instance (each line)
            for self.X, self.y in zip(X, y):
                self.Y = numpy.array([])

                # Computes each estimator output
                for W in self.WEst:
                    self.Y = numpy.append(self.Y, numpy.dot(self.X, W.T))

                # Computes ensemble result
                Ens = numpy.dot(self.Y, self.WEns.T)

                error = mean_squared_error(Ens, numpy.array([self.y]))

                # Computes ensemble derivative
                dJdEns = (Ens-self.y)/self.n_estimators

                # Corrects ensemble weights
                self.WEns -= self.learning_rate*dJdEns

                # Computes each estimator derivative
                for i in xrange(0, self.n_estimators):
                    dJdEst = (self.Y[i]-self.y)
                    for j in xrange(0, self.n_estimators):
                        if i != j:
                            dJdEst += normalize((self.Y[i]-self.y)*(self.Y[j]-self.y))/10
                    self.WEst[i] -= self.learning_rate*dJdEst

            remaining_epochs -= 1

        return self

    def predict(self, X):
        """Predicts test values"""
        y_pred = numpy.array([])
        # For each input instance (each line)
        for X_test in X:
            Y = numpy.array([])

            # Calculates each estimator output
            for W in self.WEst:
                Y = numpy.append(Y, numpy.dot(X_test, W.T))

            # Computes ensemble result
            y_pred = numpy.append(y_pred, numpy.dot(Y, self.WEns.T))

        return y_pred

    def score(self, X, y_true):
        """Calculates accuracy"""
        y_pred = numpy.array([])
        
        # For each input instance (each line)
        for X_test in zipX:
            Y = numpy.array([])

            # Calculates each estimator output
            for W in self.WEst:
                Y = numpy.append(Y, numpy.dot(X_test, W.T))

            # Computes ensemble result
            y_pred = numpy.append(y_pred, numpy.dot(Y, self.WEns.T))

        return accuracy_score(y_true.flatten(), y_pred.flatten())


    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
