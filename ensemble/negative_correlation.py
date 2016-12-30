# -*- coding: utf-8 -*-

import numpy
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error

class NegativeCorrelationEnsemble(BaseEstimator, RegressorMixin):
    """Classe que implementa um ensemble por correlação negativa."""
    def __init__(self, estimators=[], lambd=0.5, max_epochs=100):
        self.estimators = estimators
        self.lambd = lambd
        self.learning_rate = 0.01
        self.max_epochs = max_epochs

    def fit(self, X, y):
        """Treina um ensemble e retorna o modelo treinado."""
        self.samples_size = X.shape[0]
        self.n_input = X.shape[1]
        remaining_epochs = self.max_epochs
        epsilon = 0.001
        error = 1

        # Inicializa pesos para cada classificador
        self.WEst = [numpy.random.rand(1, self.n_input) for x in xrange(len(self.estimators))]

        # Inicializa pesos para o ensemble
        self.WEns = numpy.random.rand(1, len(self.estimators))

        # Repete até que o erro seja pequeno, ou o máximo de épocas é alcançado
        while error > epsilon and remaining_epochs > 0:

            # Treinamento padrão-a-padrão
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
                for i in xrange(self.n_estimators):
                    dJdEst = (1-self.lambd)*(self.Y[i]-self.y) + self.lambd*(Ens-self.y)
                    # Corrects estimator weights
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
        return {"n_estimators": self.n_estimators, "lambd": self.lambd}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
