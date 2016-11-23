# -*- coding: utf-8 -*-

import numpy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize

class SVM(BaseEstimator, ClassifierMixin):
    """Classe que implementa um SVM."""

    def __init__(self, C, kernel='linear', gamma='auto'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.eps = 1e-5
        self.tolerance = 1e-5 # ksi

    def fit(self, X, y):
        """Treina o SVM utilizando o algoritmo SMO. 
           Busca na região factível do problema dual e maximiza a função objetivo. 
           A solução ótima pode ser checada utilizando as condições de KKT."""
        (samples_size, input_size) = X.shape

        self.classes = numpy.unique(y)
        if self.classes.size != 2:
            raise Exception('Multiclass SVM not implemented')

        self.X = normalize(X)
        self.y = y

        self.partition_1 = numpy.where(y == self.classes[0])
        self.partition_2 = numpy.where(y == self.classes[1])
        self.y[self.partition_1] = 1
        self.y[self.partition_2] = -1

        # Inicializa multiplicadores de Lagrange para cada amostra
        self.alph = numpy.zeros((X.shape[0],))

        # A cada iteração seleciona duas amostras de classes diferentes
        # que violem condições de KKT por mais que a tolerância 
        # e que sejam 'non-bound' (vetores de margem ou de erro)

        # Seleciona amostra da classe "1"
        alph1 = self.select_sample_1()

        return self

    def select_sample_1(self):
        alph1 = 0
        for x, y, alph in zip(self.X[self.partition_1], [1], self.alph[self.partition_1]):
            kkt = self.f(x.T) * y - 1
            if ((alph1 == 0 and kkt < -self.tolerance) or
                (alph == self.C and kkt > -self.tolerance) or
                (0 < alph1 < self.C and abs(kkt) < self.eps)):
                alph1 = alph
        return alph1

    def f(self, x):
        """Calcula o valor de f(x) para kernel linear."""
        fx = sum(numpy.multiply(numpy.multiply(self.y, self.alph), numpy.dot(self.X, x)) + self.tolerance)
        return -1 if fx < 0 else 1


    def predict(self, X):
        """Rotula amostras utilizando o SVM previamente treinado."""
        y = []
        for x_row in X:
            y_row = self.classes[0] if self.f(x_row) == 1 else self.classes[1]
            y.append(y_row)

        y = numpy.asarray(y)
        return y