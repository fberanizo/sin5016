# -*- coding: utf-8 -*-

import numpy, enum
from sklearn.base import BaseEstimator, ClassifierMixin

class SVM(BaseEstimator, ClassifierMixin):
    """Classe que implementa um SVM."""

    def __init__(self, c, kernel='linear', gamma='auto'):
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.eps = 1e-5
        self.tolerance = 1e-5

    def fit(X, y):
        """Treina o SVM utilizando o algoritmo SMO. 
           Busca na região factível do problema dual e maximiza a função objetivo. 
           A solução ótima pode ser checada utilizando as condições de KKT."""
           return self