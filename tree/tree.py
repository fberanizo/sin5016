# -*- coding: utf-8 -*-

import numpy, time
from sklearn.base import BaseEstimator, ClasifierMixin

class DecisionTree(BaseEstimator, ClasifierMixin):
    """Class that implements a decision tree classifier."""
    def __init__(self):
        pass

    def fit(self, X, y):
        """Trains the decision tree and returns the trained model"""
        return self

    def predict(self, X):
        pass

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self