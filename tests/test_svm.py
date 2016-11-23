# -*- coding: utf-8 -*-

from context import svm

import unittest, numpy, pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SVMTestSuite(unittest.TestCase):
    """Suíte de testes para máquina de vetores suporte."""

    def test_iris(self):
        """Treina um svm para classificação do conjunto de dados Iris."""

        # Lê arquivo de dados
        dataset = pandas.read_csv('../datasets/iris/iris.data', sep=',').as_matrix()
        X = dataset[:,:-1]
        y = dataset[:,-1]

        # Deixa apenas 2 classes pois versão multiclasses não foi implementada
        classes = numpy.unique(y)
        partition = numpy.where((y == classes[0]) | (y == classes[1]))
        X = X[partition]
        y = y[partition]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        classifier = svm.SVM(C=0.1)
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

        assert True

if __name__ == '__main__':
    unittest.main()