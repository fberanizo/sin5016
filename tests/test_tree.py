# -*- coding: utf-8 -*-

from context import tree

import unittest, numpy, pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class TreeTestSuite(unittest.TestCase):
    """Suíte de testes para árvores de decisão."""

    def test_iris(self):
        """Treina uma árvore de decisão para classificação do conjunto de dados Iris."""

        # Lê arquivo de dados
        dataset = pandas.read_csv('../datasets/iris/iris.data', sep=',').as_matrix()
        X = dataset[:,:-1]
        y = dataset[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        classifier = tree.DecisionTree()
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

        assert True

if __name__ == '__main__':
    unittest.main()