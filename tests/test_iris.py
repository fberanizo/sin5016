# -*- coding: utf-8 -*-

from context import mlp, svm

import unittest, numpy, pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class IrisTestSuite(unittest.TestCase):
    """Suíte de testes para MLP e SVM utilizando conjunto de dados iris."""

    def test_iris(self):
        """Treina um MLP e um SVM para classificação do conjunto de dados iris."""

        # Lê arquivo de dados
        dataset = pandas.read_csv('../datasets/iris/iris.data', sep=',')
        X = dataset.ix[:, dataset.columns != 'class'].to_dict(orient='records')
        y = dataset.ix[:, dataset.columns == 'class'].as_matrix()[:,0]

        vectorizer = DictVectorizer(sparse = False)
        X = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        classifier = mlp.MLP(hidden_layer_size=10)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

        classifier = svm.SVM(kernel='linear', C=1)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

        assert True

if __name__ == '__main__':
    unittest.main()