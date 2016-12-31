# -*- coding: utf-8 -*-

from context import mlp

import unittest, numpy, pandas
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class MLPTestSuite(unittest.TestCase):
    """Suíte de testes para multilayer perceptron."""

    def test_iris(self):
        """Treina um svm para classificação do conjunto de dados iris."""

        # Lê arquivo de dados
        dataset = pandas.read_csv('../datasets/iris/iris.data', sep=',')
        X = dataset.ix[:, dataset.columns != 'class'].to_dict(orient='records')
        y = dataset.ix[:, dataset.columns == 'class'].as_matrix()

        vectorizer = DictVectorizer(sparse = False)
        X = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        classifier = mlp.MLP(hidden_layer_size=10)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print(y_pred)
        print(classification_report(y_test, y_pred))
        #joblib.dump(classifier, 'mlp.pkl') 

        assert True

if __name__ == '__main__':
    unittest.main()