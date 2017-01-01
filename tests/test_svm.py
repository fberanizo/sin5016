# -*- coding: utf-8 -*-

from context import svm

import unittest, numpy, pandas
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SVMTestSuite(unittest.TestCase):
    """Suíte de testes para máquina de vetores suporte."""

    def test_iris(self):
        """Treina um svm para classificação do conjunto de dados iris."""

        # Lê arquivo de dados
        dataset = pandas.read_csv('../datasets/iris/iris.data', sep=',')
        X = dataset.ix[:, dataset.columns != 'class'].to_dict(orient='records')
        y = dataset.ix[:, dataset.columns == 'class'].as_matrix()[:,0]

        vectorizer = DictVectorizer(sparse = False)
        X = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        classifier = svm.SVM(kernel='linear', C=1)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))
        #joblib.dump(classifier, 'svm.pkl') 

        assert True

if __name__ == '__main__':
    unittest.main()