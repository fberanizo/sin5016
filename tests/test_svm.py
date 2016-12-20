# -*- coding: utf-8 -*-

from context import svm

import unittest, numpy, pandas, cProfile
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

class SVMTestSuite(unittest.TestCase):
    """Suíte de testes para máquina de vetores suporte."""

    def test_iris(self):
        """Treina um svm para classificação do conjunto de dados adult."""

        # Lê arquivo de dados
        dataset = pandas.read_csv('../datasets/adult/adult.data', sep=',')
        X = dataset.ix[:, dataset.columns != 'income'].to_dict(orient='records')
        y = dataset.ix[:, dataset.columns == 'income'].as_matrix()

        vectorizer = DictVectorizer(sparse = False)
        X = vectorizer.fit_transform(X)
        y = label_binarize(y, classes=numpy.unique(y), neg_label=-1, pos_label=1)

        X_train, X_test, y_train, y_test = train_test_split(X[:100,], y[:100], test_size=0.2)

        classifier = svm.SVM(C=1.0, kernel='linear')
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_train)
        print(y_pred)
        print(classification_report(y_train, y_pred))

        #classifier = LinearSVC(C=1.0)
        #classifier.fit(X_train, y_train)
        
        #y_pred = classifier.predict(X_train)
        #print(y_pred)
        #print(classification_report(y_train, y_pred))

        assert True

if __name__ == '__main__':
    unittest.main()