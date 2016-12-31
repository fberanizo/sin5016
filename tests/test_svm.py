# -*- coding: utf-8 -*-

from context import svm

import unittest, numpy, pandas, cProfile
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

class SVMTestSuite(unittest.TestCase):
    """Suíte de testes para máquina de vetores suporte."""

    def test_iris(self):
        """Treina um svm para classificação do conjunto de dados iris."""

        # Lê arquivo de dados
        dataset = pandas.read_csv('../datasets/iris/iris.data', sep=',')
        X = dataset.ix[:, dataset.columns != 'class'].to_dict(orient='records')
        y = dataset.ix[:, dataset.columns == 'class'].as_matrix()

        vectorizer = DictVectorizer(sparse = False)
        X = vectorizer.fit_transform(X)

        p = numpy.random.permutation(y.size)
        X, y = X[p], y[p]

        #partition = numpy.where(y!='Iris-virginica')
        #X, y = X[partition[0],:2], y[partition]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        classifier = svm.SVM(kernel='linear', C=1)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))
        #joblib.dump(classifier, 'svm-linear.pkl') 

        #x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        #y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        #xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, .2), numpy.arange(y_min, y_max, .2))

        #Z = classifier.predict(numpy.c_[xx.ravel(), yy.ravel()])
        #Z = label_binarize(Z, classes=numpy.unique(y_train), neg_label=-1, pos_label=1)

        #Z = Z.reshape(xx.shape)
        #plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        #plt.scatter(X_train[:, 0], X_train[:, 1], c=map(lambda y: 1 if y == 'Iris-setosa' else -1, y_train), cmap=plt.cm.coolwarm)
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.xlim(xx.min(), xx.max())
        #plt.ylim(yy.min(), yy.max())
        #plt.show()

        assert True

if __name__ == '__main__':
    unittest.main()