# -*- coding: utf-8 -*-

from context import svm, mlp

import unittest, numpy, time
from os import listdir
from os.path import isfile, join
from random import shuffle
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

class SdumlaHtmTestSuite(unittest.TestCase):
    """Suíte de testes para o conjunto de dados SDUMLA-HTM utilizando lendo parâmetros do teclado."""

    def test_sdumla_htm(self):
        """Lê parâmetros, treina e testa modelos."""
        # Parâmetros
        # O usuário deve definir três parâmetros, a saber:
        # Nível de decomposição (1, 2 ou 3)
        # Função wavelet mãe (haar, Daubechies, Coiflets, Symlets)
        # Qual(is) sub-banda(s) utilizar (LL, HL, LH, HH)

        level = raw_input('Nível de decomposição (1, 2 ou 3): ')
        wavelet = raw_input('Função wavelet mãe (db2, db4, sym3, sym4, sym5): ')
        band = raw_input('Qual sub-banda) utilizar (LL, HL, LH, HH): ')
        band_dict = {'LL':0, 'HL':1, 'LH':2, 'HH':3}

        # Lê diretório com o conjunto de dados
        path = join('/', 'home', 'fabio', 'imagens_clodoaldo', 'Wavelet', wavelet)
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files = sorted(files, key=lambda file: int(file.split('.')[0][1:]))
        shuffle(files)

        X, y = [], []
        for file in files:
            try:
                print(file)
                dataset = loadmat(join(path, file))
            except Exception, e:
                continue
            finally:
                # dataset['coef'][0][0][SUB-BANDA][0,LEVEL], SUB-BANDAS = [0..3] (LL, LH, HL, HH)
                data = numpy.ravel(dataset['coef'][0][0][band_dict[band]][0,int(level)-1])
                X.append(data)
                y.append(int(file.split('.')[0][1:]))
        X, y = numpy.array(X), numpy.array(y)
        #numpy.save(join('..', 'datasets', 'sdumla-htm', wavelet+'.'+level+'.'+band+'.data'), X)
        #numpy.save(join('..', 'datasets', 'sdumla-htm', wavelet+'.'+level+'.'+band+'.target'), y)

        #X = numpy.load(join('..', 'datasets', 'sdumla-htm', wavelet+'.'+level+'.'+band+'.data.npy'))
        #y = numpy.load(join('..', 'datasets', 'sdumla-htm', wavelet+'.'+level+'.'+band+'.target.npy'))

        # Separa 25% do conjunto de dados para teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Realiza busca em grid de parâmetros com 5x2 Fold cross-validation
        skf_outer = StratifiedKFold(n_splits=5)
        skf_inner = StratifiedKFold(n_splits=2)
        
        # Treina SVM Linear
        clf1 = svm.SVM(kernel='linear')
        grid = {'C': [0.1, 1, 10, 100]}
        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf1, param_grid=grid, cv=skf_inner)
        clf.fit(X_train, y_train)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X_train, y=y_train, cv=skf_outer)
        print("Linear SVM validation accuracy" % validation_score.mean())
        y_pred = clf.predict(X_test)
        print("Linear SVM test score")
        print(classification_report(y_test, y_pred))
        joblib.dump(clf, 'trained-estimators/svm-linear-'+wavelet+'-best.pkl') 

        # Treina SVM RBF
        clf2 = svm.SVM(kernel='rbf')
        grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0, 0.1, 1]}
        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf2, param_grid=grid, cv=skf_inner)
        clf.fit(X_train, y_train)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X_train, y=y_train, cv=skf_outer)
        print("RBF SVM validation accuracy" % validation_score.mean())
        y_pred = clf.predict(X_test)
        print("RBF SVM test score")
        print(classification_report(y_test, y_pred))
        joblib.dump(clf, 'trained-estimators/svm-rbf-'+wavelet+'-best.pkl') 

        # Treina MLP
        grid = {'hidden_layer_size': [100, 300, 500]}
        clf3 = mlp.MLP()
        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf3, param_grid=grid, cv=skf_inner)
        clf.fit(X_train, y_train)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X_train, y=y_train, cv=skf_outer)
        print("MLP validation accuracy" % validation_score.mean())
        y_pred = clf.predict(X_test)
        print("MLP test score")
        print(classification_report(y_test, y_pred))
        joblib.dump(clf, 'trained-estimators/mlp-sym3-best.pkl') 

        assert True

if __name__ == '__main__':
    unittest.main()