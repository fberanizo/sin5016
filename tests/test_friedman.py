# -*- coding: utf-8 -*-

from context import svm, mlp

import unittest, numpy, time
from os import listdir
from os.path import isfile, join
from random import shuffle
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

class FriedmanTestSuite(unittest.TestCase):
    """Treina modelos e realiza comparação utilizando teste de Friedman."""

    def test_friedman(self):
        """Lê parâmetros, treina e testa modelos."""
        path = join('/', 'home', 'fabio', 'imagens_clodoaldo', 'Wavelet')#input('Diretório com o conjunto de dados pré-processado (com os arquivos *.mat): ')
        level = '3'#input('Nível de decomposição (1, 2 ou 3): ')
        wavelet = 'db2'#input('Função wavelet mãe (db2, db4, sym3, sym4, sym5): ')
        band = 'LL'#input('Qual sub-banda) utilizar (LL, HL, LH, HH): ')
        band_dict = {'LL':0, 'HL':1, 'LH':2, 'HH':3}

        # Lê diretório com o conjunto de dados
        path = join(path, wavelet)
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files = sorted(files, key=lambda file: int(file.split('.')[0][1:]))

        X, y = [], []
        for file in files[:1680]:
            try:
                print('Lendo arquivo %s'% file)
                dataset = loadmat(join(path, file))
            except Exception:
                continue
            finally:
                # dataset['coef'][0][0][SUB-BANDA][0,LEVEL], SUB-BANDAS = [0..3] (LL, LH, HL, HH)
                data = numpy.ravel(dataset['coef'][0][0][band_dict[band]][0,int(level)-1])
                X.append(data)
                y.append(int(file.split('.')[0][1:]))
        X, y = numpy.array(X), numpy.array(y)

        # Separa 25% do conjunto de dados para teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Realiza busca em grid de parâmetros com 5x2 Fold cross-validation
        skf_outer = StratifiedKFold(n_splits=5)
        skf_inner = StratifiedKFold(n_splits=2)

        # Treina SVM RBF
        clf2 = svm.SVM(kernel='rbf')
        grid = {'C': [1], 'gamma': [0]}
        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf2, param_grid=grid, cv=skf_inner, verbose=0, n_jobs=2)
        clf.fit(X_train, y_train)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X_train, y=y_train, cv=skf_outer, verbose=0, n_jobs=1)
        print("RBF SVM validation accuracy" % validation_score.mean())
        y_pred = clf.predict(X_test)
        print("RBF SVM test score")
        print(classification_report(y_test, y_pred))
        joblib.dump(clf, 'trained-estimators/svm-rbf-'+wavelet+'-'+level+'-'+band+'.pkl') 
        return

        # Aplica PCA para diminuir dimensionalidade dos dados até que a variância seja maior que 0.9
        pca = PCA(n_components=0.9)
        X_mlp = pca.fit_transform(X)
        print('Selecionou %d componentes no PCA' % pca.n_components_)

        # Treina MLP
        grid = {'hidden_layer_size': [pca.n_components_ / 4]}
        clf3 = mlp.MLP()

        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf3, param_grid=grid, cv=skf_inner, verbose=0, n_jobs=2)
        clf.fit(X_train, y_train)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X_train, y=y_train, cv=skf_outer, verbose=0, n_jobs=1)
        print("MLP validation accuracy" % validation_score.mean())
        y_pred = clf.predict(X_test)
        print("MLP test score")
        print(classification_report(y_test, y_pred))
        joblib.dump(clf, 'trained-estimators/mlp-'+wavelet+'-'+level+'-'+band+'.pkl')

if __name__ == '__main__':
    unittest.main()
