# -*- coding: utf-8 -*-

from context import svm, mlp

import unittest, numpy, time
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

class SdumlaHtmTestSuite(unittest.TestCase):
    """Suíte de testes para o conjunto de dados SDUMLA-HTM utilizando lendo parâmetros do teclado."""
    def __init__(self, *args, **kwargs):
        super(SdumlaHtmTestSuite, self).__init__(*args, **kwargs)
        X, y = self.read_dataset()
        self.n_datasets = 5
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []
        # Divide conjunto de dados em 5 subconjuntos, 1680 amostras de 21 classes
        # 25% de cada conjunto de dados será para teste
        for i in range(self.n_datasets):
            self.X_train[i], self.X_test[i], self.y_train[i], self.y_test[i] = train_test_split(X[i*1680:(i+1)*1680], y[i*1680:i+*1680], test_size=0.25)

    def test_dataset(self):
        """Lê parâmetros, treina e testa modelos."""
        k, clf1, clf2, clf3 = 3, [], [], []
        # Treina classificadores em cada um dos 5 conjunto de dados
        for i in range(self.n_datasets):
            clf1.append(self.train_svm_linear(self.X_train[i], self.y_train[i]))
            clf2.append(self.train_svm_rbf(self.X_train[i], self.y_train[i]))
            clf3.append(self.train_mlp(self.X_train[i], self.y_train[i]))

        # Teste de Friedman
        rank = []
        for i in range(self.n_datasets):
            # 1) Cria um rank por acurácia no teste para cada modelo
            rank.append(sorted([(1, clf1.score(X_test)), (2, clf2.score(X_test)), (2, clf3.score(X_test))], key=lambda t: t[1], reverse=True))
        rank = numpy.array(map(lambda r: r[1], rank))
        average_rank = numpy.mean(rank, axis=0)
        print("Rank médio para SVMLin, SVMRbf, MLP: %s" % average_rank)
        # Calcula estatística
        chi_square = ((12*self.n_datasets)/k*(k+1)) * ((sum(average_rank**2)-(k*(k+1)**2))/4)
        print("chi_square = %f" % chi_square)
        # para k=3 e N = 5, p-valor < 0.05, chi^2 < 6.4

        assert True

    def read_dataset(self):
        """Lê o conjunto de dados e o divide em 5 partes."""
        # O usuário deve definir três parâmetros, a saber:
        # Nível de decomposição (1, 2 ou 3)
        # Função wavelet mãe (db2, db4, sym3, sym4, sym5)
        # Qual(is) sub-banda(s) utilizar (LL, HL, LH, HH)
        path = join('/', 'home', 'fabio', 'imagens_clodoaldo', 'Wavelet')#input('Diretório com o conjunto de dados pré-processado (com os arquivos *.mat): ')
        level = '3'#raw_input('Nível de decomposição (1, 2 ou 3): ')
        wavelet = 'db2'#raw_input('Função wavelet mãe (db2, db4, sym3, sym4, sym5): ')
        band = 'LL'#raw_input('Qual sub-banda) utilizar (LL, HL, LH, HH): ')
        band_dict = {'LL':0, 'HL':1, 'LH':2, 'HH':3}

        # Lê diretório com o conjunto de dados
        path = join(path, wavelet)
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files = sorted(files, key=lambda file: int(file.split('.')[0][1:]))

        X, y = [], []
        for file in files:
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
        return X, y

    def train_svm_linear(self, X, y):
        """Treina um SVM Linear e retorna o classificador treinado."""
        clf = svm.SVM(kernel='linear')
        grid = {'C': [1]}
        # Realiza busca em grid de parâmetros com 5x2 Fold cross-validation
        skf_outer = StratifiedKFold(n_splits=5)
        skf_inner = StratifiedKFold(n_splits=2)
        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf, param_grid=grid, cv=skf_inner, verbose=0, n_jobs=2)
        clf.fit(X_train, y_train)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X_train, y=y_train, cv=skf_outer, verbose=0, n_jobs=1)
        print("Linear SVM validation accuracy" % validation_score.mean())
        y_pred = clf.predict(X_test)
        print("Linear SVM test score")
        print(classification_report(y_test, y_pred))
        return clf

    def train_svm_rbf(self, X, y):
        """Treina um SVM RBF e retorna o classificador treinado."""
        clf = svm.SVM(kernel='rbf')
        grid = {'C': [1], 'gamma': [0]}
        # Realiza busca em grid de parâmetros com 5x2 Fold cross-validation
        skf_outer = StratifiedKFold(n_splits=5)
        skf_inner = StratifiedKFold(n_splits=2)
        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf, param_grid=grid, cv=skf_inner, verbose=0, n_jobs=2)
        clf.fit(X_train, y_train)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X_train, y=y_train, cv=skf_outer, verbose=0, n_jobs=1)
        print("RBF SVM validation accuracy" % validation_score.mean())
        y_pred = clf.predict(X_test)
        print("RBF SVM test score")
        print(classification_report(y_test, y_pred))
        joblib.dump(clf, 'trained-estimators/svm-rbf-'+wavelet+'-'+level+'-'+band+'.pkl') 
        return clf

    def train_mlp(self, X, y):
        """Treina MLP e retorna o classificador treinado."""
        # Aplica PCA para diminuir dimensionalidade dos dados até que a variância seja maior que 0.9
        pca = PCA(n_components=0.9)
        X = pca.fit_transform(X)
        print(pca.n_components_)
        grid = {'hidden_layer_size': [pca.n_components_ / 4]}
        clf = mlp.MLP()
        # Realiza busca em grid de parâmetros com 5x2 Fold cross-validation
        skf_outer = StratifiedKFold(n_splits=5)
        skf_inner = StratifiedKFold(n_splits=2)
        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf, param_grid=grid, cv=skf_inner, verbose=0, n_jobs=2)
        clf.fit(X_train, y_train)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X_train, y=y_train, cv=skf_outer, verbose=0, n_jobs=1)
        print("MLP validation accuracy" % validation_score.mean())

        
if __name__ == '__main__':
    unittest.main()
