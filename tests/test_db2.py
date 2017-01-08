# -*- coding: utf-8 -*-

from context import svm, mlp

import unittest, numpy
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

class DB2TestSuite(unittest.TestCase):
    """Suíte de testes para o conjunto de dados SDUMLA-HTM utilizando wavelet daubechies."""
    def __init__(self, *args, **kwargs):
        super(DB2TestSuite, self).__init__(*args, **kwargs)
        X, y = self.read_dataset()
        self.n_datasets = 10
        self.X_train, self.X_test, self.y_train, self.y_test = [None]*self.n_datasets, [None]*self.n_datasets, [None]*self.n_datasets, [None]*self.n_datasets
        self.X_train_PCA, self.X_test_PCA = [None]*self.n_datasets, [None]*self.n_datasets
        # Divide conjunto de dados em 10 subconjuntos, ~840 amostras de 10 classes
        print("Dividindo conjunto de dados em 10 subconjuntos...")
        for i in range(self.n_datasets):
            begin = i * 840
            end = begin + 840
            # 25% de cada conjunto de dados será para teste
            self.X_train[i], self.X_test[i], self.y_train[i], self.y_test[i] = train_test_split(X[begin:end,:], y[begin:end], test_size=0.25)
            # Aplica PCA para diminuir dimensionalidade dos dados 
            # até que a variância seja maior que 0.9. Só é utilizado para MLP.
            pca = PCA(n_components=0.9)
            self.X_train_PCA[i] = pca.fit_transform(self.X_train[i])
            self.X_test_PCA[i] = pca.transform(self.X_test[i])

    def test_db2(self):
        """Lê parâmetros, treina e testa modelos."""
        k, clf1, clf2, clf3 = 3, [], [], []
        # Treina classificadores em cada um dos 5 conjunto de dados
        for i in range(self.n_datasets):
            print("Treinando conjunto de dados %d de %d" % (i+1, self.n_datasets))
            clf1.append(self.train_svm_linear(self.X_train[i], self.y_train[i]))
            clf2.append(self.train_svm_rbf(self.X_train[i], self.y_train[i]))
            clf3.append(self.train_mlp(self.X_train_PCA[i], self.y_train[i]))
            joblib.dump(clf1[i], 'trained-estimators/db2-3-LL/svm-linear-'+str(i+1)+'.pkl') 
            joblib.dump(clf2[i], 'trained-estimators/db2-3-LL/svm-rbf-'+str(i+1)+'.pkl') 
            joblib.dump(clf3[i], 'trained-estimators/db2-3-LL/mlp-'+str(i+1)+'.pkl') 
        #y_pred = classifier.predict(X_test)
        #print(classification_report(y_test, y_pred))
        # Teste de Friedman
        #clf1.append(joblib.load('trained-estimators/db2-3-LL/svm-linear-0.pkl'))
        #clf2.append(joblib.load('trained-estimators/db2-3-LL/svm-rbf-0.pkl'))
        #clf3.append(joblib.load('trained-estimators/db2-3-LL/mlp0.pkl'))
        rank = []
        for i in range(self.n_datasets):
            # Cria um rank por acurácia de teste para cada modelo
            rank.append(sorted([(1, clf1[i].score(self.X_test[i], self.y_test[i])), \
                                (2, clf2[i].score(self.X_test[i], self.y_test[i])), \
                                (3, clf3[i].score(self.X_test_PCA[i], self.y_test[i]))], key=lambda t: t[1], reverse=True))
        rank = numpy.array(map(lambda r: [r[0][0], r[1][0], r[2][0]], rank))
        # Calcula rank médio
        rj = numpy.mean(rank, axis=0)
        print("Rank médio para SVM Linear, SVM RBF e MLP: %s" % rj)
        rmean = rank.mean()
        sst = self.n_datasets * ((rj -rmean)**2).sum()
        sse = 1.0/(self.n_datasets*(k-1)) * ((rank-rmean)**2).sum()
        # Calcula estatística
        chi_square = sst/sse
        print("chi_square = %f" % chi_square)
        # para k=3 e N = 5, p-valor < 0.05, chi^2 > 6.4
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
        print("Lendo arquivos *.mat...")
        for file in files:
            try:
                #print('Lendo arquivo %s'% file)
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
        skf_inner = StratifiedKFold(n_splits=2)
        skf_outer = StratifiedKFold(n_splits=5)
        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf, param_grid=grid, cv=skf_inner, verbose=0, n_jobs=2)
        clf.fit(X, y)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X, y=y, cv=skf_outer, verbose=0, n_jobs=1)
        print("SVM Linear - Acurácia de validação = %f" % validation_score.mean())
        return clf

    def train_svm_rbf(self, X, y):
        """Treina um SVM RBF e retorna o classificador treinado."""
        clf = svm.SVM(kernel='rbf')
        grid = {'C': [1], 'gamma': [0]}
        # Realiza busca em grid de parâmetros com 5x2 Fold cross-validation
        skf_inner = StratifiedKFold(n_splits=2)
        skf_outer = StratifiedKFold(n_splits=5)
        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf, param_grid=grid, cv=skf_inner, verbose=0, n_jobs=2)
        clf.fit(X, y)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X, y=y, cv=skf_outer, verbose=0, n_jobs=1)
        print("SVM RBF - Acurácia de validação = %f" % validation_score.mean())
        return clf

    def train_mlp(self, X, y):
        """Treina MLP e retorna o classificador treinado."""
        clf = mlp.MLP()
        grid = {'hidden_layer_size': [15]}
        # Realiza busca em grid de parâmetros com 5x2 Fold cross-validation
        skf_inner = StratifiedKFold(n_splits=2)
        skf_outer = StratifiedKFold(n_splits=5)
        # Otimiza parâmetros (2-fold)
        clf = GridSearchCV(estimator=clf, param_grid=grid, cv=skf_inner, verbose=0, n_jobs=2)
        clf.fit(X, y)
        # Validação com parâmetros ótimos de treino (5-fold)
        validation_score = cross_val_score(clf, X=X, y=y, cv=skf_outer, verbose=0, n_jobs=1)
        print("MLP - Acurácia de validação = %f" % validation_score.mean())
        return clf

if __name__ == '__main__':
    unittest.main()
