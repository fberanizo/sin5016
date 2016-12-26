# -*- coding: utf-8 -*-

from context import svm

import unittest, numpy, mlp
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

class SDUMLAHTMTestSuite(unittest.TestCase):
    """Suíte de testes para máquina de vetores suporte."""

    def test_db2(self):
        """Treina um svm para classificação do conjunto de dados SDUMLA-HTM."""
        # Parâmetros
        # O usuário deve definir três parâmetros, a saber:
        # Nível de decomposição (1, 2 ou 3)
        # Função wavelet mãe (haar, Daubechies, Coiflets, Symlets)
        # Qual(is) sub-banda(s) utilizar (LL, HL, LH, HH)

        level = 2 # Level 3
        wavelet = 'db2'
        band = 'LL'  # LL, HL, LH, HH
        band_dict = {'LL':0, 'HL':1, 'LH':2, 'HH':3}

        # Lê diretório com o conjunto de dados
        path = join('..', 'datasets', 'sdumla-htm', 'Wavelet', wavelet)
        files = [f for f in listdir(path) if isfile(join(path, f))]
        
        X, y = [], []
        for file in files:
            dataset = loadmat(join(path, file))
            # dataset['coef'][0][0][SUB-BANDA][0,LEVEL], SUB-BANDAS = [0..3] (LL, LH, HL, HH)
            data = numpy.ravel(dataset['coef'][0][0][band_dict[band]][0,level])
            X.append(data)
            y.append(numpy.random.randint(1, 4))
            #y.append(int(files[0].split('.')[0][1:]))
        X, y = numpy.array(X), numpy.array(y)
        print(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = mlp.MLP(hidden_layer_size=3)
        clf.fit(X_train, y_train)
        #clf = svm.SVM(kernel='linear', C=1)
        #clf.fit(X_train, y_train)

        assert True

if __name__ == '__main__':
    unittest.main()