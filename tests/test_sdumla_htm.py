# -*- coding: utf-8 -*-

from context import svm

import unittest, numpy
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

class SDUMLAHTMTestSuite(unittest.TestCase):
    """Suíte de testes para máquina de vetores suporte."""

    def test_svm(self):
        """Treina um svm para classificação do conjunto de dados SDUMLA-HTM."""

        # Lê diretório com o conjunto de dados
        path = '../datasets/sdumla-htm/Wavelet/db2'
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        dataset = loadmat(files[0])
        # dataset['coef'][0][0][NIVEL][BANDAS]
        print(dataset['coef'][0][0][3][0])
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Parâmetros
        # O usuÁrio deve definir três parâmetros, a saber:
        # Nível de decomposição (1, 2 ou 3)
        # Função wavelet mãe (haar, Daubechies, Coiflets, Symlets)
        # Qual(is) sub-banda(s) utilizar (LL, HL, LH, HH)

        assert True

if __name__ == '__main__':
    unittest.main()