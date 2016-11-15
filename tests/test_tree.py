# -*- coding: utf-8 -*-

from context import tree

import unittest, numpy, pandas

class TreeTestSuite(unittest.TestCase):
    """Suíte de testes para árvores de decisão."""

    def test_iris(self):
        """Treina uma árvore de decisão para classificação do conjunto de dados Iris."""

        # Lê arquivo de dados
        dataset = pandas.read_csv('../datasets/iris/iris.data', sep=',').as_matrix()
        X = dataset[:,:-1]
        y = dataset[:,-1]


        assert True

if __name__ == '__main__':
    unittest.main()