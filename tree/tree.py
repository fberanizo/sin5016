# -*- coding: utf-8 -*-

import numpy, enum
from sklearn.base import BaseEstimator, ClassifierMixin

class DecisionTree(BaseEstimator, ClassifierMixin):
    """Classe que implementa uma árvore de decisão para classificação."""
    def __init__(self, attribute_selection_method='random'):
        self.attribute_selection_method = attribute_selection_method
        self.input_size = None

    def fit(self, X, y):
        """Treina uma árvore de decisão e retorna o modelo treinado."""

        # Obtém uma representação da lista de atributos para construção da árvore
        attribute_list = range(X.shape[1])

        # Inicializa a raiz da árvore
        self.tree = None

        # Treina a árvore de decisão recursivamente
        self.fit_tree(X, y, self.tree, attribute_list)

        return self

    def predict(self, X):
        pass

    def fit_tree(self, X, y, tree, attribute_list):
        """Treina uma árvore de decisão criando os nós que a compõe."""
        samples_size, self.input_size = X.shape

        # Cria um nó de árvore utilizando um dict() como estrutura
        node = dict([('decision', None), ('partitions', [])])

        classes = numpy.unique(y)

        # Se os rótulos em y são todos da mesma classe C, então a decisão é C
        if len(classes) == 1:
            node['decision'] = classes[0]
            return node

        # Se a lista de atributos está vazia, então a decisão é a classe majoritária de y
        if len(attribute_list) == 0:
            unique, pos = numpy.unique(y, return_inverse=True)
            counts = numpy.bincount(pos)
            node['decision'] = unique[counts.argmax()]
            return node

        # Realiza seleção de atributo com base no critério definido no construtor
        if self.attribute_selection_method == 'random':
            splitting_attribute, attribute_type, splitting_values = self.select_attribute_by_random(X, y, attribute_list)
        elif self.attribute_selection_method == 'gain':
            splitting_attribute, attribute_type, splitting_values = self.select_attribute_by_information_gain(X, y, attribute_list)
        elif self.attribute_selection_method == 'gini':
            splitting_attribute, attribute_type, splitting_values = self.select_attribute_by_gini(X, y, attribute_list)
        else:
            raise Exception('Método de seleção de atributo não implementado.')

        attribute_list.remove(splitting_attribute)

        if attribute_type == AttributeType.discrete:
            for spliting_value in splitting_values:
                # Obtém um conjunto de tuplas que satisfaçam o critério
                partition = numpy.where(X[:,splitting_attribute] == spliting_value)
                X_partition = X[partition]
                y_partition = y[partition]

                subtree = self.fit_tree(X_partition, y_partition, tree, attribute_list)
                node['partitions'].append(subtree)

        elif attribute_type == AttributeType.continuous:
            partition = numpy.where(X[:,splitting_attribute] <= splitting_values[0])
            X_partition = X[partition]
            y_partition = y[partition]
            subtree = self.fit_tree(X_partition, y_partition, tree, attribute_list)
            node['partitions'].append(subtree)

            for idx in xrange(len(splitting_values)-1):
                lbound = splitting_values[idx]
                ubound = splitting_values[idx+1]
                partition = numpy.where(numpy.logical_and(X[:,splitting_attribute] > lbound, X[:,splitting_attribute] <= ubound))
                X_partition = X[partition]
                y_partition = y[partition]

                subtree = self.fit_tree(X_partition, y_partition, tree, attribute_list)
                node['partitions'].append(subtree)

            partition = numpy.where(X[:,splitting_attribute] > splitting_values[-1])
            X_partition = X[partition]
            y_partition = y[partition]
            subtree = self.fit_tree(X_partition, y_partition, tree, attribute_list)
            node['partitions'].append(subtree)

        else:
            raise Exception('Tipo de atributo inexistente.')

        return node

    def select_attribute_by_random(self, X, y, attribute_list):
        best_attribute = numpy.random.choice(attribute_list)
        attribute_type = self.get_attribute_type(X, best_attribute)
        splitting_values = numpy.median(X[:,best_attribute])

        return best_attribute, attribute_type, [splitting_values]

    def select_attribute_by_information_gain(self, X, y, attribute_list):
        """Seleciona um atributo que crie a melhor partição no conjunto de dados em relação à entropia."""
        pass

    def select_attribute_by_gini(self, X, y, attribute_list):
        """Seleciona um atributo que crie a melhor partição no conjunto de dados em relação ao Gini Index."""
        pass

    def get_attribute_type(self, X, attribute):
        if isinstance(X[:,attribute][-1], float):
            return AttributeType.continuous
        else:
            return AttributeType.discrete

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class AttributeType(enum.Enum):
    discrete = 1
    continuous = 2