# -*- coding: utf-8 -*-
import numpy

def gain(X, y, attribute, attribute_type):
    # TODO: Criar partições
    # TODO: Para cada atributo:
    #         Calcular info(y) - info(y_particionado_por_atributo)
    pass
    

def info(X, y, attribute, attribute_type):
    # TODO: Somar (tamanho da partição/ tamanho do conjunto de dados)*entropia()
    pass

def entropy(y_partition):
    """Calcula a entropia de uma partição do conjunto de dados"""
    
    # Entropia igual a zero para conjunto vazio
    if y_partition.size == 0:
        return 0

    # Computa probabilidades de cada classe
    prob = numpy.bincount(y_partition) / y_partition.shape[0]
    classes_size = numpy.count_nonzero(prob)

    entropy = 0.

    # Realiza o somatório de -pi*log(pi)
    for i in prob:
        entropy -= i * log(i, base=classes_size)

    return entropy
