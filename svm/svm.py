# -*- coding: utf-8 -*-

import numpy, random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize

class SVM(BaseEstimator, ClassifierMixin):
    """Classe que implementa um SVM."""

    def __init__(self, C, kernel='linear', gamma='auto'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.eps = 1e-3
        self.tol = 1e-3 # ksi
        self.b = 0

    def fit(self, X, y):
        """Treina o SVM utilizando o algoritmo SMO. 
           Busca na região factível do problema dual e maximiza a função objetivo. 
           A solução ótima pode ser checada utilizando as condições de KKT."""
        (samples_size, input_size) = X.shape

        self.classes = numpy.unique(y)
        if self.classes.size != 2:
            raise Exception('Multiclass SVM not implemented')

        self.X = normalize(X)
        self.y = y

        self.partition_1 = numpy.where(y == self.classes[0])
        self.partition_2 = numpy.where(y == self.classes[1])
        self.y[self.partition_1] = 1
        self.y[self.partition_2] = -1

        # Inicializa multiplicadores de Lagrange para cada amostra
        self.alph = numpy.zeros((X.shape[0],))

        samples_violating_kkt_conditions = 0
        loop_all_samples = True

        # Alterna entre todas amostras e amostras non-bound (0 < alpha < C)
        # até que todas amostras obedeçam as condições de KKT por um epsilon  
        while samples_violating_kkt_conditions > 0 or loop_all_samples:
            
            partition = range(self.alph.size)
            if not loop_all_samples:
                partition = numpy.where(numpy.logical_and(self.alph > 0, self.alph < self.C))[0]

            for sample in partition:
                samples_violating_kkt_conditions += self.examine_sample(sample)

            if loop_all_samples:
                loop_all_samples = False
            elif samples_violating_kkt_conditions == 0:
                loop_all_samples = True

        return self

    def examine_sample(self, sample2):
        """Retorna 1 caso otimização foi possível, do contrário retorna 0."""
        y2 = self.y[sample2]
        alph2 = self.alph[sample2]
        E2 = self.f(self.X[sample2,:]) - y2 # TODO: buscar E2 na cache
        r2 = E2*y2
        print("examine_sample %d, y = %d, alph2 = %f" % (sample2, y2, alph2))

        if (r2 < -self.tol and alph2 < self.C) or (r2 > self.tol and alph2 > 0):
            # Há 3 estratégias para selecionar a segunda amostra:
            # 1) Escolhe a segunda amostra que maximize o passe de otimização, aproximado por |E1-E2|

            # 2) Escolhe amostra aleatória dentre as non-bound (0 < alpha < C)

            # 3) Escolhe amostra aleatória dentre todas
            sample_list = list(range(self.X.shape[0]))
            random.shuffle(sample_list)
            for sample1 in sample_list:
                self.optimize_samples(sample1, sample2)
            return 1

        return 0

    def optimize_samples(self, sample1, sample2):
        """"""
        if sample1 == sample2:
            return 0

        y1 = self.y[sample1]
        E1 = self.f(self.X[sample1,:]) - y1 # TODO: buscar E1 na cache
        alph1 = self.alph[sample1]
        y2 = self.y[sample2]
        E2 = self.f(self.X[sample2,:]) - y2 # TODO: buscar E2 na cache
        alph2 = self.alph[sample2]
        s = y1*y2
        lower_bound, upper_bound = self.compute_bounds(sample1, sample2)
        if lower_bound == upper_bound:
            return 0

        k11 = self.kernel_func(self.X[sample1,:], self.X[sample1,:])
        k12 = self.kernel_func(self.X[sample1,:], self.X[sample2,:])
        k22 = self.kernel_func(self.X[sample2,:], self.X[sample2,:])
        eta = k11 + k22 -2*k12

        if eta > 0:
            a2 = alph2 + y2*(E1-E2)/eta
            if a2 < lower_bound:
                a2 = lower_bound
            elif a2 > upper_bound:
                a2 = upper_bound
        else:
            # objective function at a2=lower_bound
            self.alph[sample2] = lower_bound
            l_fx = self.f(self.X[sample2])

            # objective function at a2=upper_bound
            self.alph[sample2] = upper_bound
            u_fx = self.f(self.X[sample2])

            if l_fx < u_fx - self.eps:
                a2 = lower_bound
            elif l_fx > u_fx + self.eps:
                a2 = upper_bound
            else:
                a2 = alph2

        if numpy.abs(a2-alph2) < self.eps*(a2 + alph2 + self.eps):
            return 0

        a1 = alph1 + s*(alph2 - a2)
        
        # Atualiza o parâmetro b
        self.b = self.threshold(sample1, sample2, a1, a2)

        # Update weight vector to reflect change in a1 & a2, if SVM is linear
        # Update error cache using new Lagrange multipliers
        
        self.alph[sample1] = a1
        self.alph[sample2] = a2
        return 1

    def f(self, X):
        """Calcula a saída do classificador para uma dada amostra X."""
        return sum(numpy.multiply(numpy.multiply(self.y, self.alph), numpy.apply_along_axis(self.kernel_func, 1, self.X, X)) - self.b)

    def threshold(self, sample1, sample2, a1, a2):
        """Calcula o parâmetro b."""
        y1 = self.y[sample1]
        X1 = self.X[sample1,:]
        alpha1 = self.alph[sample1]
        E1 = self.f(self.X[sample1,:]) - y1 # TODO: buscar E1 na cache
        y2 = self.y[sample2]
        X2 = self.X[sample2,:]
        alpha2 = self.alph[sample2]
        E2 = self.f(self.X[sample2,:]) - y2 # TODO: buscar E1 na cache
        
        b1 = E1 + y1*(a1 - alpha1)*self.kernel_func(X1, X1) + y2*(a2 - alpha2)*self.kernel_func(X1, X2) + self.b

        if 0 < a1 < self.C:
            return b1

        b2 = E2 + y1*(a1 - alpha1)*self.kernel_func(X1, X2) + y2*(a2 - alpha2)*self.kernel_func(X2, X2) + self.b

        if 0 < a2 < self.C:
            return b2

        return numpy.mean([b1, b2])

    def kernel_func(self, Xj, X):
        """Calcula saída da função kernel."""
        if self.kernel == 'linear':
            return numpy.dot(Xj, X)
        else:
            raise Exception('Kernel %s not implemented' % self.kernel)

    def compute_bounds(self, sample1, sample2):
        y1 = self.y[sample1]
        alpha1 = self.alph[sample1]
        y2 = self.y[sample2]
        alpha2 = self.alph[sample2]

        if y1 == y2:
            lower_bound = max(0, alpha2 + alpha1 - self.C)
            upper_bound = min(self.C, alpha2 + alpha1)
        else:
            lower_bound = max(0, alpha2 - alpha1)
            upper_bound = min(self.C, self.C + alpha2-alpha1)

        return lower_bound, upper_bound

    def predict(self, X):
        """Rotula amostras utilizando o SVM previamente treinado."""
        y = []
        for x_row in X:
            y_row = self.classes[0] if self.f(x_row) >= 1-self.eps else self.classes[1]
            y.append(y_row)

        y = numpy.asarray(y)
        return y