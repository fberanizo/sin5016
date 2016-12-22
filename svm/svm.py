# -*- coding: utf-8 -*-

import numpy, random, time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

class SVM(BaseEstimator, ClassifierMixin):
    """Classe que implementa um SVM."""

    def __init__(self, C, kernel='rbf', gamma=0, degree=2):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma # se gamma == 0, gamma = 1/samples_size
        self.degree = degree
        self.tol = 1e-3 # ksi
        self.max_iterations = 10000

    def fit(self, X, y):
        """Treina o SVM utilizando o algoritmo SMO. 
           Busca na região factível do problema dual e maximiza a função objetivo. 
           A solução ótima pode ser checada utilizando as condições de KKT."""
        (samples_size, input_size) = X.shape

        self.classes = numpy.unique(y)
        if self.classes.size != 2:
            raise Exception('Multiclass SVM not implemented')

        if self.gamma == 0:
            self.gamma = float(samples_size)

        scaler = MinMaxScaler((-1,1))
        self.X = scaler.fit_transform(X)
        self.y = y

        # Inicializa multiplicadores de Lagrange
        self.alph = numpy.zeros((X.shape[0],))  
        # Inicializa threshold
        self.b = 0.0
        # Inicializa vetor w
        self.w = numpy.zeros((X.shape[1],))
        # Inicializa cache de erro
        self.error_cache = numpy.zeros((X.shape[0],))

        iterations_without_improvements = 0
        remaining_iterations = self.max_iterations
        # Repete até que não haja mudanças nos multiplicadores por 2 iterações
        while iterations_without_improvements < 2 and remaining_iterations > 0:
            changed_alphas = 0
            #print("===============")
            #print("Starting iteration")
            #print("===============")

            for sample1 in range(self.alph.size):
                X1, y1, alph1 = self.X[sample1,:], self.y[sample1], self.alph[sample1]
                E1 = self.error_cache[sample1] if 0 < alph1 < self.C else self.f(X1) - y1

                r1 = E1*y1
                if (r1 < -self.tol and alph1 < self.C) or (r1 > self.tol and alph1 > 0):
                    # Tenta heurística 1: sample2 que maximiza |E1-E2|
                    changed_alphas += self.try_heuristic_1(sample1, E1)
                    # Tenta heurística 2: sample2 de amostras non-bound
                    if changed_alphas == 0:
                        changed_alphas += self.try_heuristic_2(sample1, E1)
                    # Tenta heurística 3: sample2 aleatório
                    if changed_alphas == 0:
                        changed_alphas += self.try_heuristic_3(sample1, E1)

            if changed_alphas == 0:
                iterations_without_improvements += 1
            else:
                iterations_without_improvements = 0

            remaining_iterations -= 1

            print("===============")
            print("remaining_iterations %d" % remaining_iterations)
            print("===============")

        # Exibe vetores suporte
        alpha_idx = numpy.where(self.alph > 0)[0]
        support_vectors = self.X[alpha_idx, :]

        print("===============")
        print("support_vectors %s" % support_vectors)
        print("===============")
        return self

    def predict(self, X):
        """Rotula amostras utilizando o SVM previamente treinado."""
        y = []
        for x_row in X:
            y_row = self.classes[1] if self.f(x_row) >= 0 else self.classes[0]
            y.append(y_row)
        y = numpy.asarray(y)
        return y

    def try_heuristic_1(self, sample1, E1):
        """Heurística que escolhe sample2 maximizando |E1-E2|."""
        X1, y1 = self.X[sample1,:], self.y[sample1]
        sample2, sample2_step = -1, -1
        for s2 in range(self.alph.size):
            X2, y2, alph2 = self.X[s2,:], self.y[s2], self.alph[s2]
            if 0 < alph2 < self.C:
                E2 = self.error_cache[sample2]
                step = abs(E1 - E2)
                if step > sample2_step:
                    sample2_step, sample2 = step, s2
                    if self.try_optimization(sample1, sample2, E1, E2):
                        return 1
        return 0

    def try_heuristic_2(self, sample1, E1):
        """Heurística que escolhe sample2 a partir de amostras non-bound."""
        return 0

    def try_heuristic_3(self, sample1, E1):
        """Heurística que escolhe sample2 aleatoriamente do conjunto de dados."""
        sample_list = list(range(self.alph.size))
        random.shuffle(sample_list)
        for sample2 in sample_list:
            X2, y2, alph2 = self.X[sample2,:], self.y[sample2], self.alph[sample2]
            E2 = self.error_cache[sample2] if 0 < alph2 < self.C else self.f(X2) - y2
            if self.try_optimization(sample1, sample2, E1, E2):
                return 1
        return 0

    def try_optimization(self, sample1, sample2, E1, E2):
        """Tenta otimizar 2 multiplicadores de Lagrange"""
        if sample1 == sample2: return False

        X1, y1, alph1 = self.X[sample1,:], self.y[sample1], self.alph[sample1]
        X2, y2, alph2 = self.X[sample2,:], self.y[sample2], self.alph[sample2]

        alpha_prev = self.alph.copy()

        L, U = self.compute_optimization_bounds(y1, y2, alph1, alph2)
        if L == U: return False

        k11 = self.kernel_func(X1, X1)
        k12 = self.kernel_func(X1, X2)
        k22 = self.kernel_func(X2, X2)
        eta = 2.0 * k12 - k11 - k22
        if eta >= 0: return False
        
        if sample2 not in self.error_cache: self.error_cache[sample2] = self.f(X2) - y2
        E2 = self.error_cache[sample2]

        # Calcula novo valor de alph2, e alph1
        alph2 -= float(y2 * (E1 - E2)) / eta
        alph2 = max(alph2, L)
        alph2 = min(alph2 , U)
        if abs(alph2 - self.alph[sample2]) < 1e-3 * (alph2 + self.alph[sample2] + 1e-3): return False
        alph1 += float(y1 * y2) * (alpha_prev[sample2] - alph2)

        # Atualiza threshold
        b1 = self.b - E1 - y1 * (alph1 - alpha_prev[sample1]) * k11 + y2 * (alph2 - alpha_prev[sample2]) * k12
        b2 = self.b + E2 + y1 * (alph1 - alpha_prev[sample1]) * k12 + y2 * (alph2 - alpha_prev[sample2]) * k22
        b = b1 if 0 < self.alph[sample1] < self.C else b2 if 0 < self.alph[sample2] < self.C else (b1 + b2) / 2.0
        delta_b = b - self.b
        self.b = b

        t1 = y1 * (alph1 - alpha_prev[sample1])
        t2 = y2 * (alph2 - alpha_prev[sample2])

        self.w += t1 * X1 + t2 * X2

        # Atualiza cache de erro
        for sample in range(self.alph.size):
            if 0 < self.alph[sample] < self.C:
                self.error_cache[sample] += t1 * self.kernel_func(X1, self.X[sample,:]) + t2 * self.kernel_func(X2, self.X[sample,:]) - delta_b
        self.error_cache[sample1] = 0.0
        self.error_cache[sample2] = 0.0

        self.alph[sample1] = alph1
        self.alph[sample2] = alph2

        return True

    def f(self, X):
        """Calcula a saída do classificador para uma dada amostra X."""
        #return numpy.sign(numpy.dot(self.w.T, X.T) + self.b).astype(int)
        if self.kernel == 'linear':
            return numpy.dot(self.w, X.T) - self.b
        else:
            value = 0.0
            for sample_i in range(self.alph.size):
                X_i, y_i, alph_i = self.X[sample_i,:], self.y[sample_i], self.alph[sample_i]
                if alph_i > 0:
                    value += alph_i * y_i * self.kernel_func(X_i, X)
            #print("value = %f, b = %f" % (value, self.b))
            return value - self.b
            #return sum(numpy.multiply(numpy.multiply(self.y, self.alph), numpy.apply_along_axis(self.kernel_func, 1, self.X, X)) + self.b)

    def kernel_func(self, Xj, X):
        """Calcula saída da função kernel."""
        if self.kernel == 'linear':
            return numpy.dot(Xj, X.T)
        elif self.kernel == 'polynomial':
            return numpy.dot(Xj, X.T) ** self.degree
        elif self.kernel == 'rbf':
            return numpy.exp(-self.gamma * numpy.linalg.norm(Xj - X) ** 2)
        else:
            raise Exception('Kernel %s not implemented' % self.kernel)

    def compute_optimization_bounds(self, y1, y2, alph1, alph2):
        """Calcula as fronteiras de otimização de um multiplicador de Lagrange."""
        if y1 == y2:
            return max(0, alph1 + alph2 - self.C), min(self.C, alph1 + alph2)
        else:
            return max(0, alph2 - alph1), min(self.C, self.C - alph1 + alph2)

    def accuracy(self):
        func = lambda prediction: self.classes[0] if prediction == 1 else self.classes[1]
        y_true = list(map(func, self.y))
        y_pred = numpy.apply_along_axis(self.f, 1, self.X)
        y_pred = list(map(int, numpy.sign(y_pred)))
        y_pred = list(map(func, y_pred))
        #print("y_true = %s" % y_true)
        #print("y_pred = %s" % y_pred)
        return accuracy_score(y_true, y_pred)