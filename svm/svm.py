# -*- coding: utf-8 -*-

import numpy, random, time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

class SVM(BaseEstimator, ClassifierMixin):
    """Classe que implementa um SVM."""

    def __init__(self, C, kernel='rbf', gamma=0, degree=2, coef0=0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma # se gamma == 0, gamma = 1/input_size
        self.degree = degree
        self.coef0 = coef0
        self.tol = 1e-3 # ksi
        self.max_iterations = 10000
        self.classifiers = []

    def fit(self, X, y):
        """Treina o SVM utilizando o algoritmo SMO. 
           Busca na região factível do problema dual e maximiza a função objetivo. 
           A solução ótima pode ser checada utilizando as condições de KKT."""
        (samples_size, input_size) = X.shape

        self.classes = numpy.unique(y)
        
        # Utiliza estratégia one-vs-all para tratar problemas multiclasses
        if self.classes.size > 2:
            self.classifiers = []
            for clazz in self.classes:
                partition_not_clazz = numpy.where(y!=clazz)
                X_binary, y_binary = numpy.array(X, copy=True), numpy.array(y, copy=True)
                y_binary[partition_not_clazz] = 'not_clazz'
                clf = SVM(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
                clf.fit(X_binary, y_binary)
                self.classifiers.append((clazz, clf))
            return self

        if self.gamma == 0:
            self.gamma = 1/float(input_size)

        scaler = MinMaxScaler((-1,1))
        self.X = scaler.fit_transform(X)
        self.y = label_binarize(y, classes=numpy.unique(y), neg_label=-1, pos_label=1)[:,0]

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
        return self

    def predict(self, X):
        """Rotula amostras utilizando o SVM previamente treinado."""
        y = []
        for X_row in X:
            if self.classes.size > 2:
                clazz, clf = max(self.classifiers, key=lambda (clazz, clf): clf.confidence_score(X_row))
                y.append(clazz)
            else:
                clazz = self.classes[1] if self.f(X_row) >= 0 else self.classes[0]
                y.append(clazz)
        return numpy.asarray(y)

    def confidence_score(self, X_sample):
        """Retorna a distância da amostra ao hiperplano de separação."""
        return self.f(X_sample)

    def try_heuristic_1(self, sample1, E1):
        """Heurística que escolhe sample2 maximizando |E1-E2|."""
        X1, y1 = self.X[sample1,:], self.y[sample1]
        nonbound = filter(lambda sample: 0 < self.alph[sample] < self.C, range(self.alph.size))
        if len(nonbound) == 0:
            return 0
        sample2 = max(nonbound, key=lambda sample2: abs(E1 - self.error_cache[sample2]))
        E2 = self.error_cache[sample2]
        if self.try_optimization(sample1, sample2, E1, E2):
            print("Heurística 1: %s, %s" % (self.X[sample1,:], self.X[sample2,:]))
            time.sleep(5)
            return 1

        #sample2, sample2_step = -1, -1
        #for s2 in range(self.alph.size):
        #    X2, y2, alph2 = self.X[s2,:], self.y[s2], self.alph[s2]
        #    if 0 < alph2 < self.C:
        #        E2 = self.error_cache[sample2]
        #        step = abs(E1 - E2)
        #        if step > sample2_step:
        #            sample2_step, sample2 = step, s2
        #            if self.try_optimization(sample1, sample2, E1, E2):
        #                print("Heurística 1")
        #                return 1
        return 0

    def try_heuristic_2(self, sample1, E1):
        """Heurística que escolhe sample2 a partir de amostras non-bound."""
        sample_list = list(range(self.alph.size))
        random.shuffle(sample_list)
        for sample2 in sample_list:
            X2, y2, alph2 = self.X[sample2,:], self.y[sample2], self.alph[sample2]
            if 0 < alph2 < self.C:
                E2 = self.error_cache[sample2]
                if self.try_optimization(sample1, sample2, E1, E2):
                    print("Heurística 2: %s, %s" % (self.X[sample1,:], self.X[sample2,:]))
                    time.sleep(5)
                    return 1
        return 0

    def try_heuristic_3(self, sample1, E1):
        """Heurística que escolhe sample2 aleatoriamente do conjunto de dados."""
        sample_list = list(range(self.alph.size))
        random.shuffle(sample_list)
        for sample2 in sample_list:
            X2, y2, alph2 = self.X[sample2,:], self.y[sample2], self.alph[sample2]
            E2 = self.error_cache[sample2] if 0 < alph2 < self.C else self.f(X2) - y2
            if self.try_optimization(sample1, sample2, E1, E2):
                print("Heurística 3: %s, %s" % (self.X[sample1,:], self.X[sample2,:]))
                time.sleep(5)
                return 1
        return 0

    def try_optimization(self, sample1, sample2, E1, E2):
        """Tenta otimizar 2 multiplicadores de Lagrange."""
        if sample1 == sample2: return False

        X1, y1, alph1 = self.X[sample1,:], self.y[sample1], self.alph[sample1]
        X2, y2, alph2 = self.X[sample2,:], self.y[sample2], self.alph[sample2]

        alph1_prev, alph2_prev = self.alph[sample1], self.alph[sample2]

        L, U = self.compute_optimization_bounds(y1, y2, alph1, alph2)
        if abs(L - U) < 1e-5: return False

        k11 = self.kernel_func(X1, X1)
        k12 = self.kernel_func(X1, X2)
        k22 = self.kernel_func(X2, X2)
        eta = 2.0 * k12 - k11 - k22
        if eta < 0:
            # Calcula novo valor de alph2, e alph1
            alph2 += float(y2 * (E1 - E2)) / eta
            alph2 = max(alph2, L)
            alph2 = min(alph2, U)
        else:
            c1 = eta / 2.0
            c2 = y2 * (E1 - E2) - eta * _alpha2
            low_obj = c1 * L * L + c2 * L
            high_obj = c1 * U * U + c2 * U
            if low_obj > high_obj + 1e-5:
                alph2 = L
            elif low_obj < high_obj - 1e-5:
                alph2 = U
            else:
                alph2 = alph2_prev

        if abs(alph2 - alph2_prev) < 1e-5 * (alph2 + alph2_prev + 1e-5): return False
        alph1 -= float(y1 * y2) * (alph2_prev - alph2)
        if alph1 < 0:
            alph2 += s * alph1
            alph1 = 0
        elif alph1 > self.C:
            t = alph1 - self.C
            alph2 += float(y1 * y2) * t
            alph1 = self.C

        # Atualiza threshold
        b1 = self.b - E1 - y1 * (alph1 - alph1_prev) * k11 + y2 * (alph2 - alph2_prev) * k12
        b2 = self.b + E2 + y1 * (alph1 - alph1_prev) * k12 + y2 * (alph2 - alph2_prev) * k22
        b = b1 if 0 < alph1_prev < self.C else b2 if 0 < alph2_prev < self.C else (b1 + b2) / 2.0
        delta_b = b - self.b
        self.b = b

        t1 = y1 * (alph1 - alph1_prev)
        t2 = y2 * (alph2 - alph2_prev)

        self.w += t1 * X1 + t2 * X2

        # Atualiza cache de erro
        for sample in range(self.alph.size):
            if 0 < self.alph[sample] < self.C:
                self.error_cache[sample] += t1 * self.kernel_func(X1, self.X[sample,:]) + t2 * self.kernel_func(X2, self.X[sample,:]) - delta_b
        self.error_cache[sample1], self.error_cache[sample2] = 0.0, 0.0
        self.alph[sample1], self.alph[sample2] = alph1, alph2

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
            return value - self.b

    def kernel_func(self, Xj, X):
        """Calcula saída da função kernel."""
        if self.kernel == 'linear':
            return numpy.dot(Xj, X.T)
        elif self.kernel == 'polynomial':
            return numpy.power(self.gamma * numpy.dot(Xj, X.T) + self.coef0, self.degree)
        elif self.kernel == 'rbf':
            return numpy.exp(-self.gamma * numpy.power(numpy.linalg.norm(Xj - X), 2))
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
        return accuracy_score(y_true, y_pred)