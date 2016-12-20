# -*- coding: utf-8 -*-

import numpy, random, time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

class SVM(BaseEstimator, ClassifierMixin):
    """Classe que implementa um SVM."""

    def __init__(self, C, kernel='rbf', gamma='auto'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = 1e-3 # ksi
        self.max_passes = 2

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

        # Inicializa multiplicadores de Lagrange para cada amostra
        self.alph = numpy.zeros((X.shape[0],))

        # Inicializa Threshold
        self.b = 0

        # Inicializa cache
        error_cache = dict()

        passes = 0
        while passes < self.max_passes:
            print("===============")
            print("OUTER ITERATION")
            print("===============")
            changed_alphas = 0

            for sample1 in range(self.alph.size):
                X1, y1, alph1 = self.sample_data(sample1)

                # Calcula E1 = f(x) - y
                if sample1 not in error_cache:
                    error_cache[sample1] = self.f(X1) - y1
                E1 = error_cache[sample1]

                r1 = E1*y1
                if (r1 < -self.tol and alph1 < self.C) or (r1 > self.tol and alph1 > 0):
                    # Usa heurística que escolhe sample2 que maximiza |E1-E2|
                    sample2 = -1
                    sample2_step = -1
                    for s2 in range(self.alph.size):
                        X2, y2, alph2 = self.sample_data(s2)

                        # Calcula E2 = f(x) - y
                        if s2 not in error_cache:
                            error_cache[s2] = self.f(X2) - y2
                        E2 = error_cache[s2]
                        step = abs(E1-E2)

                        if step > sample2_step:
                            sample2_step = step
                            sample2 = s2

                    # Select sample2 randomly, different from sample1
                    #sample2 = numpy.random.choice(self.alph.size)
                    #sample2 = (sample2 + 1) % self.alph.size if sample1 == sample2 else sample2
                    #X2, y2, alph2 = self.sample_data(sample2)

                    # Calcula E2 = f(x) - y
                    if sample2 not in error_cache:
                        error_cache[sample2] = self.f(X2) - y2

                    E2 = error_cache[sample2]

                    alph1_old = alph1
                    alph2_old = alph2

                    if y1 == y2:
                        L = max(0, alph1 + alph2 - self.C)
                        U = min(self.C, alph1 + alph2)
                    else:
                        L = max(0, alph2 - alph1)
                        U = min(self.C, self.C + alph2 - alph1)

                    if L == U: continue

                    k11 = self.kernel_func(X1, X1)
                    k12 = self.kernel_func(X1, X2)
                    k22 = self.kernel_func(X2, X2)
                    eta = 2*k12 - k11 - k22 

                    if eta >= 0: continue

                    alph2 -= y2*(E1-E2)/eta
                    alph2 = U if alph2 > U else L if alph2 < L else alph2
                    self.alph[sample2] = alph2

                    if abs(alph2 - alph2_old) < 1e-5: continue

                    alph1 += y1*y2*(alph2_old - alph2) # -y1y2 delta alph2
                    self.alph[sample1] = alph1

                    b1 = self.b - E1 - y1*(alph1 - alph1_old)*k11 - y2*(alph2 - alph2_old)*k12
                    b2 = self.b - E2 - y1*(alph1 - alph1_old)*k12 - y2*(alph2 - alph2_old)*k22
                    self.b = b1 if 0 < alph1 < self.C else b2 if 0 < alph2 < self.C else (b1+b2)/2

                    error_cache[sample1] = self.f(X1) - y1
                    error_cache[sample2] = self.f(X2) - y1

                    changed_alphas += 1

            if changed_alphas == 0:
                passes += 1
            else:
                passes = 0

            print("===============")
            print("changed_alphas %d" % changed_alphas)
            print("===============")

        print("===============")
        print("alphas %s" % self.alph)
        print("===============")

    def f(self, X):
        """Calcula a saída do classificador para uma dada amostra X."""
        value = self.b
        for sample_i in range(self.alph.size):
            X_i, y_i, alph_i = self.sample_data(sample_i)
            if alph_i > 0:
                value += alph_i * y_i * self.kernel_func(X_i, X)
        #print("value = %f, b = %f" % (value, self.b))
        return value
        #return sum(numpy.multiply(numpy.multiply(self.y, self.alph), numpy.apply_along_axis(self.kernel_func, 1, self.X, X)) + self.b)

    def kernel_func(self, Xj, X):
        """Calcula saída da função kernel."""
        if self.kernel == 'linear':
            return numpy.dot(Xj.T, X)
        elif self.kernel == 'rbf':
            s = numpy.dot(Xj.T, Xj) + numpy.dot(X.T, X) - 2*numpy.dot(Xj.T, X)
            return numpy.exp(-s/2)
        else:
            raise Exception('Kernel %s not implemented' % self.kernel)

    def sample_data(self, sample):
        X = self.X[sample,:]
        y = self.y[sample]
        alph = self.alph[sample]
        return X, y, alph

    def predict(self, X):
        """Rotula amostras utilizando o SVM previamente treinado."""
        y = []
        for x_row in X:
            y_row = self.classes[0] if self.f(x_row) >= 1-self.tol else self.classes[1]
            y.append(y_row)

        y = numpy.asarray(y)
        return y

    def accuracy(self):
        func = lambda prediction: self.classes[0] if prediction == 1 else self.classes[1]
        y_true = list(map(func, self.y))
        y_pred = numpy.apply_along_axis(self.f, 1, self.X)
        y_pred = list(map(int, numpy.sign(y_pred)))
        y_pred = list(map(func, y_pred))
        #print("y_true = %s" % y_true)
        #print("y_pred = %s" % y_pred)
        return accuracy_score(y_true, y_pred)