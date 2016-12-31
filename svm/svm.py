# -*- coding: utf-8 -*-

import numpy, random, time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

class SVM(BaseEstimator, ClassifierMixin):
    """Classe que implementa um SVM."""

    def __init__(self, C, kernel='linear', gamma=0, degree=2, coef0=0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = 1e-3 # ksi
        self.max_iterations = 10000
        self.classifiers = []
        #TODO; Receber porcentagem que deve ser utilizada para validação e parada de treinamento (padrão: 0.75)

    def fit(self, X, y):
        """Treina o SVM utilizando o algoritmo SMO. 
           Busca na região factível do problema dual e maximiza a função objetivo. 
           A solução ótima pode ser checada utilizando as condições de KKT."""
        self.classes = numpy.unique(y)
        if self.gamma == 0:
            self.gamma = 1.0/float(X.shape[1])

        self.scaler = MinMaxScaler((-1,1))
        self.X = self.scaler.fit_transform(X)

        # Utiliza estratégia one-vs-all para tratar problemas multiclasses
        if self.classes.size > 2:
            self.classifiers = []
            for clazz in self.classes:
                X_binary, y_binary = numpy.array(self.X, copy=True), numpy.array(y, copy=True)
                partition_clazz, partition_not_clazz = numpy.where(y_binary==clazz), numpy.where(y_binary!=clazz)
                y_binary[partition_not_clazz], y_binary[partition_clazz] = -1, 1
                clf = SVM(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
                clf.fit(X_binary, y_binary)
                clf.negative_class, clf.positive_class = 'not clazz', clazz
                self.classifiers.append(clf)
            return self

        if numpy.array_equal(self.classes, numpy.array([-1,1])):
            self.y = y[:,0]
            self.negative_class, self.positive_class = -1, 1
        else:
            self.binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
            self.y = self.binarizer.fit_transform(y)[:,0]
            self.negative_class, self.positive_class = self.binarizer.inverse_transform(numpy.array([-1]))[0], self.binarizer.inverse_transform(numpy.array([1]))[0]

        self.alpha = numpy.zeros((self.X.shape[0],))
        self.error = numpy.zeros((self.X.shape[0],))
        self.b = 0

        # Calcula valores do kernel para diminuir retrabalho
        self.K = numpy.zeros((self.X.shape[0],self.X.shape[0]))
        for sample1 in range(self.X.shape[0]):
            for sample2 in range(self.X.shape[0]):
                self.K[sample1,sample2] = self.kernel_func(self.X[sample1,:], self.X[sample2,:])

        num_changed, examine_all = 0, True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for sample in range(self.alpha.size):
                    num_changed += self.examine_example(sample)
            else:
                partition = numpy.where((self.error > self.tol) & (self.error < self.C - self.tol))[0]
                for sample in partition:
                    num_changed += self.examine_example(sample)
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
        S = 0
        for sample in range(self.X.shape[0]):
            if self.f(sample)*self.y[sample] < 0.0:
                S += 1
        S =  float(S) / self.X.shape[0] * 100.0
        print("Erro de treinamento %f%%" % S)
        return self

    def examine_example(self, sample2):
        X2, y2, alpha2 = self.X[sample2,:], self.y[sample2], self.alpha[sample2]
        if alpha2 < self.tol or alpha2 > self.C - self.tol:
            E2 = self.f(sample2) - y2
        else:
            E2 = self.error[sample2]
        r2 = E2 * y2

        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            partition = numpy.where((self.error > self.tol) & (self.error < self.C - self.tol))[0]
            if partition.size > 0 and E2 > 0:
                sample1 = numpy.argmax(self.error)
                if self.take_step(sample1, sample2, E2):
                    return 1
            elif partition.size > 0 and E2 < 0:
                sample1 = numpy.argmin(self.error)
                if self.take_step(sample1, sample2, E2):
                    return 1

            if partition.size > 0:
                partition = numpy.random.permutation(partition)
                for sample1 in partition:
                    if self.take_step(sample1, sample2, E2):
                        return 1

            partition = numpy.random.permutation(self.alpha.size)
            for sample1 in partition:
                if self.take_step(sample1, sample2, E2):
                    return 1

        return 0

    def take_step(self, sample1, sample2, E2):
        if sample1 == sample2: return False

        X1, y1, alpha1 = self.X[sample1,:], self.y[sample1], self.alpha[sample1]
        X2, y2, alpha2 = self.X[sample2,:], self.y[sample2], self.alpha[sample2]

        if alpha1 < self.tol or alpha1 > self.C - self.tol:
            E1 = self.f(sample1) - y1
        else:
            E1 = self.error[sample1]

        s = y1 * y2

        if s > 0:
            L = max(0, alpha2 + alpha1 - self.C);
            H = min(self.C, alpha1 + alpha2);
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)

        if L == H: return False

        k11 = self.K[sample1,sample1]
        k12 = self.K[sample1,sample2]
        k22 = self.K[sample2,sample2]

        eta = 2 * k12 - k11 - k22
        gamma = self.alpha[sample1] + s * self.alpha[sample2]
        if eta < 0:
            a2 = alpha2 - y2 * (E1-E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            Lobj = - s * L + L -0.5 * k11 * (gamma - s * L)**2 - 0.5 * k22 *L**2 - s * k12 * (gamma - s * L) * L -y1 * (gamma - s * L) * (self.f(sample1) + self.b - y1 * alpha1 * k11 - y2 * alpha2 * self.K[sample2,sample1]) - y2 * L * (self.f(sample2) + self.b - y1 * alpha1 * k12 - y2 * alpha2 * k22)
            Hobj = - s * H + H -0.5 * k11 * (gamma - s * H)**2 - 0.5 * k22 *H**2 - s * k12 * (gamma - s * H) * H -y1 * (gamma - s * H) * (self.f(sample1) + self.b - y1 * alpha1 * k11 - y2 * alpha2 * self.K[sample2,sample1]) - y2 * H * (self.f(sample2) + self.b - y1 * alpha1 * k12 - y2 * alpha2 * self.K[sample2,sample1])
            if Lobj < Hobj - 1e-3:
                a2 = L
            elif Lobj > Hobj + 1e-3:
                a2 = H
            else:
                a2 = alpha2

        if a2 < 1e-8:
            a2 = 0
        elif a2 > self.C-1e-8:
            a2 = self.C

        if abs(a2 - alpha2) < 1e-3 * (a2 + alpha2 + 1e-3): return False

        a1 = alpha1 + s * (alpha2 - a2)

        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b
        b_old = self.b
        self.b = (b1 + b2) / 2.0

        self.error += y1 * numpy.multiply((a1 - alpha1), self.K[sample1,:]) + y2 * numpy.multiply((a2 - alpha2), self.K[sample2,:]) + b_old - self.b
        self.error[sample1], self.error[sample2] = 0, 0
        self.alpha[sample1] = a1
        self.alpha[sample2] = a2
        return True

    def predict(self, X):
        """Rotula amostras utilizando o SVM previamente treinado."""
        X = self.scaler.transform(X)
        y = []
        for sample1 in range(X.shape[0]):
            if self.classes.size > 2:
                clf = max(self.classifiers, key=lambda clf: clf.separation_margin(X[sample1,:]))
                y.append(clf.positive_class)
            else:
                K = numpy.zeros((X.shape[0],self.X.shape[0]))
                for sample2 in range(self.X.shape[0]):
                    K[sample1,sample2] = self.kernel_func(X[sample1,:], self.X[sample2,:])
                func_value = sum(numpy.multiply(numpy.multiply(self.y.T, self.alpha.T), K[sample1,:])) - self.b

                clazz = self.positive_class if func_value >= 0 else self.negative_class
                y.append(clazz)
        return numpy.asarray(y)

    def f(self, sample):
        return sum(numpy.multiply(numpy.multiply(self.y.T, self.alpha.T), self.K[sample,:])) - self.b

    def kernel_func(self, X1, X2):
        """Calcula saída da função kernel."""
        if self.kernel == 'linear':
            return numpy.sum(numpy.multiply(X1, X2))
            #return numpy.dot(Xj, X.T)
        elif self.kernel == 'polynomial':
            return numpy.sum(numpy.power(numpy.multiply(self.gamma, numpy.multiply(X1, X2)) + self.coef0, self.degree))
        elif self.kernel == 'rbf':
            return numpy.exp(-self.gamma * numpy.power(numpy.linalg.norm(X1 - X2), 2))
        else:
            raise Exception('Kernel %s not implemented' % self.kernel)

    def separation_margin(self, X):
        """Retorna a distância da amostra ao hiperplano de separação."""
        K = numpy.zeros((1,self.X.shape[0]))
        for sample2 in range(self.X.shape[0]):
            K[0,sample2] = self.kernel_func(X, self.X[sample2,:])
        return sum(numpy.multiply(numpy.multiply(self.y.T, self.alpha.T), K[0,:])) - self.b

