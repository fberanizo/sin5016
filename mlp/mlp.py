# -*- coding: utf-8 -*-

import numpy, hashlib, time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

class MLP(BaseEstimator, ClassifierMixin):
    """Classe que implementa um multilayer perceptron (MLP)."""
    def __init__(self, hidden_layer_size=3, max_epochs=10000, validation_size=0.25):
        self.hidden_layer_size = hidden_layer_size
        self.max_epochs = max_epochs
        self.validation_size = validation_size

    def fit(self, X, y):
        """Trains the network and returns the trained network"""
        # Normaliza valores e adiciona coluna de bias
        self.scaler = MinMaxScaler((-1,1))
        self.X = numpy.c_[self.scaler.fit_transform(X), numpy.ones(X.shape[0])]
        self.binarizer = LabelBinarizer()
        self.y = self.binarizer.fit_transform(y)

        X_train, X_validation, y_train, y_validation = train_test_split(self.X, self.y, test_size=self.validation_size)

        self.input_layer_size = X_train.shape[1]
        self.output_layer_size = y_train.shape[1]

        # Inicializa pesos da rede
        W1 = numpy.random.rand(self.hidden_layer_size, self.input_layer_size)
        W2 = numpy.random.rand(self.output_layer_size, 1 + self.hidden_layer_size)

        epoch = 1
        epochs_without_improvement = 0
        best_params = {'validation_error':1, 'W1':numpy.array(W1, copy=True), 'W2':numpy.array(W2, copy=True)}
        # Repete até que o erro de validação não melhore por 5 épocas, ou o máximo de épocas é alcançado
        while epochs_without_improvement < 5 and epoch <= self.max_epochs:
            train_error = []

            # Treinamento padrão-a-padrão
            for X, y in zip(X_train, y_train):
                X, y = numpy.array([X]), numpy.array([y])
                Y, J, dJdW1, dJdW2 = self.single_step(X, y, W1, W2)
                train_error.append(mean_squared_error(y, Y))

                # Algoritmo de gradiente conjugado
                d1, d2 = g1, g2 = -dJdW1, -dJdW2
                self.mg1, self.mg2 = numpy.mean(g1, axis=1), numpy.mean(g2, axis=1)

                iteration = 0
                while numpy.linalg.norm(self.mg1) > 1e-4 or numpy.linalg.norm(self.mg2) > 1e-4 and iteration < 200:
                    # Encontra alfa que otimiza passo
                    alpha1, alpha2 = self.bisection(X, y, W1, W2, d1, d2)
                    # Atualiza o vetor peso
                    W1 += alpha1 * d1
                    W2 += alpha2 * d2
                    # Usa backpropagation para computar o vetor gradiente 
                    Y, J, dJdW1, dJdW2 = self.single_step(X, y, W1, W2)
                    g1, g2 = -dJdW1, -dJdW2
                    mg1, mg2 = numpy.mean(g1, axis=1), numpy.mean(g2, axis=1)
                    # Usa o método de Polak-Ribiére para calcular beta
                    beta1 = max(0, numpy.dot(mg1.T, mg1 - self.mg1) / numpy.dot(self.mg1.T, self.mg1))
                    beta2 = max(0, numpy.dot(mg2.T, mg2 - self.mg2) / numpy.dot(self.mg2.T, self.mg2))
                    # Atualiza a direção conjugada
                    d1, d2 = g1 + beta1 * d1, g2 + beta2 * d2
                    # Salva valores para utilização no próximo passo
                    self.mg1, self.mg2 = mg1, mg2
                    iteration += 1

            # Calcula erro médio para época
            train_error = numpy.array(train_error).mean()

            # Calcula erro de validação
            validation_error = []
            for X, y in zip(X_validation, y_validation):
                X, y = numpy.array([X]), numpy.array([y])
                Y, J, dJdW1, dJdW2 = self.single_step(X, y, W1, W2)
                validation_error.append(mean_squared_error(y, Y))
            validation_error = numpy.array(validation_error).mean()

            if validation_error < best_params['validation_error']:
                best_params = {'validation_error':validation_error, 'W1':numpy.array(W1, copy=True), 'W2':numpy.array(W2, copy=True)}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print('Epoch: ' + str(epoch))
            print('Train Error: ' + str(train_error))
            print('Validation Error: ' + str(validation_error))

            epoch += 1

        self.W1, self.W2 = best_params['W1'], best_params['W2']
        return self

    def predict(self, X):
        """Estima classes para as entradas informadas."""
        X = numpy.c_[self.scaler.transform(X), numpy.ones(X.shape[0])]
        return [self.binarizer.classes_[numpy.argmax(self.forward(X[sample:sample+1,:], self.W1, self.W2)[0])] for sample in range(X.shape[0])]

    def single_step(self, X, y, W1, W2):
        """Executa um passo do treinamento (forward + backpropagation)."""
        Y, Yin, Z, Zin = self.forward(X, W1, W2)
        J = Y - y
        dJdW1, dJdW2 = self.backpropagate(X, y, J, Y, Yin, Z, Zin, W1, W2)

        return Y, J, dJdW1, dJdW2

    def forward(self, X, W1, W2):
        """Passa os valores de entrada pela rede e retorna a saída."""
        Zin = numpy.dot(X, W1.T)
        Z = self.logistic(Zin)
        Z = numpy.nan_to_num(Z)
        Z = numpy.c_[Z, numpy.ones(Z.shape[0])]

        Yin = numpy.dot(Z, W2.T)
        Y = self.linear(Yin)
        Y = numpy.nan_to_num(Y)
        return Y, Yin, Z, Zin

    def backpropagate(self, X, y, J, Y, Yin, Z, Zin, W1, W2):
        """Propaga erros pela rede."""
        delta2 = numpy.multiply(J, self.linear_derivative(Yin))
        dJdW2 = delta2.T.dot(Z)
        delta1 = numpy.multiply(delta2.dot(W2)[:,:-1], self.logistic_derivative(Zin))
        dJdW1 = delta1.T.dot(X)

        #delta3 = numpy.multiply(-(y-Y), self.linear_derivative(Yin))
        #dJdW2 = numpy.dot(Z.T, delta3)
        #dJdW2 = numpy.append(dJdW2, numpy.dot(numpy.ones((1, 1)), delta3), axis=0)

        #delta2 = numpy.dot(delta3, W2[:-1,:].T) * self.logistic_derivative(Zin)
        #dJdW1 = numpy.dot(X.T, delta2)
        #dJdW1 = numpy.append(dJdW1, numpy.dot(numpy.ones((1, 1)), delta2), axis=0)

        return dJdW1, dJdW2

    def logistic(self, z):
        """Aplica função de ativação logística (sigmóide)."""
        return 1 / (1 + numpy.exp(-z))

    def logistic_derivative(self, z):
        """Derivada da função logística: f'(x) = f(x).(1-f(x))."""
        logistic = self.logistic(z)
        return numpy.multiply(logistic, numpy.ones(z.shape) - logistic)

    def hyperbolic_tangent(self, z):
        """Aplica função de ativação tangente hiperbólica."""
        return numpy.tanh(z)

    def hyperbolic_tangent_derivative(self, z):
        """Derivada da função de tangente hiperbólica: f'(x) = 1 - f(x)²."""
        hyperbolic_tangent = self.hyperbolic_tangent(z)
        return numpy.ones(z.shape) - numpy.multiply(hyperbolic_tangent, hyperbolic_tangent)

    def linear(self, z):
        """Aplicação função de ativação linear."""
        return z

    def linear_derivative(self, z):
        """Derivada da função linear."""
        return 1

    def bisection(self, X, y, W1, W2, dJdW1, dJdW2):
        """Estima alfas ótimos pelo método da bisseção."""
        alpha_l, alpha_u = 0.0, 1.0
        Y, J, hlinha1, hlinha2 = self.single_step(X, y, W1 + alpha_u * dJdW1, W2)
        hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(dJdW1, axis=1))
        iteration = 0
        #print("hlinha1 %f, alpha_u = %f" % (hlinha1, alpha_u))
        #time.sleep(2)
        while hlinha1 < -1e-4 and iteration < 10:
            alpha_u *= 2.0
            Y, J, hlinha1, hlinha2 = self.single_step(X, y, W1 + alpha_u * dJdW1, W2)
            hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(dJdW1, axis=1))
            iteration += 1
            #print("hlinha1 %f, alpha_u = %f" % (hlinha1, alpha_u))
            #time.sleep(2)

        alpha1 = (alpha_l + alpha_u) / 2.0
        Y, J, hlinha1, hlinha2 = self.single_step(X, y, W1 + alpha1 * dJdW1, W2)
        hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(dJdW1, axis=1))
        iteration = 0
        #print("hlinha1 %f, alpha1 = %f" % (hlinha1, alpha1))
        #time.sleep(2)
        while abs(hlinha1) > 1e-4 and abs(alpha1 - alpha_l) > 1e-4 and abs(alpha1 - alpha_u) > 1e-4 and iteration < 200:
            if hlinha1 > 0:
                alpha_u = alpha1
            else:
                alpha_l = alpha1
            alpha1 = (alpha_l + alpha_u) / 2.0
            Y, J, hlinha1, hlinha2 = self.single_step(X, y, W1 + alpha1 * dJdW1, W2)
            hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(dJdW1, axis=1))
            iteration += 1
            #print("hlinha1 %f, alpha1 = %f" % (hlinha1, alpha1))
            #time.sleep(2)

        # Utiliza método da bisseção para encontrar alfa2 ótimo
        alpha_l, alpha_u = 0.0, 1.0
        Y, J, hlinha1, hlinha2 = self.single_step(X, y, W1, W2 + alpha_u * dJdW2)
        hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(dJdW2, axis=1))
        iteration = 0
        #print("hlinha2 %f, alpha_u = %f" % (hlinha2, alpha_u))
        #time.sleep(2)
        while hlinha2 < -1e-4 and iteration < 10:
            alpha_u *= 2.0
            Y, J, hlinha1, hlinha2 = self.single_step(X, y, W1, W2 + alpha_u * dJdW2)
            hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(dJdW2, axis=1))
            iteration += 1
            #print("hlinha2 %f, alpha_u = %f" % (hlinha2, alpha_u))
            #time.sleep(2)
        
        alpha2 = (alpha_l + alpha_u) / 2.0
        Y, J, hlinha1, hlinha2 = self.single_step(X, y, W1, W2 + alpha2 * dJdW2)
        hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(dJdW2, axis=1))
        iteration = 0
        #print("hlinha2 %f, alpha2 = %f" % (hlinha2, alpha2))
        #time.sleep(2)
        while abs(hlinha2) > 1e-4 and abs(alpha2 - alpha_l) > 1e-4 and abs(alpha2 - alpha_u) > 1e-4 and iteration < 200:
            if hlinha2 > 0:
                alpha_u = alpha2
            else:
                alpha_l = alpha2
            alpha2 = (alpha_l + alpha_u) / 2.0
            Y, J, hlinha1, hlinha2 = self.single_step(X, y, W1, W2 + alpha2 * dJdW2)
            hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(dJdW2, axis=1))
            iteration += 1
            #print("hlinha2 %f, alpha2 = %f" % (hlinha2, alpha2))
            #time.sleep(2)
        return alpha1, alpha2

    def score(self, X, y=None):
        """Retorna a acurácia média para os dados informados."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
