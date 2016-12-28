# -*- coding: utf-8 -*-

import numpy, hashlib, time
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import MinMaxScaler, label_binarize

class MLP(object):
    """Classe que implementa um multilayer perceptron (MLP)."""
    def __init__(self, hidden_layer_size=3, max_epochs=10000):
        self.hidden_layer_size = hidden_layer_size
        self.max_epochs = max_epochs
        self.cache = {}

    def fit(self, X, y):
        """Trains the network and returns the trained network"""
        self.input_layer_size = X.shape[1]
        self.output_layer_size = y.shape[1]

        # Normaliza valores e adiciona coluna de bias
        scaler = MinMaxScaler((-1,1))
        self.X = numpy.c_[scaler.fit_transform(X), numpy.ones(X.shape[0])]
        self.y = label_binarize(y, classes=numpy.unique(y))

        # Inicializa pesos da rede
        self.W1 = numpy.random.rand(self.hidden_layer_size, 1 + self.input_layer_size)
        self.W2 = numpy.random.rand(self.output_layer_size, 1 + self.hidden_layer_size)
        
        # Useful for loss calculation
        self.W1_history = []
        self.W2_history = []
        self.J = [] # erro

        epoch, error = 1, 1
        # Repete até que o erro seja pequeno, ou o máximo de época é alcançado
        while error > 1e-2 and epoch <= self.max_epochs:
            total_error = []

            # Saves old weights
            self.W1_history.append(self.W1)
            self.W2_history.append(self.W2)

            # Treinamento padrão-a-padrão
            for X, y in zip(self.X, self.y):
                X, y = numpy.array([self.X]), numpy.array([y])
                Y, J, dJdW1, dJdW2 = self.single_step(self.X, self.y, self.W1, self.W2)
                total_error = numpy.append(total_error, J)

                # Algoritmo de gradiente conjugado
                d1, d2 = g1, g2 = -dJdW1, -dJdW2
                self.mg1, self.mg2 = numpy.mean(g1, axis=1), numpy.mean(g2, axis=1)

                while numpy.linalg.norm(self.mg1) > 1e-5 or numpy.linalg.norm(self.mg2) > 1e-5:
                    # Encontra alfa que otimiza passo
                    alpha1, alpha2 = self.bisection(self.X, self.y, self.W1, self.W2, d1, d2)
                    # Atualiza o vetor peso
                    self.W1 += alpha1 * d1
                    self.W2 += alpha2 * d2
                    # Usa backpropagation para computar o vetor gradiente 
                    error, (dJdW1, dJdW2) = self.single_step(self.X, self.y, self.W1, self.W2)
                    g1, g2 = -dJdW1, -dJdW2
                    mg1, mg2 = numpy.mean(g1, axis=1), numpy.mean(g2, axis=1)
                    # Usa o método de Polak-Ribiére para calcular beta
                    beta1 = max(0, numpy.dot(mg1.T, mg1 - self.mg1) / numpy.dot(self.mg1.T, self.mg1))
                    beta2 = max(0, numpy.dot(mg2.T, mg2 - self.mg2) / numpy.dot(self.mg2.T, self.mg2))
                    # Atualiza a direção conjugada
                    d1, d2 = g1 + beta1 * d1, g2 + beta2 * d2
                    # Salva valores para utilização no próximo passo
                    self.mg1, self.mg2 = mg1, mg2

            # Saves error for plot
            error = total_error.mean()
            self.J.append(error)

            print('Epoch: ' + str(epoch))
            print('Error: ' + str(error))

            epoch += 1

        return self

    def predict(self, X):
        """Predicts test values"""
        Y = map(lambda x: self.forward(numpy.array([x]))[0], X)
        Y = map(lambda y: 1 if y > self.auc else 0, Y)
        return numpy.array(Y)

    def score(self, X, y_true):
        """Calculates accuracy"""
        y_pred = map(lambda x: self.forward(numpy.array([x]))[0], X)
        auc = roc_auc_score(y_true, y_pred)
        y_pred = map(lambda y: 1 if y > self.auc else 0, y_pred)
        y_pred = numpy.array(y_pred)
        return accuracy_score(y_true.flatten(), y_pred.flatten())

    def error_per_epoch(self, X, y_true):
        """Return scores at each previously calculated epoch"""
        curr_W1 = self.W1
        curr_W2 = self.W2

        errors = []

        for epoch, (self.W1, self.W2) in enumerate(zip(self.W1_history, self.W2_history)):
            #print 'Epoch: ' + str(epoch)
            y_pred = map(lambda x: self.forward(numpy.array([x]))[0], X)
            #auc = roc_auc_score(y_true, y_pred)
            #y_pred = map(lambda y: 1 if y > self.auc else 0, y_pred)
            y_pred = numpy.array(y_pred)

            errors.append(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        
        self.W1 = curr_W1
        self.W2 = curr_W2

        return errors

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
        dJdW2 = numpy.dot(J, self.linear_derivative(self.Yin))
        dJdW2 = numpy.dot(dJdW2, self.Z)

        delta3 = numpy.multiply(-(y-self.Y), self.linear_derivative(self.Yin))
        dJdW2 = numpy.dot(self.Z.T, delta3)
        dJdW2 = numpy.append(dJdW2, numpy.dot(numpy.ones((1, 1)), delta3), axis=0)

        delta2 = numpy.dot(delta3, W2[:-1,:].T) * self.logistic_derivative(self.Zin)
        dJdW1 = numpy.dot(X.T, delta2)
        dJdW1 = numpy.append(dJdW1, numpy.dot(numpy.ones((1, 1)), delta2), axis=0)

        return dJdW1, dJdW2

    def logistic(self, z):
        """Apply logistic activation function"""
        h = hashlib.sha1(z.view(numpy.uint8)).hexdigest()
        if h not in self.cache:
            self.cache[h] = 1 / (1 + numpy.exp(-z))
        return self.cache[h]

    def logistic_derivative(self, z):
        """Derivative of logistic function: f'(x) = f(x).(1-f(x))"""
        logistic = self.logistic(z)
        return numpy.multiply(logistic, numpy.ones(z.shape) - logistic)

    def hyperbolic_tangent(self, z):
        """Apply hyperbolic tangent activation function"""
        return numpy.tanh(z)

    def hyperbolic_tangent_derivative(self, z):
        """Derivative of hyperbolic tangent function: f'(x) = 1 - f(x)²"""
        hyperbolic_tangent = self.hyperbolic_tangent(z)
        return numpy.ones(z.shape) - numpy.multiply(hyperbolic_tangent, hyperbolic_tangent)

    def linear(self, z):
        """Apply linear activation function"""
        return z

    def linear_derivative(self, z):
        """Derivarive linear function"""
        return 1

    def bisection(self, X, y, W1, W2, dJdW1, dJdW2):
        """Estima alfas ótimos pelo método da bisseção"""
        alpha_l, alpha_u = 0.0, 1.0
        error, (hlinha1, hlinha2) = self.single_step(X, y, W1 + alpha_u * dJdW1, W2)
        hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(dJdW1, axis=1))
        #print("hlinha1 %f, alpha_u = %f" % (hlinha1, alpha_u))
        #time.sleep(2)
        while hlinha1 < -1e-5:
            alpha_u *= 2.0
            error, (hlinha1, hlinha2) = self.single_step(X, y, W1 + alpha_u * dJdW1, W2)
            hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(dJdW1, axis=1))
            #print("hlinha1 %f, alpha_u = %f" % (hlinha1, alpha_u))
            #time.sleep(2)

        alpha1 = (alpha_l + alpha_u) / 2.0
        error, (hlinha1, hlinha2) = self.single_step(X, y, W1 + alpha1 * dJdW1, W2)
        hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(dJdW1, axis=1))
        #print("hlinha1 %f, alpha1 = %f" % (hlinha1, alpha1))
        #time.sleep(2)
        while abs(hlinha1) > 1e-5 and abs(alpha1 - alpha_l) > 1e-5 and abs(alpha1 - alpha_u) > 1e-5:
            if hlinha1 > 0:
                alpha_u = alpha1
            else:
                alpha_l = alpha1
            alpha1 = (alpha_l + alpha_u) / 2.0
            error, (hlinha1, hlinha2) = self.single_step(X, y, W1 + alpha1 * dJdW1, W2)
            hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(dJdW1, axis=1))
            #print("hlinha1 %f, alpha1 = %f" % (hlinha1, alpha1))
            #time.sleep(2)

        # Utiliza método da bisseção para encontrar alfa2 ótimo
        alpha_l, alpha_u = 0.0, 1.0
        error, (hlinha1, hlinha2) = self.single_step(X, y, W1, W2 + alpha_u * dJdW2)
        hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(dJdW2, axis=1))
        #print("hlinha2 %f, alpha_u = %f" % (hlinha2, alpha_u))
        #time.sleep(2)
        while hlinha2 < -1e-5:
            alpha_u *= 2.0
            error, (hlinha1, hlinha2) = self.single_step(X, y, W1, W2 + alpha_u * dJdW2)
            hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(dJdW2, axis=1))
            #print("hlinha2 %f, alpha_u = %f" % (hlinha2, alpha_u))
            #time.sleep(2)
        
        alpha2 = (alpha_l + alpha_u) / 2.0
        error, (hlinha1, hlinha2) = self.single_step(X, y, W1, W2 + alpha2 * dJdW2)
        hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(dJdW2, axis=1))
        #print("hlinha2 %f, alpha2 = %f" % (hlinha2, alpha2))
        #time.sleep(2)
        while abs(hlinha2) > 1e-5 and abs(alpha2 - alpha_l) > 1e-5 and abs(alpha2 - alpha_u) > 1e-5:
            if hlinha2 > 0:
                alpha_u = alpha2
            else:
                alpha_l = alpha2
            alpha2 = (alpha_l + alpha_u) / 2.0
            error, (hlinha1, hlinha2) = self.single_step(X, y, W1, W2 + alpha2 * dJdW2)
            hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(dJdW2, axis=1))
            #print("hlinha2 %f, alpha2 = %f" % (hlinha2, alpha2))
            #time.sleep(2)
        return alpha1, alpha2