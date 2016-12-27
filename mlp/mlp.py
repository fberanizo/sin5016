# -*- coding: utf-8 -*-

import numpy, hashlib, time
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import MinMaxScaler, label_binarize

class MLP(object):
    """Class that implements a multilayer perceptron (MLP)"""
    def __init__(self, hidden_layer_size=3, max_epochs=10000):
        self.hidden_layer_size = hidden_layer_size
        self.max_epochs = max_epochs
        self.auc = 0.5
        self.cache = {}

    def fit(self, X, y):
        """Trains the network and returns the trained network"""
        scaler = MinMaxScaler((-1,1))
        X = scaler.fit_transform(X)
        y = label_binarize(y, classes=numpy.unique(y))

        self.input_layer_size = X.shape[1]
        self.output_layer_size = y.shape[1]
        epoch = 1

        # Initialize weights
        self.W1 = numpy.random.rand(1 + self.input_layer_size, self.hidden_layer_size)
        self.W2 = numpy.random.rand(1 + self.hidden_layer_size, self.output_layer_size)
        
        # Useful for loss calculation
        self.W1_history = []
        self.W2_history = []

        epsilon = 0.001
        error = 1
        self.J = [] # error

        # Repeats until error is small enough or max epochs is reached
        while error > epsilon and epoch <= self.max_epochs:
            total_error = numpy.array([])

            # Saves old weights
            self.W1_history.append(self.W1)
            self.W2_history.append(self.W2)

            # For each input instance
            for self.X, self.y in zip(X, y):
                self.X = numpy.array([self.X])
                self.y = numpy.array([self.y])
                error, (dJdW1, dJdW2) = self.single_step(self.X, self.y, self.W1, self.W2)
                total_error = numpy.append(total_error, error)

                # Algoritmo de gradiente conjugado
                self.d1, self.d2 = self.g1, self.g2 = -dJdW1, -dJdW2
                self.mg1, self.mg2 = numpy.mean(self.g1, axis=1), numpy.mean(self.g2, axis=1)

                # No passo n, usar a busca em linha para encontrar eta(n) que minimiza
                while numpy.linalg.norm(self.mg1) > 1e-5 or numpy.linalg.norm(self.mg2) > 1e-5:
                    print("mg1 norm %f" % numpy.linalg.norm(self.mg1))
                    # Utiliza método da bisseção para encontrar alfa1 ótimo
                    alpha_l, alpha_u = 0.0, 1.0
                    error, (hlinha1, hlinha2) = self.single_step(self.X, self.y, self.W1 + alpha_u * self.d1, self.W2)
                    hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(self.d1, axis=1))
                    #print("hlinha1 %f, alpha_u = %f" % (hlinha1, alpha_u))
                    #time.sleep(2)
                    while hlinha1 < -1e-5:
                        alpha_u *= 2.0
                        error, (hlinha1, hlinha2) = self.single_step(self.X, self.y, self.W1 + alpha_u * self.d1, self.W2)
                        hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(self.d1, axis=1))
                        #print("hlinha1 %f, alpha_u = %f" % (hlinha1, alpha_u))
                        #time.sleep(2)

                    alpha1 = (alpha_l + alpha_u) / 2.0
                    error, (hlinha1, hlinha2) = self.single_step(self.X, self.y, self.W1 + alpha1 * self.d1, self.W2)
                    hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(self.d1, axis=1))
                    #print("hlinha1 %f, alpha1 = %f" % (hlinha1, alpha1))
                    #time.sleep(2)
                    while abs(hlinha1) > 1e-5:
                        if hlinha1 > 0:
                            alpha_u = alpha1
                        else:
                            alpha_l = alpha1
                        alpha1 = (alpha_l + alpha_u) / 2.0
                        error, (hlinha1, hlinha2) = self.single_step(self.X, self.y, self.W1 + alpha1 * self.d1, self.W2)
                        hlinha1 = numpy.dot(numpy.mean(hlinha1, axis=1).T, numpy.mean(self.d1, axis=1))
                        #print("hlinha1 %f, alpha1 = %f" % (hlinha1, alpha1))
                        #time.sleep(2)

                    # Utiliza método da bisseção para encontrar alfa2 ótimo
                    alpha_l, alpha_u = 0.0, 1.0
                    error, (hlinha1, hlinha2) = self.single_step(self.X, self.y, self.W1, self.W2 + alpha_u * self.d2)
                    hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(self.d2, axis=1))
                    #print("hlinha2 %f, alpha_u = %f" % (hlinha2, alpha_u))
                    #time.sleep(2)
                    while hlinha2 < -1e-5:
                        alpha_u *= 2.0
                        error, (hlinha1, hlinha2) = self.single_step(self.X, self.y, self.W1, self.W2 + alpha_u * self.d2)
                        hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(self.d2, axis=1))
                        #print("hlinha2 %f, alpha_u = %f" % (hlinha2, alpha_u))
                        #time.sleep(2)
                    
                    alpha2 = (alpha_l + alpha_u) / 2.0
                    error, (hlinha1, hlinha2) = self.single_step(self.X, self.y, self.W1, self.W2 + alpha2 * self.d2)
                    hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(self.d2, axis=1))
                    #print("hlinha2 %f, alpha2 = %f" % (hlinha2, alpha2))
                    #time.sleep(2)
                    while abs(hlinha2) > 1e-5:
                        if hlinha2 > 0:
                            alpha_u = alpha2
                        else:
                            alpha_l = alpha2
                        alpha2 = (alpha_l + alpha_u) / 2.0
                        error, (hlinha1, hlinha2) = self.single_step(self.X, self.y, self.W1, self.W2 + alpha2 * self.d2)
                        hlinha2 = numpy.dot(numpy.mean(hlinha2, axis=1).T, numpy.mean(self.d2, axis=1))
                        #print("hlinha2 %f, alpha2 = %f" % (hlinha2, alpha2))
                        #time.sleep(2)

                    # Atualiza o vetor peso
                    self.W1 += alpha1 * self.d1
                    self.W2 += alpha2 * self.d2
                    # Usa backpropagation para computar o vetor gradiente 
                    error, (dJdW1, dJdW2) = self.single_step(self.X, self.y, self.W1, self.W2)
                    g1, g2 = -dJdW1, -dJdW2
                    mg1, mg2 = numpy.mean(g1, axis=1), numpy.mean(g2, axis=1)
                    # Usa o método de Polak-Ribiére para calcular beta
                    beta1 = max(0, numpy.dot(mg1.T, mg1 - self.mg1) / numpy.dot(self.mg1.T, self.mg1))
                    beta2 = max(0, numpy.dot(mg2.T, mg2 - self.mg2) / numpy.dot(self.mg2.T, self.mg2))
                    # Atualiza a direção conjugada
                    d1 = g1 + beta1 * self.d1
                    d2 = g2 + beta2 * self.d2
                    # Salva valores para utilização no próximo passo
                    self.d1, self.d2, self.g1, self.g2, self.mg1, self.mg2 = d1, d2, g1, g2, mg1, mg2

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
        """Runs single step training method"""
        self.Y = self.forward(X, W1, W2)
        cost = self.cost(self.Y, y)
        gradients = self.backpropagate(X, y, W1, W2)

        return cost, gradients

    def forward(self, X, W1, W2):
        """Passes input values through network and return output values"""
        self.Zin = numpy.dot(X, W1[:-1,:])
        self.Zin += numpy.dot(numpy.ones((1, 1)), W1[-1:,:])
        self.Z = self.logistic(self.Zin)
        self.Z = numpy.nan_to_num(self.Z)

        self.Yin = numpy.dot(self.Z, W2[:-1,])
        self.Yin += numpy.dot(numpy.ones((1, 1)), W2[-1:,:])
        Y = self.linear(self.Yin)
        Y = numpy.nan_to_num(Y)
        return Y

    def cost(self, Y, y):
        """Calculates network output error"""
        return mean_squared_error(Y, y)

    def backpropagate(self, X, y, W1, W2):
        """Backpropagates costs through the network"""
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

