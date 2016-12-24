# -*- coding: utf-8 -*-

import numpy, hashlib
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, log_loss

class MLP(object):
    """Class that implements a multilayer perceptron (MLP)"""
    def __init__(self, hidden_layer_size=3, learning_rate=0.2, max_epochs=10000):
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.auc = 0.5
        self.cache = {}

    def fit(self, X, y):
        """Trains the network and returns the trained network"""
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
                error, gradients = self.single_step(self.X, self.y)
                total_error = numpy.append(total_error, error)
                dJdW1 = gradients[0]
                dJdW2 = gradients[1]

                # Calculates new weights
                self.W1 = self.W1 - self.learning_rate * dJdW1
                self.W2 = self.W2 - self.learning_rate * dJdW2

            # Saves error for plot
            error = total_error.mean()
            self.J.append(error)

            
            print 'Epoch: ' + str(epoch)
            #print 'Learning Rate: ' + str(self.learning_rate)
            print 'Error: ' + str(error)

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

    def single_step(self, X, y):
        """Runs single step training method"""
        self.Y = self.forward(X)
        cost = self.cost(self.Y, y)
        gradients = self.backpropagate(X, y)

        return cost, gradients

    def forward(self, X):
        """Passes input values through network and return output values"""
        self.Zin = numpy.dot(X, self.W1[:-1,:])
        self.Zin += numpy.dot(numpy.ones((1, 1)), self.W1[-1:,:])
        self.Z = self.logistic(self.Zin)
        self.Z = numpy.nan_to_num(self.Z)

        self.Yin = numpy.dot(self.Z, self.W2[:-1,])
        self.Yin += numpy.dot(numpy.ones((1, 1)), self.W2[-1:,:])
        Y = self.linear(self.Yin)
        Y = numpy.nan_to_num(Y)
        return Y

    def cost(self, Y, y):
        """Calculates network output error"""
        return mean_squared_error(Y, y)

    def backpropagate(self, X, y):
        """Backpropagates costs through the network"""
        delta3 = numpy.multiply(-(y-self.Y), self.linear_derivative(self.Yin))
        dJdW2 = numpy.dot(self.Z.T, delta3)
        dJdW2 = numpy.append(dJdW2, numpy.dot(numpy.ones((1, 1)), delta3), axis=0)

        delta2 = numpy.dot(delta3, self.W2[:-1,:].T) * self.logistic_derivative(self.Zin)
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
