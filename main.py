import numpy as np
import activation_functions as af


class NeuralNetwork():
    def __init__(self, init_f, act_f, sizes):
        # set number of layers
        self.num_layers = len(sizes)
        # set layer sizes
        self.sizes = sizes
        # set activation function
        self.act_f = act_f
        # initialize weights and biases
        self.weights, self.biases = init_f(sizes)

    def feedforward(self, x):
        # x is the input layer of neural network
        a = x
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = self.act_f.act(z)
        return a

    def backprop(self, x, y):
        # this function calculates gradient values for a single training input
        # calculate activations
        a = x
        activations = [x]
        sums = []
        # calculate sums and activations
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            sums.append(z)
            a = self.act_f.act(z)
            activations.append(a)
        # calculate cost relative to sum in ouput layer (delta)
        delta = (activations[-1] - y) * self.act_f.act_dir(sums[-1])
        # calculate gradients of the cost function
        w_gradients = [np.zeros(w.shape) for w in self.weights]
        b_gradients = [np.zeros(b.shape) for b in self.biases]
        # gradients at the output layer
        w_gradients[-1] = np.dot(delta, activations[-2].transpose())
        b_gradients[-1] = delta
        for l in range(2, self.num_layers):
            sum = sums[-l]
            # calculate delta of the previous layer
            delta = np.dot(
                self.weights[-1 + l].transpose(), delta) * self.act_f.act_dir(sum)
            # calculate gradients
            w_gradients[-l] = np.dot(delta, activations[-l-1].transpose())
            b_gradients[-l] = delta
        return (w_gradients, b_gradients)
