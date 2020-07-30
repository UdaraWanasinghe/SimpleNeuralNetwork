import numpy as np
import activation_functions as af
import random
import matplotlib
import matplotlib.pyplot as plt


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

    def update_mini_batch(self, mini_batch, learning_rate):
        w_gradients = [np.zeros(w.shape) for w in self.weights]
        b_gradients = [np.zeros(b.shape) for b in self.biases]
        m = len(mini_batch)
        # calculate average of delta values in the mini batch
        for x, y in mini_batch:
            delta_w_gradients, delta_b_gradients = self.backprop(x, y)
            w_gradients = [(x + y) / m for x,
                           y in zip(w_gradients, delta_w_gradients)]
            b_gradients = [(x + y) / m for x,
                           y in zip(b_gradients, delta_b_gradients)]
        # adjust weights and biases
        self.weights = [w - (learning_rate * d)
                        for w, d in zip(self.weights, w_gradients)]
        self.biases = [b - (learning_rate * d)
                       for b, d in zip(self.weights, w_gradients)]

    def evaluate(self, test_data):
        # evaluate test data, result is a list of tuples (output_number, desired_output)
        result = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in result)

    def SGD(self, training_data, mini_batch_size, learning_rate, epoches, test_data=None):
        """Train the neural network using stochastic gradient descent
        The training data is the list of tuples (x, y) representing
        training inputs and desired outputs. If test data is provided,
        then the network will be evaluated against the test data after
        each epoch."""
        results = []
        # iterate through each epoch
        for e in range(epoches):
            # shuffle the training data set
            random.shuffle(training_data)
            # divide training data into mini-batches
            mini_batches = [training_data[k: k + mini_batch_size]
                            for k in range(0, len(training_data), mini_batch_size)]
            # update weights and biases from mini batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            # if test data is provided, evaluate the accuracy
            if test_data:
                accuracy = self.evaluate(test_data)
                print("Epoch ", e, ": ", accuracy)
                results.append(accuracy)

    def plot_graph(self, results, test_data_len):
        t = np.arange(0, len(results), 1)
        fig, ax = plt.subplots()
        ax.plot(t, results)
        ax.set(xlabel='time (s)', ylabel='voltage (mV)',
               title='About as simple as it gets, folks')
