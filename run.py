import data_loader as dl
from main import NeuralNetwork
import initialization_functions as ifn
from activation_functions import Sigmoid

print("Reading training data")
training_data = dl.prepare_training_data()
print("Reading testing data")
testing_data = dl.prepare_testing_data()
init_func = ifn.random_initialize
act_func = Sigmoid()
sizes = [784, 30, 10]
network = NeuralNetwork(init_func, act_func, sizes)
network.SGD(training_data, 10, 3.0, 30, test_data=testing_data)
