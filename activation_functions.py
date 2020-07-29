import numpy as np


class Sigmoid():

    def act(self, x):
        return 1 / (1 + np.exp(-x))

    def act_dir(self, x):
        return (1 - self.act(x))*(self.act(x))
