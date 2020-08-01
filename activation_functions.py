import numpy as np


class Sigmoid():

    def act(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def act_dir(self, x):
        return (1.0 - self.act(x))*self.act(x)
