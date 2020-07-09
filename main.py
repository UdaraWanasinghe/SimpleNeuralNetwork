class NeuralNetwork():
    def __init__(self, sizes):
        '''
        We are giving number of neurons in each layer through sizes array.
        The length of the sizes array is equal to the number of layers in our neural network.
        Input layer of our neural network has 28 x 28 = 784 nodes.
        That's because we are giving pixel data of each digit as the input of the neural network.
        ''' 
        self.num_layers = len(sizes)
        self.sizes = sizes
        '''
        Next we have to initialize biases and weights.
        This is very important part.
        Because initial biases and weight values are directly effect how well our neural network perform.
        Initialization steps can be critical to model's ultimate parameters.
        Zero or constant initializer can perform poorly on the neural network.
        Initializing all the weights with zeros leads the neurons to learn the same features during training.
        Initializing the weights with values (i) too small or (ii) too large leads respectively to (i) slow learning or (ii) divergence.
        A too-small initialization leads to vanishing gradients.
        A too-large initialization leads to exploding gradients. This leads the cost to oscillate around its minimum value.
        How to find appropriate initialization values?
        The mean of the activations should be zero.
        The variance of the activations should stay the same across every layer.
        The recommended initialization is Xavier initialization.
        All the weights of layer ll are picked randomly from a normal distribution with mean μ=0 and variance σ^2 = 1 / n^(l-1)
        where n^(l-1) is the number of neuron in layer (l−1). Biases are initialized with zeros.
        '''
        