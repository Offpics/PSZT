import numpy as np
from activations import linear, sigmoid, relu, softmax

class MLP:
    """ Multilayer perceptron model. """


    def __init__(self):
        self.nn_architecture = [
            {"input_dim": 8257, "output_dim": 4000, "activation": linear},
            {"input_dim": 4000, "output_dim": 100, "activation": linear},
            {"input_dim": 100, "output_dim": 10, "activation": linear},
            {"input_dim": 10, "output_dim": 3, "activation": softmax},
        ]

        self.param_values = {}

        self.memory = {}


    def init_weights(self):
        """ Initialize weights in every layer. """

        for idx, layer in enumerate(self.nn_architecture, start = 1):
            input_size = layer["input_dim"]
            output_size = layer["output_dim"]

            self.param_values['W' + str(idx)] = np.random.randn(
                output_size, input_size) * 0.1
            self.param_values['b' + str(idx)] = np.random.randn(
                output_size, 1) * 0.1


    def forward_propagation(self, X):
        """ Perform forward propagation on one sentence. """
        A_curr = X

        for idx, layer in enumerate(self.nn_architecture, start = 1):
            A_prev = A_curr

            W_curr = self.param_values["W" + str(idx)]
            b_curr = self.param_values["b" + str(idx)]

            Z_curr = np.dot(W_curr, A_prev) + b_curr

            activation_function = layer["activation"]
            A_curr = activation_function(Z_curr)

            self.memory["A" + str(idx-1)] = A_prev
            self.memory["Z" + str(idx)] = Z_curr

        return A_curr


    def cross_entropy_loss(self, y_pred, y_true):
        cross_entropy = 0

        for i in range(len(y_pred)):
            cross_entropy += y_true[i] * np.log(y_pred[i])
    
        cross_entropy = -cross_entropy.sum()

        return cross_entropy

            

    def add(self, input_dim, output_dim, activation):
        """ Add layer to the neural network.

        Args:
            input_dim: Dimension of the input
            output_dim: Dimension of the output
            activation: Activation function
        """

        # Create new layer.
        layer = {"input_dim": input_dim,
                 "output_dim": output_dim,
                 "activation": activation}

        self.nn_architecture.append(layer)
