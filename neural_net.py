import numpy as np
from activations import linear, sigmoid, relu, softmax, relu_backward, sigmoid_backward

class MLP:
    """ Multilayer perceptron model. """


    def __init__(self):
        self.nn_architecture = [
            {"input_dim": 8257, "output_dim": 4000, "activation": sigmoid},
            {"input_dim": 4000, "output_dim": 100, "activation": sigmoid},
            {"input_dim": 100, "output_dim": 10, "activation": sigmoid},
            {"input_dim": 10, "output_dim": 3, "activation": softmax},
        ]

        self.param_values = {}

        self.memory = {}

        self.grad_values = {}


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

    def backward_propagation(self, Y_hat, Y):
        
        # m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)
    
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
        
        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]
            
            dA_curr = dA_prev
            
            A_prev = self.memory["A" + str(layer_idx_prev)]
            Z_curr = self.memory["Z" + str(layer_idx_curr)]
            W_curr = self.param_values["W" + str(layer_idx_curr)]
            b_curr = self.param_values["b" + str(layer_idx_curr)]
            
            ##################
            # dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            #     dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            m = A_prev.shape[1]

            if layer["activation"] == sigmoid:
                backward_activation_func = sigmoid_backward
            elif layer["activation"] == relu:
                backward_activation_func = relu_backward
            else:
                Exception('Non-supported activation function')

            dZ_curr = backward_activation_func(dA_curr, Z_curr)
            dW_curr = np.dot(dZ_curr, A_prev.T) / m
            db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
            dA_prev = np.dot(W_curr.T, dZ_curr)

            # return dA_prev, dW_curr, db_curr
            ####################
            self.grad_values["dW" + str(layer_idx_curr)] = dW_curr
            self.grad_values["db" + str(layer_idx_curr)] = db_curr

        return self.grad_values


    def cross_entropy_loss(self, y_pred, y_true):
        cross_entropy = 0

        for i in range(len(y_pred)):
            cross_entropy += y_true[i] * np.log(y_pred[i])
    
        cross_entropy = -cross_entropy.sum()

        return cross_entropy

    
    def prob_to_class(self, y_pred):
        idx = np.argmax(y_pred)
        if idx == 0:
            return "1"
        elif idx == 1:
            return "3"
        elif idx == 2:
            return "5"

            
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
