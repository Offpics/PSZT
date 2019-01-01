import numpy as np


class MLP():
    """ Multilayer Perceptron.

    This artificial neural network model is designed for sentiment analysis.
    The activation function of the last layer should always be 'softmax'. """

    def __init__(self):
        # List of layers of the model.
        self.layers = []

        # Dictionary that holds w, b params.
        self.param_values = {}

        # Dictionary that holds z, a params.
        self.memory = {}

        # Dictionary that holds gradients.
        self.grad_values = {}

        # Learning rate of the optimizer.
        self.lr = 10e-4

    def init_layers(self):
        """ Initialize weights w, b in every layer of the neural network. """

        for i, layer in enumerate(self.layers, start=1):
            # Number of inputs to the layer equals to the number of neurons
            # in the previous layer.
            input_size = layer["input_dim"]

            # Number of outputs of the layer equals to the number of neurons
            # in the current layer.
            output_size = layer["output_dim"]

            # Initialize weights w, b and save them to dictionary param_values
            # for later usage.
            self.param_values['w' + str(i)] = np.random.rand(
                input_size, output_size) * 0.01
            self.param_values['b' + str(i)] = np.random.randn(
                output_size) * 0.01

    def train(self, x, y_true, epochs, silent=False):
        """ Perform training of neural network.

        Args:
            x(numpy.ndarray): Array of input data in shape
                              (num_of_inputs, vector_input).
            y_true(numpy.ndarray): Array of true labels in one-hot format
                                   and shape (num_of_inputs, one-hot).
            epochs(int): Number of epochs to perform.
        """

        for _ in range(epochs):
            # Perform forward propagation over neural network.
            self._forward(x)

            # Perform backward propagation over neural network.
            self._backward(x, y_true)

            # Update weights of neural network.
            self._update_weights()

            if silent == False:
                # Calculate cross_entropy_loss.
                y_pred = self.memory['a' + str(len(self.layers))]
                cross_entropy_loss = np.sum(-y_true * np.log(y_pred))

                # Calculate accuracy over whole input data.
                accuracy = self.calculate_accuracy(y_true, y_pred)

                # Print current loss and accuracy of the neural net.
                print(f'loss: {cross_entropy_loss:.2f}, accuracy: {accuracy:.1f}%')

    def calculate_accuracy(self, y_true, y_pred):
        correct_pred = 0
        for i in range(len(y_pred)):
            equal = np.equal(np.argmax(y_true[i]), np.argmax(y_pred[i]))
            correct_pred += equal.astype(float)
        accuracy = (correct_pred/len(y_true))*100

        return accuracy

    def score(self, x, y):
        self._forward(x)
        y_pred = self.memory['a' + str(len(self.layers))]
        
        accuracy = self.calculate_accuracy(y, y_pred)

        return accuracy

    def k_fold_validation(self, x, y_true, k):
        """ Perform k-fold cross validation. """

        # Split dataset into k folds.
        x_folds = np.array_split(x, k)
        y_folds = np.array_split(y_true, k)

        accuracies = []

        for i, fold in enumerate(x_folds[:(k-1)]):
            # Init new weights for layers.
            self.init_layers()

            # Train network with one fold.
            self.train(fold, y_folds[i], 300, True)

            # Calculate accuracy of the trained network on last fold.
            accuracy = self.score(x_folds[k-1], y_folds[k-1])

            accuracies.append(accuracy)

        print(f'Accuracies on k_folds: {accuracies}')
        print(f'Mean of accuracies: {np.mean(accuracies)}')

    def _forward(self, x):
        """ Perform forward step in the neural network.

        Args:
            x(numpy.ndarray): Array of input data in shape
                              (num_of_inputs, vector_input).
        """
        # This is done only for input layer. Assign input data to a and save
        # it to dictionary memory for later usage.
        a = x
        self.memory['a0'] = a

        for i, layer in enumerate(self.layers, start=1):
            # Assing output of previous layer to a_prev.
            a_prev = a

            # Get activation function of the current layer.
            activation_name = layer["activation"]
            act_func = self.activation_function(activation_name)

            # Get weights w, b of current layer.
            w = self.param_values['w' + str(i)]
            b = self.param_values['b' + str(i)]

            # Calculate input with weights and perform activation.
            z = np.dot(a_prev, w) + b
            a = act_func(z)

            # Save calculated values to the memory dictionary for later usage.
            self.memory['a' + str(i)] = a
            self.memory['z' + str(i)] = z

    def _backward(self, x, y_true):
        """ Perform backward step in the neural network.

        Args:
            x(numpy.ndarray): Array of input data in shape
                              (num_of_inputs, vector_input).
            y_true(numpy.ndarray): Array of true labels in one-hot format
                                   and shape (num_of_inputs, one-hot).
        """

        # Traverse through layers in descending order.
        for i, layer in reversed(list(enumerate(self.layers, start=1))):
            if i == len(self.layers):
                # Calculate this only for the output layer.

                # Get output of current and current-1 layer.
                a = self.memory['a' + str(i)]
                a_prev = self.memory['a' + str(i-1)]

                dcost_dz = a - y_true  # dcost/dzo = dcost/dao * dao/dzo

            else:
                # Calculate this for all hidden layers.

                # Get necessary data.
                z = self.memory['z' + str(i)]
                w_prev = self.param_values['w' + str(i+1)]
                dcost_dz_prev = self.grad_values['dcost_dz' + str(i+1)]
                a_prev = self.memory['a' + str(i-1)]

                # Get derivate activation function.
                activation_name = layer["activation"]
                act_func_der = self.activation_function_der(activation_name)

                # Calculate derivatives.
                dz_da = w_prev
                dcost_da = np.dot(dcost_dz_prev, dz_da.T)
                da_dz = act_func_der(z)

                dcost_dz = da_dz * dcost_da  # dcost/dzh = dcost/dah * dah/dzh

            # Calculate derivatives.
            dz_dw = a_prev
            dcost_w = np.dot(dz_dw.T, dcost_dz)
            dcost_b = dcost_dz

            # Save derivatives for later usage.
            self.grad_values['dcost_dz' + str(i)] = dcost_dz
            self.grad_values['dcost_w' + str(i)] = dcost_w
            self.grad_values['dcost_b' + str(i)] = dcost_b

    def _update_weights(self):
        """ Update of weights in every layer based on stored gradients. """
        for i, _ in enumerate(self.layers, start=1):
            # Get gradients for current layer.
            dcost_w = self.grad_values['dcost_w' + str(i)]
            dcost_b = self.grad_values['dcost_b' + str(i)]

            # Update w and b.
            self.param_values['w' + str(i)] -= self.lr * dcost_w
            self.param_values['b' + str(i)] -= self.lr * dcost_b.sum(axis=0)

    def set_lr(self, lr):
        """ Set learning rate.

        Args:
            lr(int): Learning rate.
        """
        self.lr = lr

    def add_layer(self, input_dim, output_dim, activation):
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

        self.layers.append(layer)

    def print_layers(self):
        """ Print layers of the neural network. """
        for i, layer in enumerate(self.layers, start = 1):
            print(f'{i}. Layer - input_dim: {layer["input_dim"]}, ', end='')
            print(f'output_dim: {layer["output_dim"]}, ', end='')
            print(f'activation: {layer["activation"]}')

    def activation_function(self, name):
        """ Return activation function based on name.

        Args:
            name(string): Name of activation function.
        """
        if name == 'relu':
            return self.relu
        elif name == 'sigmoid':
            return self.sigmoid
        elif name == 'softmax':
            return self.softmax
        else:
            raise TypeError()

    def activation_function_der(self, name):
        """ Return activation function derivative based on name.

        Args:
            name(string): Name of activation function.
        """
        if name == 'relu':
            return self.relu_der
        elif name == 'sigmoid':
            return self.sigmoid_der
        else:
            raise TypeError()

    def relu(self, x):
        return np.maximum(0, x)

    def relu_der(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)
