import numpy as np


class MLP():
    """ Multilayer Perceptron.

    This artificial neural network model is designed for sentiment analysis.
    The activation function of the last layer should always be 'softmax'
    and input layer should be ommited. """

    def __init__(self):
        # List of layers of the model.
        self.layers = []

        # Dictionary that holds w, b params.
        self.param_values = {}

        # Dictionary that holds z, a params.
        self.memory = {}

        # Dictionary that holds gradients.
        self.grad_values = {}

        # List to store accuracies calculated in an epoch.
        self.accuracies = []

        # List to store losses calculated in epoch.
        self.losses = []

        # Learning rate of the optimizer.
        self.lr = 10e-4

    def add_layer(self, input_dim, output_dim, activation):
        """ Add layer to the neural network.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            activation: Activation function.
        """

        # Create new layer.
        layer = {"input_dim": input_dim,
                 "output_dim": output_dim,
                 "activation": activation}

        self.layers.append(layer)

    def init_layers(self):
        """ Initialize weights w, b in every layer of the neural network. """

        for i, layer in enumerate(self.layers, start=1):
            # Number of inputs to the layer equals to the number of neurons
            # in the previous layer.
            input_size = layer["input_dim"]

            # Number of outputs of the layer equals to the number of neurons
            # in the current layer.
            output_size = layer["output_dim"]

            if i == len(self.layers):
                # Create arrays of w, b for last layer and populate
                # it with zeros.
                self.param_values['w' + str(i)] = np.zeros(
                    shape=(input_size, output_size)
                )

                self.param_values['b' + str(i)] = np.zeros(shape=output_size)
            else:

                # Create arrays of w, b and populate it with random samples
                # from a uniform distribution (U) and save
                # them for later usage.
                self.param_values['w' + str(i)] = np.random.uniform(
                    low=(-1/np.sqrt(input_size)),
                    high=(1/np.sqrt(input_size)),
                    size=(input_size, output_size)
                )
                self.param_values['b' + str(i)] = np.random.uniform(
                    low=(-1/np.sqrt(input_size)),
                    high=(1/np.sqrt(input_size)),
                    size=output_size
                )

    def train(self, x, y_true, x_test, y_test, epochs, silent=False):
        """ Perform training of neural network.

        Args:
            x(numpy.ndarray): Training dataset of shape (num_of_inputs, vector_input).
            y_true(numpy.ndarray): Array of true labels in one-hot format
                                    for training set of shape (num_of_inputs, one-hot).
            x_test(numpy.ndarray): Test dataset of shape (num_of_inputs, vector_input).
            y_test(numpy.ndarray): Array of true labels in one-hot format
                                    for test set of shape (num_of_inputs, one-hot).
            epochs(int): Number of epochs to perform.
            silent(bool): Whether to calculate loss and accuracy and print it.
        """

        for i in range(1, epochs+1):
            # Perform forward propagation over neural network.
            self._forward(x)

            # Perform backward propagation over neural network.
            self._backward(x, y_true)

            # Update weights of neural network.
            self._update_weights()

            # Calculate accuracy and loss of the trained network on train set.
            accuracy_train, loss_train = self.score(x, y_true)

            # Calculate accuracy and loss of the trained network on test set.
            accuracy_test, loss_test = self.score(x_test, y_test, True)

            # Store current accuracy.
            self.accuracies.append([accuracy_train, accuracy_test])
            self.losses.append([loss_train, loss_test])

            if silent is False:
                if i % 10 == 0:
                    # Print current loss and accuracy of the neural net.
                    print(f'Epoch: {i}, loss_train: {loss_train:.2f}, accuracy_train: {accuracy_train:.1f}%')
                    print(f'\t  loss_test: {loss_test:.2f}, accuracy_test: {accuracy_test:.1f}%')

    def train_minibatch(self, x, y_true, x_test, y_test,
                                  epochs, silent=False, batch_size=64):
        """ Perform training of neural network using minibatch.

        Args:
            x(numpy.ndarray): Training dataset of shape (num_of_inputs, vector_input).
            y_true(numpy.ndarray): Array of true labels in one-hot format
                                    for training set of shape (num_of_inputs, one-hot).
            x_test(numpy.ndarray): Test dataset of shape (num_of_inputs, vector_input).
            y_test(numpy.ndarray): Array of true labels in one-hot format
                                    for test set of shape (num_of_inputs, one-hot).
            epochs(int): Number of epochs to perform.
            silent(bool): Whether to calculate loss and accuracy and print it.
            batch_size(int): Number of samples from dataset to train net.
        """

        for i in range(1, epochs+1):
            for index in range(0, x.shape[0], batch_size):
                # Create x, y minibatches.
                x_batch = x[index:min(index+batch_size, x.shape[0])]
                y_batch = y_true[index:min(index+batch_size, x.shape[0])]
                
                # Perform forward propagation over neural network.
                self._forward(x_batch)

                # Perform backward propagation over neural network.
                self._backward(x_batch, y_batch)

                # Update weights of neural network.
                self._update_weights()

                # Calculate accuracy and loss of the trained network on train set.
                accuracy_train, loss_train = self.score(x, y_true, True)

                # Calculate accuracy and loss of the trained network on test set.
                accuracy_test, loss_test = self.score(x_test, y_test, True)

            # Store current accuracy.
            self.accuracies.append([accuracy_train, accuracy_test])
            self.losses.append([loss_train, loss_test])

            if silent is False:
                if i % 1 == 0:
                    # Print current loss and accuracy of the neural net.
                    print(f'Epoch: {i}, loss_train: {loss_train:.2f}, accuracy_train: {accuracy_train:.1f}%')
                    print(f'\t  loss_test: {loss_test:.2f}, accuracy_test: {accuracy_test:.1f}%')

    def k_fold_validation(self, x, y_true, k, epochs):
        """ Perform k-fold cross validation.
        Args:
            x(numpy.ndarray): Array of input data in shape
                              (num_of_inputs, vector_input).
            y_true(numpy.ndarray): Array of true labels in one-hot format
                                   and shape (num_of_inputs, one-hot).
            k(int): Number of folds.
            epochs(int): Number of epochs to perform.
        """

        # Create a random order to shuffle the dataset.
        s = np.arange(x.shape[0])
        np.random.shuffle(s)

        # Split dataset into k folds.
        x_folds = np.array_split(x[s], k)
        y_folds = np.array_split(y_true[s], k)

        # Lists to store accuracies and losses in ith step.
        accuracies = []
        losses = []

        for i in range(len(x_folds[:k])):
            # Init new weights for layers.
            self.init_layers()

            # Create train test T\Ti.
            x_train = np.concatenate(np.delete(x_folds, i))
            y_train = np.concatenate(np.delete(y_folds, i))

            # Create test set Ti.
            x_test = x_folds[i]
            y_test = y_folds[i]

            # Lists to store accuracies and losses in jth step.
            accuracies_curr = []
            losses_curr = []

            print(f'Current fold: T{i},')
            print(f'Len of x_train: {len(x_train)},')
            print(f'Len of x_test: {len(x_test)}.')

            # Perform training.
            for j in range(1, epochs+1):
                # Perform training and update weights.
                self._forward(x_train)
                self._backward(x_train, y_train)
                self._update_weights()

                # Calculate accuracy of the trained network on test set.
                accuracy, loss = self.score(x_test, y_test, True)

                accuracies_curr.append(accuracy)
                losses_curr.append(loss)

                if j % 10 == 0:
                    # Print current loss and accuracy of the neural net.
                    print(f'Epoch: {j}, loss: {loss:.2f}, accuracy: {accuracy:.1f}%')

            # Calculate mean losses and accuracies and print them.
            mean_accuracy = np.mean(accuracies_curr)
            mean_loss = np.mean(losses_curr)
            print(f'Mean accuracy on T{i}: {mean_accuracy:.2f}')
            print(f'Mean losses on T{i}: {mean_loss:.2f}\n')

            accuracies.append(mean_accuracy)
            losses.append(mean_loss)

        print(f'Accuracy list on Ti sets: {accuracies}')
        print(f'Losses list on Ti sets: {losses}')
        print(f'Mean accuracy on all T sets: {np.mean(accuracies):.2f}')
        print(f'Mean losses on all T sets: {np.mean(losses):.2f}')

    def k_fold_validation_minibatch(self, x, y_true, k, epochs, batch_size=256):
        """ Perform k-fold cross validation using minibatch.

        Args:
            x(numpy.ndarray): Array of input data in shape
                              (num_of_inputs, vector_input).
            y_true(numpy.ndarray): Array of true labels in one-hot format
                                   and shape (num_of_inputs, one-hot).
            k(int): Number of folds.
        """

        # Create a random order to shuffle the dataset.
        s = np.arange(x.shape[0])
        np.random.shuffle(s)

        # Split dataset into k folds.
        x_folds = np.array_split(x[s], k)
        y_folds = np.array_split(y_true[s], k)

        # Lists to store accuracies and losses in ith step.
        accuracies = []
        losses = []

        for i in range(len(x_folds[:k])):
            # Create train test T\Ti.
            x_train = np.concatenate(np.delete(x_folds, i))
            y_train = np.concatenate(np.delete(y_folds, i))

            # Create test set Ti.
            x_test = x_folds[i]
            y_test = y_folds[i]

            # Lists to store accuracies and losses in jth step.
            accuracies_curr = []
            losses_curr = []

            # Init new weights for layers.
            self.init_layers()

            print(f'Current fold: T{i},')
            print(f'Len of x_train: {len(x_train)},')
            print(f'Len of x_test: {len(x_test)}.')

            # Perform training.
            for j in range(1, epochs+1):
                for index in range(0, x_train.shape[0], batch_size):
                    # Create x, y batches.
                    x_batch = x_train[index:min(index+batch_size, x.shape[0])]
                    y_batch = y_train[index:min(index+batch_size, x.shape[0])]
                    # Perform training and update weights.
                    self._forward(x_batch)
                    self._backward(x_batch, y_batch)
                    self._update_weights()

                    # Calculate accuracy of the trained network on test set.
                    accuracy, loss = self.score(x_test, y_test, True)

                    accuracies_curr.append(accuracy)
                    losses_curr.append(loss)

                if j % 10 == 0:
                    # Print current loss and accuracy of the neural net.
                    print(f'Epoch: {j}, loss: {loss:.2f}, accuracy: {accuracy:.1f}%')

            # Calculate mean losses and accuracies and print them.
            mean_accuracy = np.mean(accuracies_curr)
            mean_loss = np.mean(losses_curr)
            print(f'Mean accuracy on T{i}: {mean_accuracy:.2f}')
            print(f'Mean losses on T{i}: {mean_loss:.2f}\n')

            accuracies.append(mean_accuracy)
            losses.append(mean_loss)

        print(f'Accuracy list on Ti sets: {accuracies}')
        print(f'Losses list on Ti sets: {losses}')
        print(f'Mean accuracy on all T sets: {np.mean(accuracies):.2f}')
        print(f'Mean losses on all T sets: {np.mean(losses):.2f}')

    def calculate_accuracy(self, y_true, y_pred):
        """ Calculate accuracy of the neural network.

        Args:
            y_true(numpy.ndarray): Array of true labels in one-hot format
                                   and shape (num_of_inputs, one-hot).
            y_pred(numpy.ndarray): Array of predicted labels in shape
                                   (num_of_inputs, one-hot).

        Returns:
            accuracy(float): Accuracy of the neural net.
        """
        # Sum of correct predictions.
        correct_pred = 0

        for i in range(len(y_pred)):
            # Boolean value wheter predicted label is the same as true label.
            equal = np.equal(np.argmax(y_true[i]), np.argmax(y_pred[i]))

            # Change boolean type to float and add it to sum of 
            # correct predictions.
            correct_pred += equal.astype(float)

        # Calculate accuracy.
        accuracy = (correct_pred/len(y_true))*100

        return accuracy

    def score(self, x, y_true, forward=False):
        """ Perform forward step, calculate accuracy and loss.

        Args:
            x(numpy.ndarray): Array of input data in shape
                              (num_of_inputs, vector_input).
            y_true(numpy.ndarray): Array of true labels in one-hot format
                                   and shape (num_of_inputs, one-hot).
            forward: Wheter to compute forward step on given set.

        Returns:
            accuracy(float): Accuracy of the neural net.
        """

        if forward:
            # Perform forward step.
            self._forward(x)

        # Get y_pred values from memory.
        y_pred = self.memory['a' + str(len(self.layers))]

        # Calculate accuracy.
        accuracy = self.calculate_accuracy(y_true, y_pred)

        # Calculate loss.
        cross_entropy_loss = np.sum(-y_true * np.log(y_pred))

        return accuracy, cross_entropy_loss

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

    def print_layers(self):
        """ Print layers of the neural network. """
        for i, layer in enumerate(self.layers, start=1):
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
