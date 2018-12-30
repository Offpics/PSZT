import numpy as np

class MLP():
    def __init__(self):
        self.layers = []
        
        # Dictionary that holds w,b params.
        self.param_values = {}

        # Dictionary that holds a, z params.
        self.memory = {}

        # Dictionary that holds gradients.
        self.grad_values = {}

        self.lr = 10e-4

    def init_layers(self):
        """ Initialize weights in every layer. """

        for i, layer in enumerate(self.layers, start=1):
            input_size = layer["input_dim"]
            output_size = layer["output_dim"]

            self.param_values['w' + str(i)] = np.random.rand(
                input_size, output_size) * 0.1
            self.param_values['b' + str(i)] = np.random.randn(
                output_size) * 0.1

    def forward(self, x):
        # Save input values to the a 0, as an output of input layer.
        self.memory['a0'] = x
        a_curr = x
        for i, layer in enumerate(self.layers, start=1):
            a_prev = a_curr
            # Get activation function
            activation_name = layer["activation"]
            act_func = self.activation_function(activation_name)

            w = self.param_values['w' + str(i)]
            b = self.param_values['b' + str(i)]

            ## IN NP.DOT(a_prev, wo)
            z = np.dot(a_prev, w) + b
            a = act_func(z)
            a_curr = a

            self.memory['a' + str(i)] = a
            self.memory['z' + str(i)] = z


    def backward(self, x, y_true):
        for i, layer in reversed(list(enumerate(self.layers, start=1))):
            if i == len(self.layers):
                # Only calculate this if layer is softmax
                a = self.memory['a' + str(i)]
                a_prev = self.memory['a' + str(i-1)]

                dcost_dz = a - y_true #dcost/dzo = dcost/dao * dao/dzo
                dz_dw = a_prev
                dcost_w = np.dot(dz_dw.T, dcost_dz)
                dcost_b = dcost_dz

                self.grad_values['dcost_dz' + str(i)] = dcost_dz
                self.grad_values['dcost_w' + str(i)] = dcost_w
                self.grad_values['dcost_b' + str(i)] = dcost_b

            else:
                # Calculate for the rest of the layers.
                z = self.memory['z' + str(i)]
                w_prev = self.param_values['w' + str(i+1)]
                dcost_dz_prev = self.grad_values['dcost_dz' + str(i+1)]
                a_prev = self.memory['a' + str(i-1)]

                activation_name = layer["activation"]
                act_func_der = self.activation_function_der(activation_name)
                
                dz_da = w_prev
                dcost_da = np.dot(dcost_dz_prev, dz_da.T)
                da_dz = act_func_der(z) 

                #dcost/dzh = dcost/dah * dah/dzh
                dcost_dz = da_dz * dcost_da
                
                dz_dw = a_prev  # input values of the previous layer
                dcost_w = np.dot(dz_dw.T, dcost_dz)
                dcost_b = dcost_dz

                self.grad_values['dcost_dz' + str(i)] = dcost_dz
                self.grad_values['dcost_w' + str(i)] = dcost_w
                self.grad_values['dcost_b' + str(i)] = dcost_b

    def update_weights(self):
        for i, _ in enumerate(self.layers, start=1):
            dcost_w = self.grad_values['dcost_w' + str(i)]
            dcost_b = self.grad_values['dcost_b' + str(i)]

            self.param_values['w' + str(i)] -= self.lr * dcost_w
            self.param_values['b' + str(i)] -= self.lr * dcost_b.sum(axis=0)

    def train(self, x, y_true):
        self.forward(x)
        self.backward(x, y_true)
        self.update_weights()

        # Calculate cross_entropy_loss
        y_pred = self.memory['a' + str(len(self.layers))]
        cross_entropy_loss = np.sum(-y_true * np.log(y_pred))

        # Calculate accuracy
        correct_pred = 0
        for i in range(len(y_pred)):
            equal = np.equal(np.argmax(y_true[i]), np.argmax(y_pred[i]))
            correct_pred += equal.astype(float)
        accuracy = (correct_pred/len(y_true))*100

        print(f'loss: {cross_entropy_loss}, accuracy: {accuracy:.1f}%')


    def set_lr(self, lr):
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
        for i, layer in enumerate(self.layers, start = 1):
            print(f'{1}. Layer - input_dim: {layer["input_dim"]}, output_dim: {layer["output_dim"]}, activation: {layer["activation"]}')



    def activation_function(self, name):
        if name == 'relu':
            return self.relu
        elif name == 'sigmoid':
            return self.sigmoid
        elif name == 'softmax':
            return self.softmax


    def activation_function_der(self, name):
        if name == 'relu':
            return self.relu_der
        elif name == 'sigmoid':
            return self.sigmoid_der


    def relu(self, x):
        return np.maximum(0, x)


    def relu_der(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x


    def sigmoid(self, x):  
        return 1/(1+np.exp(-x))


    def sigmoid_der(self, x):  
        return self.sigmoid(x) *(1-self.sigmoid(x))


    def softmax(self, A):  
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)

