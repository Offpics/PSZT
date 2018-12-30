import numpy as np
from neural_net import MLP
from activations import softmax_grad


if __name__ == "__main__":
    model = MLP()

    model.init_weights()

    x_train = np.load('x_train.npy')
    # print(x_train[0])

    y_hat = model.forward_propagation(x_train[0])
   
    y_train = np.load('y_train.npy')

    grad_values = model.backward_propagation(y_hat, y_train[0])
    # # cross = model.cross_entropy(xd, y_train[0])
    
    # # print(cross)
    
    # m = y_train.shape[0]
    # p = np.array([[0.23735235], [0.54504666], [0.21760099]])
    # p = np.squeeze(p)
    # y = y_train[0]

    # cross_entropy = 0
    # for i in range(len(p)):
    #     print(f'y_train[{i}]: {y[i]}, p[{i}]: {p[i]}')
    #     cross_entropy += y[i] * np.log(p[i])
    #     print(f'cross_entropy: {cross_entropy}')
    
    # cross_entropy = -np.squeeze(cross_entropy)

    # print(cross_entropy)

    # print(np.argmax(p))

    # print(model.memory["Z4"])
    # print(model.memory["Z4"].sum())