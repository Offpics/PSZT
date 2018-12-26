import numpy as np
from neural_net import MLP


if __name__ == "__main__":
    model = MLP()

    model.init_weights()

    x_train = np.load('x_train.npy')
    # print(x_train[0])

    model.forward_propagation(x_train[0])

    print(model.memory["Z4"].shape)