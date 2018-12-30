import numpy as np  
import matplotlib.pyplot as plt
from random import randint

np.random.seed(42)

# cat_images = np.random.randn(700, 2) + np.array([0, -3])  
# mouse_images = np.random.randn(700, 2) + np.array([3, 3])  
# dog_images = np.random.randn(700, 2) + np.array([-3, 3])

feature_set = np.load('x_train.npy')

# labels = np.array([0]*700 + [1]*700 + [2]*700)

# one_hot_labels = np.zeros((2100, 3))

# for i in range(2100):  
#     one_hot_labels[i, labels[i]] = 1

one_hot_labels = np.load('y_train.npy')

# plt.figure(figsize=(10,7))  
# plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap='plasma', s=100, alpha=0.5)  
# plt.show()

def relu(x):
    return np.maximum(0, x)

def relu_der(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_der(x):  
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):  
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

instances = feature_set.shape[0]  
attributes = feature_set.shape[1]  
hidden_nodes = 4  
output_labels = 3

wh = np.random.rand(attributes,hidden_nodes)  
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes,output_labels)  
bo = np.random.randn(output_labels)  
lr = 10e-4

error_cost = []

mini_batch = 64

start_idx = randint(0, instances-mini_batch)
end_idx = start_idx+20

for epoch in range(50000):  
    tmp = feature_set[start_idx:end_idx]
############# feedforward

    # Phase 1
    zh = np.dot(tmp, wh) + bh
    ah = relu(zh)

    # Phase 2
    zo = np.dot(ah, wo) + bo
    ao = softmax(zo)

########## Back Propagation

########## Phase 1
    tmp2 = one_hot_labels[start_idx:end_idx]

    dcost_dzo = ao - tmp2
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

########## Phases 2

    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = relu_der(zh)
    dzh_dwh = tmp
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh

    # Update Weights ================

    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)

    if epoch % 200 == 0:
        loss = np.sum(-tmp2 * np.log(ao))
        print('Loss function value: ', loss)
        error_cost.append(loss)