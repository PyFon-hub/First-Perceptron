import numpy as np
from numpy import random

def sigmoid(x):
    return 1/ (1 + np.e**(-x))

def sigmoid_deverative(x):
    return x*(1-x)

train_input = np.array([[0,0,1],
                       [1,1,1],
                       [1,0,1],
                       [0,1,1]])

train_output = np.array([[0,1,1,0]]).T

random.seed(1)
synaptic_weight = random.random((3,1))


#print(f"synaptic_weight: \n", synaptic_weight)

for iteration in  range(10000):

    input_layer = train_input

    output = sigmoid(np.dot(train_input, synaptic_weight))

    error = train_output - output

    adjustments = error * sigmoid_deverative(output)

    synaptic_weight  += np.dot(input_layer.T, adjustments)

print(f"Waga po treningu: \n", synaptic_weight)
print(f"Wynik po treningu: \n", output)

print(f"Wynik inputu: ", sigmoid(np.dot(np.array([[1,0,0]]), synaptic_weight)))





















