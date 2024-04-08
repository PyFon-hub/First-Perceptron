import numpy as np
from numpy import random

class Perceptron():
    def __init__(self) -> None:
        
        random.seed(1)
        self.synaptic_weight = random.random((3,1))

    def __sigmoid(self, x):
        return 1/ (1 + np.e**(-x))

    def __sigmoid_deverative(self, x):
        return x*(1-x)
    
    def train(self, training_input, training_output, epoch):

        for _ in range(epoch):
            
            output = self.think(training_input)
            error = training_output - output

            adjustments = error * self.__sigmoid_deverative(output)

            self.synaptic_weight  += np.dot(training_input.T, adjustments)

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.__sigmoid(np.dot(inputs, self.synaptic_weight))
        return output



if __name__ == "__main__":

    train_input = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1]])

    train_output = np.array([[0,1,1,0]]).T

    perceptron = Perceptron()
    perceptron.train(train_input, train_output, 10000)
    print(f"Answer; ", perceptron.think(np.array([[1,0,0]])) )
    



















