import np as np
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


tr_inputs = np.array([[ 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
                      [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
                      [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]])
tr_outputs = np.array([[312, 213, 123, 231]]).T

# tr_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
# tr_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)
sp_ts = np.random.random((12,3))-1

print('случайные инициализирующие')
print(sp_ts)

# метод обратного распространения
for i in range(20000):
    input_layers = tr_inputs
    outputs = sigmoid(np.dot(input_layers, sp_ts))

    err = tr_outputs - outputs
    adjustments = np.dot(input_layers.T, err * (outputs * (1 - outputs)))

    sp_ts += adjustments

print('after training')
print(sp_ts)

print('result')
print(outputs)

# test
new_inputs = np.array([ 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1])
outputs = sigmoid(np.dot(new_inputs, sp_ts))

print('new situation')
print(outputs)
