import numpy as np

inputs = np.arange(1, 7).reshape((2, 3))
print('Inputs:\n', inputs, "\n")
dvalues = np.arange(10, 16).reshape((2, 3))
print(len(dvalues[0]))

weights = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
dweights = np.matmul(inputs.T, dvalues)
dbiases =  np.sum(dvalues, axis=0, keepdims=True)
dinputs = np.dot(dvalues, weights.T)

print(dweights, "\n")
print(dbiases, "\n")
print(dinputs, "\n")