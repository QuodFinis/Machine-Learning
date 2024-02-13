import numpy as np
import matplotlib.pyplot as plt

# creating simple perceptron for NAND gate
# input1, input2, output
data = np.array([[0, 0, 1],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0]])

# AND gate
data = np.array([[0, 0, 0],
                 [0, 1, 0],
                 [1, 0, 0],
                 [1, 1, 1]])

# perceptron will take the inputs and try to predict the output
weights = np.array([0.0, 0.0, 0.0])

alpha = 0.1


def activation_fnx(z):
    return 1 if z >= 0 else 0


for _ in range(1000):
    error_sum = 0
    for row in data:
        z = np.dot(row[0:2], weights[0:2]) + weights[2]
        y = activation_fnx(z)
        error = row[2] - y
        error_sum += abs(error)
        weights[0] = weights[0] + alpha * error * row[0]
        weights[1] = weights[1] + alpha * error * row[1]
        weights[2] = weights[2] + alpha * error
    if error_sum <= 0.000001:
        break

print(weights)

# Plot the data
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
x = np.linspace(-0.5, 1.5, 100)
y = (-weights[2] - weights[0] * x) / weights[1]
# what exactly is y, how exactly do i plot the line or hyper line of y
plt.plot(x, y, '-r', label='decision boundary')
plt.ylim([-0.5, 1.5])
plt.xlim([-0.5, 1.5])
plt.show()

