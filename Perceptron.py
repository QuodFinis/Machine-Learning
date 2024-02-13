import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# activation functions
def binary_step(z):
    return 1 if z > 0 else 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class Perceptron:
    def __init__(self, learning_rate, epochs, model_data, activation_fnx, test_data=None):
        self.alpha = learning_rate
        self.epochs = epochs
        self.model_data = model_data
        self.weights = np.zeros(model_data.shape[1])
        self.test_data = test_data
        self.activation_function = activation_fnx

    def predict(self, row):
        z = np.dot(row, self.weights[:-1]) + self.weights[-1]
        y = self.activation_function(z)
        return y

    def get_error(self, row, prediction):
        return row[-1] - prediction

    def reweight(self, error, row):
        for i in range(len(self.weights) - 1):
            self.weights[i] += self.alpha * error * row[i]

        self.weights[-1] += self.alpha * error

    def fit(self):
        for _ in range(self.epochs):
            error_sum = 0
            for row in self.model_data:
                y = self.predict(row[:-1])
                error = self.get_error(row, y)
                error_sum += abs(error)
                self.reweight(error, row)
            if error_sum <= 0.000001:
                break

        return self.weights

    def show_plot(self, data=None):
        if data == "test":
            plt.scatter(self.test_data[:, 0], self.test_data[:, 1], c=self.test_data[:, 2])
        else:
            plt.scatter(self.model_data[:, 0], self.model_data[:, 1], c=self.model_data[:, 2])

        x = np.linspace(-0.5, 1.5, 100)
        y = (-self.weights[2] - self.weights[0] * x) / self.weights[1]
        plt.plot(x, y, '-r', label='decision boundary')
        plt.axis('auto')  # Automatically adjust plot limits
        plt.show()

    def test(self):
        correct_predictions = 0
        total_predictions = len(self.test_data)
        for row in self.test_data:
            prediction = self.predict(row[:-1])
            if prediction * row[-1] > 0:
                correct_predictions += 1
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy


data = np.array([[0, 0, 1],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0]])

test_data = np.array([[0.5, 0.5, 1],
                      [0.5, 1.5, 0],
                      [1.5, 0.5, 0],
                      [1.5, 1.5, 0]])

nand_percept = Perceptron(learning_rate=0.1,
                          epochs=1000,
                          model_data=data,
                          activation_fnx=sigmoid,
                          test_data=test_data)

nand_percept.fit()
nand_percept.show_plot()
nand_percept.show_plot(data="test")
print(nand_percept.test())


data = pd.read_csv('datasets/iris.csv', header=None)

# print(data.head())
"""
     0    1    2    3            4
0  5.1  3.5  1.4  0.2  Iris-setosa
1  4.9  3.0  1.4  0.2  Iris-setosa
2  4.7  3.2  1.3  0.2  Iris-setosa
3  4.6  3.1  1.5  0.2  Iris-setosa
4  5.0  3.6  1.4  0.2  Iris-setosa
"""

# sepal length, sepal width, petal length, petal width, class
# print(data[0][0], data[1][0], data[2][0], data[3][0], data[4][0])
# 5.1, 3.5, 1.4, 0.2, Iris-setosa

data = data.transpose()
# print(data[0])
"""
0            5.1
1            3.5
2            1.4
3            0.2
4    Iris-setosa
"""

# input
x = [1] + data[0].tolist()[0:4]  # adding bias
x_class = data[0].tolist()[-1]  # class
# x, x_class = [1, 5.1, 3.5, 1.4, 0.2], Iris-setosa

# initial weights w0-w4
weights = [0.0, 0.0, 0.0, 0.0, 0.0]
weights = np.array(weights)
weights = weights.transpose()

# learning rate
alpha = 0.01


# activation function
def activation_function(z):
    if z > 0:
        return 1
    else:
        return 0
