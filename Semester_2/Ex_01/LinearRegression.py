import numpy as np
import math
import sklearn.datasets as skd
import matplotlib.pyplot as plt
import sns


class LinearRegression:
    def __init__(self) -> None:
        self.w = np.array([0, 0])
        self.loss_list = []

    def fit(self, D, epochs: int, learning_rate=0.1):
        for i in range(epochs):
            train_loss, squared_loss = 0, 0
            for x, y in D:
                train_loss += self.gradient(x, y)
                squared_loss += self.squared_loss(x, y)
            train_loss = train_loss / len(D)
            squared_loss /= len(D)
            self.w = self.w - learning_rate * train_loss
            self.loss_list.append(squared_loss)
            print(f"Epoch: {i + 1} Weights: {self.w} Squared Loss: {squared_loss}")

    def phi(self, x):
        return np.array([1, x])

    def loss(self, x: float, y: float):
        p = self.phi(x)
        fw_x = np.dot(self.w, p)
        return fw_x - y

    def squared_loss(self, x: float, y: float):
        return math.pow(self.loss(x, y), 2)

    def gradient(self, x: float, y: float):
        return (2 * self.loss(x, y)) * self.phi(x)

    def predict(self, x):
        return self.w[0] + x * self.w[1]

    def plot_loss(self):
        sns.set_(style="darkgrid")
        sns.lineplot(x=range(len(self.loss_list)), y=self.loss_list)
        plt.show()

    def plot_line(self, data):
        sns.set_theme(style="darkgrid")
        x = [x for x, _ in data]
        y = [y for _, y in data]
        x_range = range(round(max(x)) + 2)
        func_values = [self.predict(i) for i in x_range]
        sns.lineplot(x=x_range, y=func_values)
        plt.scatter(x, y, marker=".")
        plt.show()

epochs = 500
D = np.array([(1, 1), (2, 3), (4, 3)])

lr = LinearRegression()
lr.fit(D, epochs)
lr.predict(1)
lr.plot_loss()
lr.plot_line(D)

epochs = 5000
sk_data = skd.make_regression(n_samples=100, n_features=2)
toy_dataset = np.abs(np.array(sk_data[0]))

lr = LinearRegression()
lr.fit(toy_dataset, epochs)
lr.plot_loss()
lr.plot_line(toy_dataset)
