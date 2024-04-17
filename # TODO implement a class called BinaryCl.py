# TODO implement a class called BinaryClassifier
# implement a method called predict that takes two arguments x1 and x2
import numpy as np

class BinaryClassifier:

    def _init_(self, w1, w2):  # w1 = 15.0, w2 = 4.5
        self.w1 = w1
        self.w2 = w2
        self.w = np.array([w1, w2])

    # feature extractor:
    def phi(self, x1: float, x2: float):  # x1 = 2, x2 = -2
        phi_x = np.array([x1, x2])
        return phi_x

    def getscore(self, x1, x2):  # calculating score
        score = np.dot(self.w, self.phi(x1, x2))
        return score

    def getsign(self, var):
        if var > 0:
            return 1
        elif var < 0:
            return -1
        else:
            return 0

    def getLabel(self, x1, x2):
        return self.getsign(self.getscore(x1, x2))

    def predict(self, x1, x2):
        return self.getLabel(x1, x2)

    def var(self, x2, y):
        value = x2 and y
        return value

    def verbosePrediction(self, x1, x2, y):

        print(f"Predicted Label: {self.getLabel(x1, x2)}, Target Label: {y}")

        if self.getLabel(x1, x2) == y:
            print("Correct")
        else:
            print("Incorrect")

    def margin(self, x1, x2, y):
        marg = (self.w1 * x1 + self.w2 * x2) * y
        return marg

    def getHingeLoss(self, x1, x2, y, gap=1):
        loss_hinge = max(gap - self.margin(x1, x2, y), 0)
        return loss_hinge

    def train(self, x1, x2, y, eta=0.1):

        if self.getHingeLoss(x1, x2, y, gap=1) > 0:

            if self.margin(x1, x2, y) <= 0:
                w1_change = x1 * y
                w2_change = x2 * y

                self.w1 = self.w1 - eta * w1_change
                self.w2 = self.w2 - eta * w2_change

    def printStats(self, x1, x2, y):
        new_label = self.getLabel(x1, x2)

        if new_label == y:
            print("Correct")
        else:
            print("Incorrect")


data = [(0.5, 0.5), (2, 0), (-1, 1), (1, -1), (1, -2), (-1, -1)]
labels = [1, 1, 1, -1, -1, -1]
n_epoch = 100
eta = 0.1
gap = 1

classifier = BinaryClassifier(15.0, 4.5)

for i in range(6):
    print("Training", i + 1, ":")
    classifier.verbosePrediction(data[i][0], data[i][1], labels[i])

for m in range(6):
    print("Margin calculated for (2, -2) :", classifier.margin(2, -2, labels[m]))

for n in range(6):
    print("Margin for training data :", classifier.margin(data[n][0], data[n][1], labels[n]))

for b in range(6):
    print("Hinge Loss for (2, -2): ", classifier.getHingeLoss(2, -2, labels[b], 1))

for epoch in range(n_epoch):
    trainLoss = 0
    for i in range(len(data)):
        classifier.train(data[i][0], data[i][1], labels[i], eta)
        trainLoss += classifier.getHingeLoss(data[i][0], data[i][1], labels[i], gap)
        print("Avrage trainloss: ", trainLoss / len(data))
        classifier.printStats(data[i][0], data[i][1], labels[i])