from sklearn.naive_bayes import GaussianNB


class BayesClassifier:

    def __init__(self):
        self.model = GaussianNB()
        self.x = None
        self.y = None

    def set_training_data(self, data):
        self.x = []
        self.y = []
        for row in data:
            self.x.append(row[:-1])
            self.y.append(row[-1])

    def set_xy(self, x, y):
        self.x = x
        self.y = y

    def learn(self):
        self.model.fit(self.x, self.y)

    def predict(self, data):
        return self.model.predict(data)
