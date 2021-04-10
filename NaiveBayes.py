import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_iris


class nbc():

    def get_prior(self, X, y):
        self.prior = (X.groupby(y).apply(lambda x: len(x)) / self.rows).to_numpy()

        return self.prior

    def get_statistics(self, X, y):

        self.mean = X.groupby(y).apply(np.mean).to_numpy()
        self.var = X.groupby(y).apply(np.var).to_numpy()

        return self.mean, self.var

    def gaussian(self, i, x):
        mean = self.mean[i]
        var = self.var[i]
        top = np.exp((-1 / 2) * ((x - mean) ** 2) / (2 * var))
        bot = np.sqrt(2 * math.pi * var)
        prob = top / bot
        return prob

    def get_posterior(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for i in range(len(self.classes)):
            prior = np.log(self.prior[i])
            conditional = sum(np.log(self.gaussian(i, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        # return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.classes = np.unique(y)
        self.feature_nums = X.shape[1]
        self.rows = X.shape[0]
        self.get_statistics(X, y)
        self.get_prior(X, y)

    def predict(self, X):
        X = pd.DataFrame(X)
        preds = []
        for hel in X.to_numpy():
            preds.append(self.get_posterior(hel))
        return preds

    def accuracy(self, test, pred):
        accuracy = sum(test == pred) / len(y_test)
        return accuracy





iris = load_iris()
X, y , feature_name = iris['data'], iris['target'], iris['feature_names']

N,D = X.shape
Ntrain = int(0.8*N)
shuffler = np.random.permutation(N)

X_train = X[shuffler[:Ntrain]]
y_train = y[shuffler[:Ntrain]]
X_test = X[shuffler[Ntrain:]]
y_test = y[shuffler[Ntrain:]]


nbc = nbc()
nbc.fit(X_train,y_train)

Predictions = nbc.predict(X_test)
print(nbc.accuracy(y_test,Predictions))



#print(calc_prior(X,y))

