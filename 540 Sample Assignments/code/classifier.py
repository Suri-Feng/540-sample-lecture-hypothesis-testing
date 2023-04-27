import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA:
    def __init__(self, X=None, y=None):
        self.clf = LinearDiscriminantAnalysis()
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        self.X = X
        self.y = y
        self.clf.fit(self.X, self.y)

    def predict(self, Xtest):
        if len(Xtest.shape) == 1:
            Xtest = Xtest.to_numpy()[:, np.newaxis]
        return self.clf.predict(Xtest)
