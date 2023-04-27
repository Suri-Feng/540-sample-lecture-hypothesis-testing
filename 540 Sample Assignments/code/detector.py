from scipy.stats import norm
import math


class NP:
    def __init__(self, alpha=0.05, mu0=-1, mu1=1, sigma=1):
        self.threshold = None
        self.alpha = alpha
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma
        self.fit()

    def fit(self):
        self.threshold = self.sigma * norm.ppf(1 - self.alpha) + self.mu0

    def predict(self, Xtest):
        return Xtest >= self.threshold


class Bayes:
    def __init__(self, theta=0.6, mu0=-1, mu1=1, sigma=1):
        self.threshold = None
        self.theta = theta
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma
        self.fit()

    def fit(self):
        self.threshold = self.sigma ** 2 / (self.mu1 - self.mu0) * math.log((1 - self.theta) / self.theta) \
                         + (self.mu1 + self.mu0) / 2

    def predict(self, Xtest):
        return Xtest >= self.threshold
