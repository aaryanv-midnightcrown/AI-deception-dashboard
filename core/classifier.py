import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class DeceptionClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X)[0]
