import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class DeceptionClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000)

        # Pre-fit on synthetic data so the demo works immediately
        self._fit_placeholder_model()

    def _fit_placeholder_model(self):
        """
        Fit the scaler and classifier on synthetic data.
        This avoids runtime training and keeps the demo stable.
        """
        rng = np.random.default_rng(42)

        # 300 fake SAE feature vectors (matches SAE hidden_dim=256)
        X_fake = rng.normal(size=(300, 256))

        # Labels: 0=truthful, 1=fabrication, 2=withholding
        y_fake = np.array([0]*100 + [1]*100 + [2]*100)

        X_scaled = self.scaler.fit_transform(X_fake)
        self.clf.fit(X_scaled, y_fake)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X)[0]
