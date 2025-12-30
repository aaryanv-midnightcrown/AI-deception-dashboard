import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class DeceptionClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000)

        # -----------------------------------------
        # V1 DEMO FIX:
        # Pre-fit the scaler + classifier on
        # synthetic data so predict_proba works.
        # -----------------------------------------
        self._fit_placeholder_model()

    def _fit_placeholder_model(self):
        """
        Fit on synthetic data to initialize the pipeline.
        This is a stand-in for a real trained model.
        """
        rng = np.random.default_rng(42)

        # 300 fake SAE feature vectors
        X_fake = rng.normal(size=(300, 256))

        # Fake labels: 0=truthful, 1=fabrication, 2=withholding
        y_fake = np.array([0]*100 + [1]*100 + [2]*100)

        X_scaled = self.scaler.fit_transform(X_fake)
        self.clf.fit(X_scaled, y_fake)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X)[0]
