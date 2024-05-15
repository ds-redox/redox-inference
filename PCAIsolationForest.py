import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator

class PCAIsolationForest(BaseEstimator):
    def __init__(self, n_pca_components=1, n_estimators = 100, random_state = None, contamination='auto', bootstrap=False):
        self.n_pca_components = n_pca_components
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.contamination = contamination
        self.bootstrap = bootstrap
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state, contamination=contamination, bootstrap=bootstrap)
        self.scaler = StandardScaler(with_std=True, with_mean=True)
        self.feature_names_in_ = []

    def _get_pca_scores(self, data):
        scaled = self.scaler.fit_transform(data)
        pca = PCA(self.n_pca_components)
        return pca.fit_transform(scaled)
    
    def fit(self, X, y=None):
        data = self._get_pca_scores(X)
        self.model.fit(data, y)
        self.feature_names_in_ = np.array(X.columns)
        return self

    def predict(self, X):
        data = self._get_pca_scores(X)
        anomaly = self.model.predict(data)
        return (anomaly == -1)