import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator

class PCAGaussianMix(BaseEstimator):
    def __init__(self, n_gm_components=3, n_pca_components=4, outliers_fraction = 0.10, random_state = None, covariance_type = 'full'):
        self.n_gm_components = n_gm_components
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        self.outliers_fraction = outliers_fraction
        self.covariance_type = covariance_type
        self.model = GaussianMixture(n_components=n_gm_components, random_state=random_state)
        self.scaler = StandardScaler(with_std=True, with_mean=True)
        self.feature_names_in_ = []

    def _get_pca_scores(self, pca_data):
        scaled = self.scaler.fit_transform(pca_data)
        pca = PCA(self.n_pca_components)
        return pca.fit_transform(scaled)
    
    def fit(self, X, y=None):
        data = self._get_pca_scores(X)
        self.model.fit(data, y)
        self.feature_names_in_ = np.array(X.columns)
        return self

    def predict(self, X):
        data = self._get_pca_scores(X)
        scaled_pca_scores = self.scaler.fit_transform(data)
        data = pd.DataFrame(scaled_pca_scores)

        scores = self.model.score_samples(data)
        outlier_num = int(self.outliers_fraction * len(scores))
        idx = np.argpartition(scores, outlier_num)
        threshold = scores[idx[outlier_num-1]]
        return (scores <= threshold)