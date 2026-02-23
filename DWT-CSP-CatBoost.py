import torch
import torch.nn as nn
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import pywt

class EEGFeatureExtractor:
    def __init__(self, wavelet='sym6', level=5, fs=250):
        self.wavelet = wavelet
        self.level = level
        self.fs = fs

    def extract_time_freq_features(self, signal):
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        bands = coeffs[1:5] + [coeffs[0]]  # D5, D4, D3, D2, A5
        features = []
        for band in bands:
            features.extend([
                np.mean(band),
                np.mean(np.abs(band - np.mean(band))),
                np.std(band),
                np.mean(np.abs(band)),
                skew(band),
                kurtosis(band)
            ])
        return features

class CSPFeatureExtractor:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, X_class1, X_class2):
        cov1 = np.mean([np.cov(x) for x in X_class1], axis=0)
        cov2 = np.mean([np.cov(x) for x in X_class2], axis=0)
        composite_cov = cov1 + cov2
        eigvals, eigvecs = np.linalg.eigh(composite_cov)
        P = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        S1 = P @ cov1 @ P.T
        _, B = np.linalg.eigh(S1)
        self.filters_ = B.T @ P

    def transform(self, X):
        Z = self.filters_ @ X
        features = np.log(np.var(Z, axis=1) / np.sum(np.var(Z, axis=1)))
        return features[:self.n_components]

class EEGClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

# Example usage
# raw_eeg: np.array of shape (channels, samples)
def process_eeg_sample(raw_eeg, extractor):
    features = []
    for ch in raw_eeg:
        feat = extractor.extract_time_freq_features(ch)
        features.append(feat)
    return np.array(features).flatten()

# Training pipeline
# 1. Preprocess signals
# 2. Extract features using EEGFeatureExtractor + CSP
# 3. Train EEGClassifier with torch DataLoader

if __name__ == '__main__':
    pass
