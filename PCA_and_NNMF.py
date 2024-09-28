import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.datasets import load_digits

# Load dataset (example: handwritten digits dataset)
digits = load_digits()
X = digits.data
y = digits.target

# Initialize lists to store reconstruction errors
pca_errors = []
nmf_errors = []
components_range = range(1, 65)  # We'll test from 1 up to 64 components

# Loop over different numbers of components to compute reconstruction errors
for n_components in components_range:
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_pca_reconstructed = pca.inverse_transform(X_pca)
    pca_error = np.mean((X - X_pca_reconstructed) ** 2)
    pca_errors.append(pca_error)

    # NNMF
    nmf = NMF(n_components=n_components, init='random', random_state=0)
    X_nmf = nmf.fit_transform(X)
    X_nmf_reconstructed = np.dot(X_nmf, nmf.components_)
    nmf_error = np.mean((X - X_nmf_reconstructed) ** 2)
    nmf_errors.append(nmf_error)

# Plot reconstruction error vs number of components for PCA and NNMF
plt.figure()
plt.plot(components_range, pca_errors, label='PCA Reconstruction Error', color='blue')
plt.plot(components_range, nmf_errors, label='NNMF Reconstruction Error', color='green')
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs Number of Components')
plt.legend()
plt.grid(True)
plt.show()

# PCA Cumulative Variance Plot
pca = PCA(n_components=64)
pca.fit(X)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(components_range, cumulative_variance[:64], label='PCA Cumulative Variance', color='blue')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('PCA: Cumulative Variance Explained')
plt.grid(True)
plt.show()

# NNMF approximation to Cumulative Variance
# Since NNMF doesn't have direct variance, we'll use the explained variance from PCA as a rough approximation.
explained_variance_nmf = []

for n_components in components_range:
    nmf = NMF(n_components=n_components, init='random', random_state=0)
    X_nmf = nmf.fit_transform(X)
    X_nmf_reconstructed = np.dot(X_nmf, nmf.components_)
    explained_variance_nmf.append(1 - np.mean((X - X_nmf_reconstructed)**2) / np.var(X))

# Plot the "cumulative variance" approximation for NNMF
plt.figure()
plt.plot(components_range, explained_variance_nmf, label='NNMF Approximate Variance Explained', color='green')
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained (Approximation)')
plt.title('NNMF: Approximate Variance Explained')
plt.grid(True)
plt.show()
