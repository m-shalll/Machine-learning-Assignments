# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


# %% [markdown]
# # PCA

# %%
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # 1. Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues & eigenvectors (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # 5. Select top components
        self.eigenvalues = eigenvalues[:self.n_components]
        self.components = eigenvectors[:, :self.n_components]

        # 6. Explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_variance

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def inverse_transform(self, X_reduced):
        return np.dot(X_reduced, self.components.T) + self.mean

    def reconstruction_error(self, X):
        X_reduced = self.transform(X)
        X_reconstructed = self.inverse_transform(X_reduced)
        return np.mean((X - X_reconstructed) ** 2)



# %% [markdown]
# # Autoencoder

# %%
class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

class Autoencoder:
    def __init__(
        self,
        layer_sizes,
        activation="relu",
        learning_rate=0.01,
        l2_lambda=0.0,
        lr_decay=0.0
    ):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.l2_lambda = l2_lambda
        self.lr_decay = lr_decay

        self.weights = []
        self.biases = []

        self.activations = []
        self.z_values = []

        # Activation selection
        if activation == "relu":
            self.act = Activation.relu
            self.act_deriv = Activation.relu_derivative
        elif activation == "sigmoid":
            self.act = Activation.sigmoid
            self.act_deriv = Activation.sigmoid_derivative
        elif activation == "tanh":
            self.act = Activation.tanh
            self.act_deriv = Activation.tanh_derivative
        else:
            raise ValueError("Unsupported activation")

        self._initialize_parameters()

    def _initialize_parameters(self):
        for i in range(len(self.layer_sizes) - 1):
            weight = np.random.randn(
                self.layer_sizes[i],
                self.layer_sizes[i + 1]
            ) * np.sqrt(2 / self.layer_sizes[i])
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for W, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], W) + b
            self.z_values.append(z)
            a = self.act(z)
            self.activations.append(a)

        return self.activations[-1]

    def compute_loss(self, X, X_hat):
        mse = np.mean((X - X_hat) ** 2)
        l2_penalty = self.l2_lambda * sum(np.sum(W ** 2) for W in self.weights)
        return mse + l2_penalty

    def backward(self, X):
        grads_W = []
        grads_b = []

        # MSE loss derivative
        delta = (self.activations[-1] - X) * self.act_deriv(self.z_values[-1])

        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.activations[i].T, delta)
            dB = np.sum(delta, axis=0, keepdims=True)

            # L2 regularization
            dW += self.l2_lambda * self.weights[i]

            grads_W.insert(0, dW)
            grads_b.insert(0, dB)

            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.act_deriv(self.z_values[i - 1])

        return grads_W, grads_b

    def update_parameters(self, grads_W, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_W[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train(self, X, epochs=100, batch_size=32):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch = X_shuffled[start:end]

                X_hat = self.forward(batch)
                grads_W, grads_b = self.backward(batch)
                self.update_parameters(grads_W, grads_b)

            # Learning rate scheduling
            self.learning_rate = self.initial_lr / (1 + self.lr_decay * epoch)

            if epoch % 10 == 0:
                loss = self.compute_loss(X, self.forward(X))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def encode(self, X):
        for i in range(len(self.weights) // 2):
            X = self.act(np.dot(X, self.weights[i]) + self.biases[i])
        return X

    def reconstruct(self, X):
        return self.forward(X)



# %% [markdown]
# # Internal Metrics

# %% [markdown]
# ### Silhouette Score

# %%
def silhouette_score(X, labels):
    X = np.asarray(X)
    labels = np.asarray(labels)

    n_samples = X.shape[0]
    unique_labels = np.unique(labels)

    distances = np.linalg.norm(
        X[:, np.newaxis, :] - X[np.newaxis, :, :],
        axis=2
    )

    silhouette_values = np.zeros(n_samples)

    for i in range(n_samples):
        same_cluster = labels == labels[i]
        other_clusters = unique_labels[unique_labels != labels[i]]

        if np.sum(same_cluster) > 1:
            a_i = np.mean(distances[i, same_cluster & (np.arange(n_samples) != i)])
        else:
            a_i = 0.0

        b_i = np.inf
        for cluster in other_clusters:
            cluster_mask = labels == cluster
            b_i = min(b_i, np.mean(distances[i, cluster_mask]))

        silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(silhouette_values)


# %% [markdown]
# ### Davies-Bouldin Index

# %%
def davies_bouldin_index(X, labels, centroids):
    X = np.asarray(X)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids)

    k = centroids.shape[0]

    S = np.zeros(k)
    for i in range(k):
        cluster_points = X[labels == i]
        S[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))

    centroid_distances = np.linalg.norm(
        centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )

    dbi = 0.0
    for i in range(k):
        R_ij = []
        for j in range(k):
            if i != j:
                R_ij.append((S[i] + S[j]) / centroid_distances[i, j])
        dbi += max(R_ij)

    return dbi / k


# %% [markdown]
# ### Calinski–Harabasz Index

# %%
def calinski_harabasz_index(X, labels, centroids):
    X = np.asarray(X)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids)

    n_samples = X.shape[0]
    k = centroids.shape[0]

    overall_mean = np.mean(X, axis=0)

    W = 0.0
    for i in range(k):
        cluster_points = X[labels == i]
        W += np.sum((cluster_points - centroids[i]) ** 2)

    B = 0.0
    for i in range(k):
        n_i = np.sum(labels == i)
        B += n_i * np.sum((centroids[i] - overall_mean) ** 2)

    return (B / (k - 1)) / (W / (n_samples - k))


# %% [markdown]
# ### Within-cluster sum of squares (WCSS)

# %%
def wcss(X, labels, centroids):
    X = np.asarray(X)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids)

    total = 0.0
    for i in range(centroids.shape[0]):
        cluster_points = X[labels == i]
        total += np.sum((cluster_points - centroids[i]) ** 2)

    return total


# %% [markdown]
# # External Metrics

# %% [markdown]
# First, we have to implement the contingency table

# %%
def contingency_table(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    clusters = np.unique(labels_pred)
    classes = np.unique(labels_true)

    table = np.zeros((clusters.size, classes.size), dtype=int)

    for i, cluster in enumerate(clusters):
        for j, cls in enumerate(classes):
            table[i, j] = np.sum(
                (labels_pred == cluster) & (labels_true == cls)
            )

    return table


# %% [markdown]
# ### Adjusted Rand Index

# %%
def nC2(n):
    return n * (n - 1) / 2

def adjusted_rand_index(labels_true, labels_pred):
    table = contingency_table(labels_true, labels_pred)
    n = np.sum(table)

    sum_nij = np.sum(nC2(table))
    sum_ai = np.sum(nC2(np.sum(table, axis=1)))
    sum_bj = np.sum(nC2(np.sum(table, axis=0)))
    total_pairs = nC2(n)

    expected_index = (sum_ai * sum_bj) / total_pairs
    max_index = 0.5 * (sum_ai + sum_bj)

    return (sum_nij - expected_index) / (max_index - expected_index)


# %% [markdown]
# ### Normalized Mutual Information

# %%
def normalized_mutual_information(labels_true, labels_pred, eps=1e-10):
    table = contingency_table(labels_true, labels_pred)
    n = np.sum(table)

    P_ij = table / n
    P_i = np.sum(P_ij, axis=1)
    P_j = np.sum(P_ij, axis=0)

    MI = 0.0
    for i in range(P_ij.shape[0]):
        for j in range(P_ij.shape[1]):
            if P_ij[i, j] > 0:
                MI += P_ij[i, j] * np.log(P_ij[i, j] / (P_i[i] * P_j[j] + eps))

    H_i = -np.sum(P_i * np.log(P_i + eps))
    H_j = -np.sum(P_j * np.log(P_j + eps))

    return MI / np.sqrt(H_i * H_j)


# %% [markdown]
# ### Purity

# %%
def purity_score(labels_true, labels_pred):
    table = contingency_table(labels_true, labels_pred)
    return np.sum(np.max(table, axis=1)) / np.sum(table)


# %% [markdown]
# ### Majority vote mapping for confusion matrix

# %%
def majority_vote_mapping(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    mapping = {}
    mapped_predictions = np.zeros_like(labels_pred)

    for cluster in np.unique(labels_pred):
        true_labels_in_cluster = labels_true[labels_pred == cluster]

        values, counts = np.unique(true_labels_in_cluster, return_counts=True)
        majority_label = values[np.argmax(counts)]

        mapping[cluster] = majority_label
        mapped_predictions[labels_pred == cluster] = majority_label

    return mapping, mapped_predictions


# %% [markdown]
# # K-Means

# %% [markdown]
# ### K-Means Class Implementation

# %%
class KMeans:
    def __init__(self, n_clusters=2, init="kmeans++", max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters # number of clusters
        self.init = init # initialization method: "random" or "kmeans++"
        self.max_iter = max_iter # maximum number of iterations
        self.tol = tol # convergence threshold on centroid movement
        self.random_state = random_state # random seed

        self.centroids_ = None # final centroids of clusters
        self.labels_ = None # cluster assignments for each point
        self.inertia_ = None # final within-cluster sum of squares
        self.inertia_history_ = [] # history of inertia values
        self.n_iter_ = 0 # number of iterations run
        self.converged_ = False # whether convergence was achieved early

        if random_state is not None:
            np.random.seed(random_state)

    def _init_random(self, X):
        n_samples = X.shape[0]
        # choose k distinct points
        # replace = False --> ensure that no 2 start indices are the same
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]

    def _init_kmeans_plus_plus(self, X):
        # this algorithm improves convergence speed and stability
        n_samples, n_features = X.shape
        centroids = np.empty((self.n_clusters, n_features))

        # pick the first sample randomly
        idx = np.random.randint(n_samples)
        centroids[0] = X[idx]

        for i in range(1, self.n_clusters):
            # calculate squared distances from nearest existing centroid
            distances = np.min(
                np.sum((X[:, np.newaxis, :] - centroids[:i]) ** 2, axis=2),
                axis=1
            )

            # compute probabilities proportional to squared distances
            # this makes the further points more likely to be chosen
            probabilities = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()

            next_idx = np.searchsorted(cumulative_probs, r)
            centroids[i] = X[next_idx]

        return centroids

    def _initialize_centroids(self, X):
        if self.init == "random":
            return self._init_random(X)
        elif self.init == "kmeans++":
            return self._init_kmeans_plus_plus(X)
        else:
            raise ValueError("init must be 'random' or 'kmeans++'")

    # Assign each point to the nearest centroid (E-step)
    def _assign_clusters(self, X, centroids):
        # compute squared distances to centroids
        distances = np.sum(
            (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )
        return np.argmin(distances, axis=1)

    # Update centroids based on current assignments (M-step)
    def _update_centroids(self, X, labels):
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                # if a cluster has no points assigned, reinitialize its centroid randomly
                centroids[k] = X[np.random.randint(X.shape[0])]
            else:
                # calculate mean of assigned points
                centroids[k] = np.mean(cluster_points, axis=0)

        return centroids

    def _compute_inertia(self, X, labels, centroids):
        inertia = 0.0
        # add up squared distances of points to their assigned centroids
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia

    def fit(self, X):
        X = np.asarray(X)
        self.inertia_history_ = []

        centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X, centroids) # E-step
            new_centroids = self._update_centroids(X, labels) # M-step

            # calculate centroid movement for convergence check
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            inertia = self._compute_inertia(X, labels, centroids)
            self.inertia_history_.append(inertia)

            if centroid_shift < self.tol:
                self.converged_ = True
                break

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration + 1

        return self

    def predict(self, X):
        X = np.asarray(X)
        return self._assign_clusters(X, self.centroids_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# %% [markdown]
# # GMM

# %%
class GMM:
    def __init__(self, n_components, covariance_type='full', tol=1e-4, max_iter=100, reg_covar=1e-6):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.log_likelihood_history_ = []
        self.converged_ = False

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        # 1. Initialize means (choose random points from data)
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices]
        
        # 2. Initialize weights (uniform)
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        
        # 3. Initialize covariances based on type
        if self.covariance_type == 'full':
            self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        elif self.covariance_type == 'tied':
            self.covariances_ = np.eye(n_features)
        elif self.covariance_type == 'diagonal':
            self.covariances_ = np.ones((self.n_components, n_features))
        elif self.covariance_type == 'spherical':
            self.covariances_ = np.ones(self.n_components)

    def _compute_log_gaussian_prob(self, X, mean, cov):
        n_samples, n_features = X.shape
        
        # Handle different covariance shapes
        if self.covariance_type == 'full':
            # shape of cov: (n_features, n_features)
            # Log determinant
            sign, log_det = np.linalg.slogdet(cov)
            if sign != 1: # Numerical stability check
                log_det = np.log(np.linalg.det(cov) + 1e-10) 
            
            # Mahalanobis distance
            prec = np.linalg.inv(cov)
            diff = X - mean
            mahalanobis = np.sum(np.dot(diff, prec) * diff, axis=1)
            
            return -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)

        elif self.covariance_type == 'tied':
            # shape of cov: (n_features, n_features) (Shared)
            sign, log_det = np.linalg.slogdet(cov)
            prec = np.linalg.inv(cov)
            diff = X - mean
            mahalanobis = np.sum(np.dot(diff, prec) * diff, axis=1)
            return -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)

        elif self.covariance_type == 'diagonal':
            # shape of cov: (n_features,)
            # cov here is the diagonal elements (variance vector)
            log_det = np.sum(np.log(cov))
            prec = 1.0 / cov # Inverse of diagonal is just 1/element
            diff = X - mean
            mahalanobis = np.sum((diff ** 2) * prec, axis=1)
            return -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)

        elif self.covariance_type == 'spherical':
            # shape of cov: scalar
            log_det = n_features * np.log(cov)
            prec = 1.0 / cov
            diff = X - mean
            mahalanobis = np.sum((diff ** 2) * prec, axis=1)
            return -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)

    def _e_step(self, X):
        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            if self.covariance_type == 'tied':
                cov = self.covariances_
            else:
                cov = self.covariances_[k]
            
            log_gauss = self._compute_log_gaussian_prob(X, self.means_[k], cov)
            log_resp[:, k] = np.log(self.weights_[k]) + log_gauss
        
        # Log-Sum-Exp for numerical stability 
        # Calculate log(sum(exp(log_resp))) safely
        log_prob_norm = np.logaddexp.reduce(log_resp, axis=1)
        
        # Normalize responsibilities (in log domain first, then exp)
        # resp[i, k] = exp(log_resp[i, k] - log_prob_norm[i])
        log_resp_norm = log_resp - log_prob_norm[:, np.newaxis]
        resp = np.exp(log_resp_norm)
        
        return resp, np.mean(log_prob_norm) # Return avg log likelihood

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        
        # 1. Update Weights (N_k / N)
        # Sum of responsibilities for each cluster k
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps # Add eps to avoid div by zero
        self.weights_ = nk / n_samples
        
        # 2. Update Means
        # weighted sum of data / total weight
        self.means_ = np.dot(resp.T, X) / nk[:, np.newaxis]
        
        # 3. Update Covariances
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                # Weighted covariance: (diff.T * resp_k) @ diff / nk
                cov_k = np.dot(resp[:, k] * diff.T, diff) / nk[k]
                # Regularization for stability 
                cov_k.flat[::n_features + 1] += self.reg_covar
                self.covariances_[k] = cov_k

        elif self.covariance_type == 'tied':
            # Tied uses the same covariance for all clusters, averaged over all data
            avg_cov = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                avg_cov += np.dot(resp[:, k] * diff.T, diff)
            avg_cov /= n_samples # Divided by total N, not nk
            avg_cov.flat[::n_features + 1] += self.reg_covar
            self.covariances_ = avg_cov

        elif self.covariance_type == 'diagonal':
            # Only store the diagonal variance vector
            avg_cov = np.zeros((self.n_components, n_features))
            for k in range(self.n_components):
                # Variance = sum(resp * (x-u)^2) / nk
                diff_sq = (X - self.means_[k]) ** 2
                avg_cov[k] = np.sum(resp[:, k][:, np.newaxis] * diff_sq, axis=0) / nk[k]
                avg_cov[k] += self.reg_covar
            self.covariances_ = avg_cov

        elif self.covariance_type == 'spherical':
            # Scalar variance: average variance across all dimensions
            self.covariances_ = np.zeros(self.n_components)
            for k in range(self.n_components):
                diff_sq = (X - self.means_[k]) ** 2
                # Mean of variance across features
                var_k = np.sum(resp[:, k][:, np.newaxis] * diff_sq) / nk[k]
                self.covariances_[k] = var_k / n_features # Average over features
                self.covariances_[k] += self.reg_covar

    def fit(self, X):
        self._initialize_parameters(X)
        self.log_likelihood_history_ = []
        
        for i in range(self.max_iter):
            prev_log_likelihood = self.log_likelihood_history_[-1] if self.log_likelihood_history_ else -np.inf
            
            # E-Step
            resp, current_log_likelihood = self._e_step(X)
            self.log_likelihood_history_.append(current_log_likelihood)
            
            # M-Step
            self._m_step(X, resp)
            
            # Convergence Check
            change = abs(current_log_likelihood - prev_log_likelihood)
            if i > 0 and change < self.tol:
                self.converged_ = True
                print(f"Converged at iteration {i} with log-likelihood {current_log_likelihood:.4f}")
                break
                
        return self

    def predict(self, X):
        resp, _ = self._e_step(X)
        return np.argmax(resp, axis=1)

    def predict_proba(self, X):
        resp, _ = self._e_step(X)
        return resp
    
    def get_params_count(self, n_features):
        # 1. Means (K * D)
        n_mean_params = self.n_components * n_features
        
        # 2. Weights (K - 1)
        n_weight_params = self.n_components - 1
        
        # 3. Covariances (Dependent on type)
        if self.covariance_type == 'full':
            # K * (D * (D + 1) / 2)
            n_cov_params = self.n_components * n_features * (n_features + 1) // 2
        elif self.covariance_type == 'tied':
            # 1 * (D * (D + 1) / 2)
            n_cov_params = n_features * (n_features + 1) // 2
        elif self.covariance_type == 'diagonal':
            # K * D
            n_cov_params = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            # K
            n_cov_params = self.n_components
            
        return n_mean_params + n_weight_params + n_cov_params

    def bic(self, X):
        # Bayesian Information Criterion
        n_samples, n_features = X.shape
        
        # Get total log likelihood (sum, not mean)
        resp, mean_log_likelihood = self._e_step(X)
        total_log_likelihood = mean_log_likelihood * n_samples
        
        k = self.get_params_count(n_features)
        
        # Formula: k * ln(N) - 2 * log_like
        return k * np.log(n_samples) - 2 * total_log_likelihood

    def aic(self, X):
        # Akaike Information Criterion
        n_samples, n_features = X.shape
        
        # Get total log likelihood
        resp, mean_log_likelihood = self._e_step(X)
        total_log_likelihood = mean_log_likelihood * n_samples
        
        k = self.get_params_count(n_features)
        
        # Formula: 2 * k - 2 * log_like
        return 2 * k - 2 * total_log_likelihood

# %% [markdown]
# # Experiments

# %% [markdown]
# ## Setup

# %%
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    return (X - mean) / std

data = load_breast_cancer()
X = data.data   # features
y = data.target # labels
feature_names = data.feature_names
target_names = data.target_names

X = standardize(X)


# %% [markdown]
# ### Helper Functions

# %%
def plot_elbow(results, init, k_values):
    # extract the inertia of the best run for each k
    inertias = [
        results[init][k]["best_run"]["inertia"]
        for k in k_values
    ]

    plt.figure()
    plt.plot(list(k_values), inertias, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS (Inertia)")
    plt.title(f"Elbow Curve ({init} initialization)")
    plt.show()


# %%
def plot_silhouette(results, init, k_values):
    # extract the silhouette score of the best run for each k
    # higher silhouette score is better
    silhouettes = [
        results[init][k]["best_run"]["silhouette"]
        for k in k_values
    ]

    plt.figure()
    plt.plot(list(k_values), silhouettes, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Analysis ({init} initialization)")
    plt.show()


# %%
def plot_davies_bouldin(results, init, k_values):
    # extract the davies-bouldin index of the best run for each k
    # lower dbi is better
    dbi_scores = [
        results[init][k]["best_run"]["dbi"]
        for k in k_values
    ]

    plt.figure()
    plt.plot(list(k_values), dbi_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Davies-Bouldin Index")
    plt.title(f"Davies-Bouldin Index vs k ({init} initialization)")
    plt.show()


# %%
def plot_calinski_harabasz(results, init, k_values):
    # extract the calinski-harabasz index of the best run for each k
    # higher chi is better
    chi_scores = [
        results[init][k]["best_run"]["chi"]
        for k in k_values
    ]

    plt.figure()
    plt.plot(list(k_values), chi_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Calinski-Harabasz Index")
    plt.title(f"Calinski-Harabasz Index vs k ({init} initialization)")
    plt.show()


# %%
def plot_convergence_speed(results, init, k_values):
    # extract the number of iterations to converge for the best run for each k
    iterations = [
        results[init][k]["best_run"]["n_iter"]
        for k in k_values
    ]

    plt.figure()
    plt.plot(list(k_values), iterations, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Iterations to converge")
    plt.title(f"Convergence Speed ({init} initialization)")
    plt.show()


# %%
def plot_contingency_table(labels_true, labels_pred):
    # compute contingency table where the rows are clusters and columns are true labels
    table = contingency_table(labels_true, labels_pred)

    plt.figure()
    plt.imshow(table)
    plt.xlabel("True Label")
    plt.ylabel("Cluster ID")
    plt.title("Contingency Table (Cluster vs True Label)")
    plt.colorbar()

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            plt.text(j, i, table[i, j], ha="center", va="center")

    plt.show()


# %%
def plot_confusion_matrix(labels_true, labels_pred_mapped):
    classes = np.unique(labels_true)
    matrix = np.zeros((classes.size, classes.size), dtype=int)

    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            matrix[i, j] = np.sum(
                (labels_true == true_cls) & (labels_pred_mapped == pred_cls)
            )

    plt.figure()
    plt.imshow(matrix)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Majority-Vote Mapping)")
    plt.colorbar()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.show()


# %% [markdown]
# ## 1) K-Means on original data

# %%
k_values = range(2, 11)
k_inits = ["random", "kmeans++"]
k_n_runs = 10 # number of independent runs per (k, init) configuration
k_random_seed_base = 42

k_results = {}

for init in k_inits:
    k_results[init] = {}
    for k in k_values:
        k_results[init][k] = {
            "runs": [],
            "best_run": None
        }

for init in k_inits:
    for k in k_values:
        best_silhouette = -np.inf
        best_run = None

        for run in range(k_n_runs):
            k_m = KMeans(
                n_clusters=k,
                init=init,
                max_iter=300,
                tol=1e-4,
                random_state=k_random_seed_base + run
            )

            labels = k_m.fit_predict(X)

            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_index(X, labels, k_m.centroids_)
            chi = calinski_harabasz_index(X, labels, k_m.centroids_)
            run_result = {
                "labels": labels,
                "centroids": k_m.centroids_,
                "inertia": k_m.inertia_,
                "silhouette": sil,
                "dbi": dbi,
                "chi": chi,
                "n_iter": k_m.n_iter_
            }

            k_results[init][k]["runs"].append(run_result)

            if sil > best_silhouette:
                best_silhouette = sil
                best_run = run_result

        k_results[init][k]["best_run"] = best_run

for init in k_inits:
    plot_elbow(k_results, init, k_values)
    plot_silhouette(k_results, init, k_values)
    plot_davies_bouldin(k_results, init, k_values)
    plot_calinski_harabasz(k_results, init, k_values)
    plot_convergence_speed(k_results, init, k_values)

k_best_config = None
k_best_score = -np.inf

for init in k_inits:
    for k in k_values:
        sil = k_results[init][k]["best_run"]["silhouette"]
        if sil > k_best_score:
            k_best_score = sil
            k_best_config = (init, k)

k_best_init, k_best_k = k_best_config
k_best_run = k_results[k_best_init][k_best_k]["best_run"]

print(adjusted_rand_index(y, k_best_run["labels"]))
print(normalized_mutual_information(y, k_best_run["labels"]))
print(purity_score(y, k_best_run["labels"]))

plot_contingency_table(y, k_best_run["labels"])

mapping, mapped_labels = majority_vote_mapping(
    y,
    k_best_run["labels"]
)

plot_confusion_matrix(y, mapped_labels)


# %% [markdown]
# ## 2) GMM on original data

# %%
def experiment_2_gmm_original(X, k_values, GMMClass):
    results = []

    for k in k_values:
        for cov in ["full", "tied", "diag", "spherical"]:
            gmm = GMMClass(
                n_components=k,
                covariance_type=cov,
                max_iter=200,
                tol=1e-4
            )
            gmm.fit(X)

            results.append({
                "components": k,
                "covariance": cov,
                "log_likelihood": gmm.log_likelihoods_[-1],
                "bic": gmm.bic(X),
                "aic": gmm.aic(X)
            })

    return results



# %% [markdown]
# ## 3) K-Means after PCA

# %%
def experiment_3_kmeans_pca(X, k, component_list, PCAClass):
    results = []

    for n_comp in component_list:
        pca = PCAClass(n_components=n_comp)
        pca.fit(X)
        X_pca = pca.transform(X)

        km = KMeans(n_clusters=k, init="kmeans++", random_state=42)
        km.fit(X_pca)

        results.append({
            "components": n_comp,
            "wcss": km.inertia_,
            "silhouette": silhouette_score(X_pca, km.labels_),
            "reconstruction_error": pca.reconstruction_error(X)
        })

    return results



# %% [markdown]
# ## 4) GMM after PCA

# %%
def experiment_4_gmm_pca(X, component_list, PCAClass, GMMClass):
    results = []

    for n_comp in component_list:
        pca = PCAClass(n_components=n_comp)
        pca.fit(X)
        X_pca = pca.transform(X)

        for cov in ["full", "tied", "diag", "spherical"]:
            gmm = GMMClass(
                n_components=2,
                covariance_type=cov
            )
            gmm.fit(X_pca)

            results.append({
                "components": n_comp,
                "covariance": cov,
                "log_likelihood": gmm.log_likelihoods_[-1]
            })

    return results



# %% [markdown]
# ## 5) K-Means after Autoencoder

# %%
def experiment_5_kmeans_autoencoder(
    X, k, bottlenecks, AutoencoderClass, ae_params
):
    results = []

    for b in bottlenecks:
        ae = AutoencoderClass(
            layer_sizes=[X.shape[1], 64, 32, b, 32, 64, X.shape[1]],
            **ae_params
        )
        ae.train(X)

        X_latent = ae.encode(X)

        km = KMeans(n_clusters=k, init="kmeans++", random_state=42)
        km.fit(X_latent)

        results.append({
            "bottleneck": b,
            "wcss": km.inertia_,
            "silhouette": silhouette_score(X_latent, km.labels_),
            "reconstruction_loss": ae.compute_loss(X, ae.reconstruct(X))
        })

    return results



# %% [markdown]
# ## 6) GMM after Autoencoder

# %%
def experiment_6_gmm_autoencoder(
    X, bottlenecks, AutoencoderClass, GMMClass, ae_params
):
    results = []

    for b in bottlenecks:
        ae = AutoencoderClass(
            layer_sizes=[X.shape[1], 64, 32, b, 32, 64, X.shape[1]],
            **ae_params
        )
        ae.train(X)

        X_latent = ae.encode(X)

        for cov in ["full", "tied", "diag", "spherical"]:
            gmm = GMMClass(
                n_components=2,
                covariance_type=cov
            )
            gmm.fit(X_latent)

            results.append({
                "bottleneck": b,
                "covariance": cov,
                "log_likelihood": gmm.log_likelihoods_[-1]
            })

    return results


