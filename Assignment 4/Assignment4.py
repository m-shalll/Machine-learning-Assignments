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
#     display_name: base
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
        return np.maximum(0, x)  # ReLU: sets negative values to 0, keeps positives

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)  # derivative: 1 if x>0 else 0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))  # Sigmoid: squashes values to [0,1]

    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)  # compute sigmoid
        return s * (1 - s)         # derivative formula: s*(1-s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)  # Tanh: squashes values to [-1,1]

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2  # derivative: 1 - tanh^2(x)


class Autoencoder:
    def __init__(
        self,
        layer_sizes,
        activation="relu",
        learning_rate=0.01,
        l2_lambda=0.0,
        lr_decay=0.0
    ):
        self.layer_sizes = layer_sizes  # list of layer sizes
        self.learning_rate = learning_rate  # initial learning rate
        self.initial_lr = learning_rate    # store initial LR for decay
        self.l2_lambda = l2_lambda        # L2 regularization strength
        self.lr_decay = lr_decay          # learning rate decay factor

        self.weights = []  # list to store weight matrices
        self.biases = []   # list to store bias vectors

        self.activations = []  # store activations during forward pass
        self.z_values = []     # store linear outputs (z=W*X+b)

        # choose activation function
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

        self._initialize_parameters()  # initialize weights and biases

    def _initialize_parameters(self):
        for i in range(len(self.layer_sizes) - 1):
            # He initialization: scale by sqrt(2/fan_in)
            weight = np.random.randn(
                self.layer_sizes[i],
                self.layer_sizes[i + 1]
            ) * np.sqrt(2 / self.layer_sizes[i])
            bias = np.zeros((1, self.layer_sizes[i + 1]))  # biases start at 0
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        self.activations = [X]  # store input as first activation
        self.z_values = []       # reset z values

        for W, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], W) + b  # linear combination
            self.z_values.append(z)
            a = self.act(z)                          # apply activation
            self.activations.append(a)

        return self.activations[-1]  # return output layer

    def compute_loss(self, X, X_hat):
        mse = np.mean((X - X_hat) ** 2)  # mean squared error
        l2_penalty = self.l2_lambda * sum(np.sum(W ** 2) for W in self.weights)  # L2
        return mse + l2_penalty  # total loss

    def backward(self, X):
        grads_W = []
        grads_b = []

        # derivative of MSE loss w.r.t output layer
        delta = (self.activations[-1] - X) * self.act_deriv(self.z_values[-1])

        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.activations[i].T, delta)  # weight gradient
            dB = np.sum(delta, axis=0, keepdims=True)  # bias gradient

            dW += self.l2_lambda * self.weights[i]    # add L2 gradient

            grads_W.insert(0, dW)  # store gradients in correct order
            grads_b.insert(0, dB)

            if i != 0:
                # propagate delta to previous layer
                delta = np.dot(delta, self.weights[i].T) * self.act_deriv(self.z_values[i - 1])

        return grads_W, grads_b

    def update_parameters(self, grads_W, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_W[i]  # update weights
            self.biases[i] -= self.learning_rate * grads_b[i]   # update biases

    def train(self, X, epochs=100, batch_size=32):
        n_samples = X.shape[0]
        history  = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)  # shuffle data
            X_shuffled = X[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch = X_shuffled[start:end]  # get batch

                X_hat = self.forward(batch)          # forward pass
                grads_W, grads_b = self.backward(batch)  # backward pass
                self.update_parameters(grads_W, grads_b)  # gradient descent step

            # decay learning rate
            self.learning_rate = self.initial_lr / (1 + self.lr_decay * epoch)
            current_loss = self.compute_loss(X, self.forward(X))
            history.append(current_loss) # Add to list

            if epoch % 10 == 0:
                loss = self.compute_loss(X, self.forward(X))  # compute full loss
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        return history

    def encode(self, X):
        # pass only through encoder layers (first half)
        for i in range(len(self.weights) // 2):
            X = self.act(np.dot(X, self.weights[i]) + self.biases[i])
        return X

    def reconstruct(self, X):
        # pass through full autoencoder
        return self.forward(X)


# %% [markdown]
# # Internal Metrics

# %% [markdown]
# ### Silhouette Score

# %%
def silhouette_score(X, labels):
    X = np.asarray(X)
    labels = np.asarray(labels)
    n_samples = X.shape[0]  # total number of samples
    unique_labels = np.unique(labels)  # list of unique cluster labels

    # Compute pairwise distances between all points
    distances = np.linalg.norm(
        # Subtract → get difference vectors between all points
        X[:, np.newaxis, :] - X[np.newaxis, :, :],
        # axis=2 → compute norm along feature axis for each pair
        axis=2
    )

    silhouette_values = np.zeros(n_samples)  # store silhouette for each point

    # Loop through each point
    for i in range(n_samples):
        # Mask for points in the same cluster
        same_cluster = labels == labels[i]
        # Mask for points in other clusters
        other_clusters = unique_labels[unique_labels != labels[i]]

        # Compute average distance to points in the same cluster (a_i)
        if np.sum(same_cluster) > 1:
            a_i = np.mean(distances[i, same_cluster & (np.arange(n_samples) != i)])
        else:
            a_i = 0.0

        # Compute minimum average distance to points in other clusters (b_i)
        b_i = np.inf
        for cluster in other_clusters:
            cluster_mask = labels == cluster
            b_i = min(b_i, np.mean(distances[i, cluster_mask]))

        # Silhouette for this point
        silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)

    # Return average silhouette score over all points
    return np.mean(silhouette_values)


# %% [markdown]
# ### Davies-Bouldin Index

# %%
def davies_bouldin_index(X, labels, centroids):
    X = np.asarray(X)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids)

    k = centroids.shape[0] # number of clusters

    # Compute S_i: average distance of points in cluster i to its centroid
    S = np.zeros(k)
    for i in range(k):
        cluster_points = X[labels == i]
        S[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))

    # Compute distance between all centroids
    centroid_distances = np.linalg.norm(
        centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )

    dbi = 0.0
    for i in range(k):
        # For cluster i, compute R_ij for all other clusters j
        R_ij = []
        for j in range(k):
            if i != j:
                R_ij.append((S[i] + S[j]) / centroid_distances[i, j])
        # Take max R_ij for cluster i
        dbi += max(R_ij)

     # Average over all clusters
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

    # Overall mean of all points
    overall_mean = np.mean(X, axis=0)

    # Compute W: sum of squared distances of points to their cluster centroid
    W = 0.0
    for i in range(k):
        cluster_points = X[labels == i]
        W += np.sum((cluster_points - centroids[i]) ** 2)

    # Compute B: sum over clusters of n_i * squared distance of centroid to overall mean
    B = 0.0
    for i in range(k):
        n_i = np.sum(labels == i)
        B += n_i * np.sum((centroids[i] - overall_mean) ** 2)

    # Calinski-Harabasz index
    return (B / (k - 1)) / (W / (n_samples - k))


# %% [markdown]
# ### Within-cluster sum of squares (WCSS)

# %%
def wcss(X, labels, centroids):
    X = np.asarray(X)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids)

    # Sum of squared distances of points to their cluster centroid
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

    clusters = np.unique(labels_pred) # get unique cluster IDs
    classes = np.unique(labels_true) # get unique true class labels

    table = np.zeros((clusters.size, classes.size), dtype=int)

    for i, cluster in enumerate(clusters):
        for j, cls in enumerate(classes):
            # count how many points in this cluster have this true label
            table[i, j] = np.sum(
                (labels_pred == cluster) & (labels_true == cls)
            )

    return table # rows=clusters, columns=true classes


# %% [markdown]
# ### Adjusted Rand Index

# %%
def nC2(n):
    return n * (n - 1) / 2  # number of ways to choose 2 items from n

def adjusted_rand_index(labels_true, labels_pred):
    table = contingency_table(labels_true, labels_pred)
    n = np.sum(table)  # total number of points

    sum_nij = np.sum(nC2(table))               # sum of combinations within each cell
    sum_ai = np.sum(nC2(np.sum(table, axis=1)))  # sum over row totals
    sum_bj = np.sum(nC2(np.sum(table, axis=0)))  # sum over column totals
    total_pairs = nC2(n)                         # total number of pairs

    expected_index = (sum_ai * sum_bj) / total_pairs  # expected number of agreements by chance
    max_index = 0.5 * (sum_ai + sum_bj)               # maximum possible agreements

    return (sum_nij - expected_index) / (max_index - expected_index)  # scale [-1,1]



# %% [markdown]
# ### Normalized Mutual Information

# %%
def normalized_mutual_information(labels_true, labels_pred, eps=1e-10):
    table = contingency_table(labels_true, labels_pred)
    n = np.sum(table)  # total points

    P_ij = table / n      # joint probability of cluster i and class j
    P_i = np.sum(P_ij, axis=1)  # probability of cluster i
    P_j = np.sum(P_ij, axis=0)  # probability of class j

    MI = 0.0
    for i in range(P_ij.shape[0]):
        for j in range(P_ij.shape[1]):
            if P_ij[i, j] > 0:
                # mutual information contribution for this cell
                MI += P_ij[i, j] * np.log(P_ij[i, j] / (P_i[i] * P_j[j] + eps))

    H_i = -np.sum(P_i * np.log(P_i + eps))  # entropy of clusters
    H_j = -np.sum(P_j * np.log(P_j + eps))  # entropy of true labels

    return MI / np.sqrt(H_i * H_j)  # normalized to [0,1]


# %% [markdown]
# ### Purity

# %%
def purity_score(labels_true, labels_pred):
    table = contingency_table(labels_true, labels_pred)
    # sum the largest counts in each cluster and divide by total points
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
        majority_label = values[np.argmax(counts)] # pick most frequent true label

        mapping[cluster] = majority_label
        mapped_predictions[labels_pred == cluster] = majority_label # assign majority to cluster

    return mapping, mapped_predictions


# %% [markdown]
# # K-Means

# %% [markdown]
# ### K-Means Class Implementation

# %%
class KMeans:
    def __init__(self, n_clusters=2, init="kmeans++", max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_history_ = []
        self.n_iter_ = 0
        self.converged_ = False

        if random_state is not None:
            np.random.seed(random_state)


    def _init_random(self, X):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]

    def _init_kmeans_plus_plus(self, X):
        n_samples, n_features = X.shape
        centroids = np.empty((self.n_clusters, n_features))

        idx = np.random.randint(n_samples)
        centroids[0] = X[idx]

        for i in range(1, self.n_clusters):
            distances = np.min(
                np.sum((X[:, np.newaxis, :] - centroids[:i]) ** 2, axis=2),
                axis=1
            )

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


    def _assign_clusters(self, X, centroids):
        distances = np.sum(
            (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                centroids[k] = X[np.random.randint(X.shape[0])]
            else:
                centroids[k] = np.mean(cluster_points, axis=0)

        return centroids

    def _compute_inertia(self, X, labels, centroids):
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia

    def fit(self, X):
        X = np.asarray(X)
        self.inertia_history_ = []

        centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)

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
            avg_cov /= n_samples 
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
            # (D * (D + 1) / 2)
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
        total_log_likelihood = mean_log_likelihood  * n_samples
        
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

X_scaled = standardize(X)
X = X_scaled

def manual_train_test_split(X, y, test_ratio=0.2, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Get total number of samples
    n_samples = X.shape[0]
    
    # Create a shuffled list of indices (0 to 568)
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * test_ratio)
    
    # Slice the indices
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # Select the actual data points using these indices
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# 5. Execute the split
X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio=0.2)


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
def manual_mse(X_original, X_reconstructed):
    # 1. Compute the difference
    diff = X_original - X_reconstructed
    
    # 2. Square the differences
    squared_diff = diff ** 2
    
    # 3. Compute the mean of all elements
    mse_score = np.mean(squared_diff)
    
    return mse_score


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

print(f"ARI: {adjusted_rand_index(y, k_best_run['labels'])}")
print(f"NMI: {normalized_mutual_information(y, k_best_run['labels'])}")
print(f"Purity: {purity_score(y, k_best_run['labels'])}")

plot_contingency_table(y, k_best_run["labels"])

mapping, mapped_labels = majority_vote_mapping(
    y,
    k_best_run["labels"]
)

plot_confusion_matrix(y, mapped_labels)

# %% [markdown]
# ## 2) GMM on original data

# %%
import seaborn as sns
from sklearn.metrics import confusion_matrix
def run_experiment_2(X, y_true):
    print("Starting Experiment 2: GMM")
    
    covariance_types = ['full', 'tied', 'diagonal', 'spherical']
    k_range = range(2, 11) # Clusters 2 to 10
    
    results = []

    best_bic = np.inf
    best_model_labels = None
    best_model_name = ""
    
    # Store BIC/AIC for plotting
    bic_scores = {cov: [] for cov in covariance_types}
    aic_scores = {cov: [] for cov in covariance_types}
    
    for cov_type in covariance_types:
        print(f"\n--- Testing Covariance: {cov_type} ---")
        
        for k in k_range:
            start_time = time.time()
            # 1. Train Model
            gmm = GMM(n_components=k, covariance_type=cov_type, max_iter=100, tol=1e-4)
            gmm.fit(X)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            # 2. Get Assignments
            # E-step gives us responsibilities, we take argmax for hard labels
            resp, _ = gmm._e_step(X)
            labels_pred = np.argmax(resp, axis=1)
            
            # 3. Calculate Internal Metrics
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            sil = silhouette_score(X, labels_pred)
            dbi = davies_bouldin_index(X, labels_pred, gmm.means_)
            chi = calinski_harabasz_index(X, labels_pred, gmm.means_)
            
            # 4. Calculate External Metrics
            ari = adjusted_rand_index(y_true, labels_pred)
            nmi = normalized_mutual_information(y_true, labels_pred)
            purity = purity_score(y_true, labels_pred)
            
            # Store for plotting
            bic_scores[cov_type].append(bic)
            aic_scores[cov_type].append(aic)
            
            # Record detailed stats
            results.append({
                'Covariance': cov_type,
                'K': k,
                'BIC': bic,
                'AIC': aic,
                'Silhouette': sil,
                'Davies-Bouldin': dbi,
                'Calinski-Harabasz': chi,
                'Purity': purity,
                'ARI': ari,       
                'NMI': nmi,       
                'Converged': gmm.converged_,
                'Time_Seconds': elapsed_time
            })

            # Save best model for Confusion Matrix
            if bic < best_bic:
                best_bic = bic
                best_model_labels = labels_pred
                best_model_name = f"{cov_type} (K={k})"
            
            print(f"K={k}: BIC={bic:.0f}, AIC={aic:.0f}, Sil={sil:.3f}")

    # --- Create Comparison Tables ---
    df_results = pd.DataFrame(results)
    print("\nExperiment 2 Summary Table:")
    print(df_results.sort_values(by='BIC').head(10)) # Show top 10 best models
    
    # --- Visualizations ---
    plot_bic_aic(k_range, bic_scores, aic_scores)

    plot_heatmap(df_results)
    
    plot_confusion_matrix_manual(y_true, best_model_labels, best_model_name)
    
    return df_results

def plot_bic_aic(k_range, bic_scores, aic_scores):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot BIC
    for cov_type, scores in bic_scores.items():
        ax1.plot(k_range, scores, marker='o', label=f'{cov_type}')
    ax1.set_title('BIC Score vs. Number of Components')
    ax1.set_xlabel('Number of Components (k)')
    ax1.set_ylabel('BIC Score (Lower is Better)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot AIC
    for cov_type, scores in aic_scores.items():
        ax2.plot(k_range, scores, marker='o', linestyle='--', label=f'{cov_type}')
    ax2.set_title('AIC Score vs. Number of Components')
    ax2.set_xlabel('Number of Components (k)')
    ax2.set_ylabel('AIC Score (Lower is Better)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_heatmap(df_results):
    # 1. Group by Covariance Type and take the mean of metrics
    metrics = ['BIC', 'AIC', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Purity', 'Time_Seconds', 'ARI', 'NMI']
    grouped = df_results.groupby('Covariance')[metrics].mean()
    
    # 2. Normalize data for heatmap (min-max scaling) so colors make sense
    # (Since BIC is ~20,000 and Purity is ~0.9, we must scale them)
    normalized = (grouped - grouped.min()) / (grouped.max() - grouped.min())
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(normalized, annot=grouped.round(2), cmap='viridis', fmt='g')
    plt.title('Average Performance Heatmap (Normalized Colors, Raw Values)')
    plt.show()

def plot_confusion_matrix_manual(y_true, y_pred, title):
    """
    Plots the confusion matrix for the best model using majority vote mapping.
    """
    # 1. Remap the cluster labels to true class labels using your function
    _, y_pred_mapped = majority_vote_mapping(y_true, y_pred)
    
    # 2. Calculate confusion matrix using the MAPPED predictions
    cm = confusion_matrix(y_true, y_pred_mapped)
    
    # 3. Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.xlabel('Predicted Class (Mapped via Majority Vote)')
    plt.ylabel('True Class')
    plt.title(f'Confusion Matrix: {title}')
    plt.tight_layout()
    plt.show()

results = run_experiment_2(X_train, y_train)



# %% [markdown]
# ## 3) K-Means after PCA

# %%
pca_components = [2, 5, 10, 15, 20]
k_values = range(2, 11)
k_inits = ["random", "kmeans++"]
pca_k_n_runs = 10
pca_k_seed_base = 42

pca_k_results = {}

for n_comp in pca_components:
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    X_reduced = pca.transform(X)

    pca_k_results[n_comp] = {}

    for init in k_inits:
        pca_k_results[n_comp][init] = {}

        for k in k_values:
            pca_k_results[n_comp][init][k] = {
                "runs": [],
                "best_run": None
            }

            best_silhouette = -np.inf
            best_run = None

            for run in range(pca_k_n_runs):
                km = KMeans(
                    n_clusters=k,
                    init=init,
                    max_iter=300,
                    tol=1e-4,
                    random_state=pca_k_seed_base + run
                )

                labels = km.fit_predict(X_reduced)

                sil = silhouette_score(X_reduced, labels)
                dbi = davies_bouldin_index(X_reduced, labels, km.centroids_)
                chi = calinski_harabasz_index(X_reduced, labels, km.centroids_)

                run_result = {
                    "labels": labels,
                    "centroids": km.centroids_,
                    "inertia": km.inertia_,
                    "silhouette": sil,
                    "dbi": dbi,
                    "chi": chi,
                    "n_iter": km.n_iter_
                }

                pca_k_results[n_comp][init][k]["runs"].append(run_result)

                if sil > best_silhouette:
                    best_silhouette = sil
                    best_run = run_result

            pca_k_results[n_comp][init][k]["best_run"] = best_run

for n_comp in pca_components:
    for init in k_inits:
        plot_elbow(pca_k_results[n_comp], init, k_values)
        plot_silhouette(pca_k_results[n_comp], init, k_values)
        plot_convergence_speed(pca_k_results[n_comp], init, k_values)
        plot_davies_bouldin(pca_k_results[n_comp], init, k_values)
        plot_calinski_harabasz(pca_k_results[n_comp], init, k_values)

pca_best_per_dim = {}

for n_comp in pca_components:
    best_score = -np.inf
    best_config = None

    for init in k_inits:
        for k in k_values:
            sil = pca_k_results[n_comp][init][k]["best_run"]["silhouette"]
            if sil > best_score:
                best_score = sil
                best_config = (init, k)

    pca_best_per_dim[n_comp] = best_config

pca_best_score = -np.inf
pca_best_config = None

for n_comp in pca_components:
    init, k = pca_best_per_dim[n_comp]
    sil = pca_k_results[n_comp][init][k]["best_run"]["silhouette"]

    if sil > pca_best_score:
        pca_best_score = sil
        pca_best_config = (n_comp, init, k)

best_n, best_init, best_k = pca_best_config
final_pca_run = pca_k_results[best_n][best_init][best_k]["best_run"]

print(f"Best PCA components: {best_n}")
print(f"Best init: {best_init}, Best k: {best_k}")

print(f"ARI: {adjusted_rand_index(y, final_pca_run['labels'])}")
print(f"NMI: {normalized_mutual_information(y, final_pca_run['labels'])}")
print(f"Purity: {purity_score(y, final_pca_run['labels'])}")

plot_contingency_table(y, final_pca_run["labels"])

mapping, mapped_labels = majority_vote_mapping(
    y,
    final_pca_run["labels"]
)

plot_confusion_matrix(y, mapped_labels)


# %% [markdown]
# ## 4) GMM after PCA

# %%
def run_experiment_4(X, y):
    # 1. Setup
    pca_variations = [2, 5, 10, 15, 20]
    covariance_types = ['full', 'tied', 'diagonal', 'spherical']
    n_classes = len(np.unique(y))
    
    results = []
    best_model_info = {'score': -1, 'model': None, 'pca': None, 'X_reduced': None, 'y_pred': None}
    
    print(f"Starting Experiment 4: GMM (k={n_classes}) after PCA...")
    print("-" * 80)
    print(f"{'Dim':<5} | {'Covariance':<10} | {'Sil':<6} | {'ARI':<6} | {'Purity':<6} | {'BIC':<10} | {'Time (s)':<8}")
    print("-" * 80)
    
    for n_dim in pca_variations:
        # Check if n_dim is valid for dataset
        if n_dim > X.shape[1]:
            continue
            
        # --- A. Dimensionality Reduction (PCA) ---
        pca = PCA(n_components=n_dim, random_state=42)
        start_pca = time.time()
        X_pca = pca.fit_transform(X)
        pca_time = time.time() - start_pca
        
        # PCA Reconstruction Error (MSE)
        X_reconstructed = pca.inverse_transform(X_pca)
        reconstruction_error = manual_mse(X, X_reconstructed)
        explained_variance = np.sum(pca.explained_variance_ratio_)

        # --- B. GMM Grid Search ---
        for cov_type in covariance_types:
            start_gmm = time.time()
            
            # Fit GMM
            gmm = GMM(n_components=n_classes, covariance_type=cov_type, random_state=42, max_iter=200)
            gmm.fit(X_pca)
            y_pred = gmm.predict(X_pca)
            
            gmm_time = time.time() - start_gmm
            total_time = pca_time + gmm_time

            # --- C. Compute Metrics ---
            
            # 1. Internal Validation
            sil = silhouette_score(X_pca, y_pred)
            db = davies_bouldin_index(X_pca, y_pred, gmm.means_)
            ch = calinski_harabasz_index(X_pca, y_pred, gmm.means_)
            wcss_ = wcss(X_pca, gmm, gmm.means_)
            bic = gmm.bic(X_pca)
            aic = gmm.aic(X_pca)
            log_likelihood = gmm.score(X_pca) * len(X) # score returns avg log-likelihood

            # 2. External Validation
            ari = adjusted_rand_index(y, y_pred)
            nmi = normalized_mutual_information(y, y_pred)
            purity = purity_score(y, y_pred)

            # Store Results
            entry = {
                'Experiment': 'Exp 4 (PCA+GMM)',
                'Dimensions': n_dim,
                'Covariance_Type': cov_type,
                'Silhouette': sil,
                'Davies_Bouldin': db,
                'Calinski_Harabasz': ch,
                'WCSS': wcss_,
                'BIC': bic,
                'AIC': aic,
                'Log_Likelihood': log_likelihood,
                'ARI': ari,
                'NMI': nmi,
                'Purity': purity,
                'MSE_Reconstruction': reconstruction_error,
                'Explained_Variance': explained_variance,
                'Time_Sec': total_time
            }
            results.append(entry)
            
            # Print progress
            print(f"{n_dim:<5} | {cov_type:<10} | {sil:.4f} | {ari:.4f} | {purity:.4f} | {bic:.1f}      | {total_time:.4f}")

            # Keep track of best model (based on Silhouette for internal or ARI for external)
            # Here choosing ARI as tie-breaker for "best" visualization
            if ari > best_model_info['score']:
                best_model_info = {
                    'score': ari,
                    'model': gmm,
                    'pca': pca,
                    'X_reduced': X_pca,
                    'y_pred': y_pred,
                    'dims': n_dim,
                    'cov': cov_type
                }

    # 3. Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # 4. Generate Visualizations
    generate_visualizations(df_results, best_model_info, X, y, n_classes)
    
    return df_results

# --- Visualization Logic ---
def generate_visualizations(df, best_info, X, y_true, n_classes):
    plt.style.use('seaborn-v0_8')
    
    # 1. 2D Projection of Best Result
    # If best result has > 2 dims, we view the first 2 PCs
    X_vis = best_info['X_reduced'][:, :2]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # True Labels
    sns.scatterplot(x=X_vis[:,0], y=X_vis[:,1], hue=y_true, palette='viridis', ax=axes[0], s=50)
    axes[0].set_title(f"Ground Truth (PCA First 2 Comps)")
    
    # Cluster Predictions
    sns.scatterplot(x=X_vis[:,0], y=X_vis[:,1], hue=best_info['y_pred'], palette='tab10', ax=axes[1], s=50)
    axes[1].set_title(f"GMM Clustering: PCA={best_info['dims']}, Cov={best_info['cov']}")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.lineplot(data=df, x='Dimensions', y='BIC', hue='Covariance_Type', marker='o', ax=axes[0])
    axes[0].set_title('BIC Score vs Dimensions (Lower is Better)')
    
    sns.lineplot(data=df, x='Dimensions', y='Silhouette', hue='Covariance_Type', marker='o', ax=axes[1])
    axes[1].set_title('Silhouette Score vs Dimensions (Higher is Better)')
    plt.show()

    # 3. Confusion Matrix for Best Method
    cm = confusion_matrix(y_true, best_info['y_pred'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: Best GMM (Dim={best_info["dims"]}, {best_info["cov"]})')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Class')
    plt.show()

    # 4. Dimensionality Impact on Optimal Covariance (Heatmap of ARI)
    pivot_ari = df.pivot(index='Covariance_Type', columns='Dimensions', values='ARI')
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot_ari, annot=True, cmap='RdYlGn', fmt='.3f')
    plt.title('Impact of Dimensionality on Performance (ARI Score)')
    plt.show()

    # 5. BIC/AIC Curve for GMM Component Selection (Elbow Method)
    # This is a separate mini-check: Using the BEST dimension found, vary K from 2 to 20
    print("\nGenerating BIC/AIC Elbow Curves for optimal dimensions...")
    best_dim = best_info['dims']
    pca_elbow = PCA(n_components=best_dim)
    X_elbow = pca_elbow.fit_transform(X)
    
    n_components_range = range(1, 15) # Adjust range as needed
    bics = []
    aics = []
    
    for n in n_components_range:
        # Use 'full' covariance for standard elbow check
        gmm = GMM(n_components=n, covariance_type='full')
        gmm.fit(X_elbow)
        bics.append(gmm.bic(X_elbow))
        aics.append(gmm.aic(X_elbow))
        
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, bics, label='BIC', marker='o')
    plt.plot(n_components_range, aics, label='AIC', marker='x')
    plt.axvline(x=n_classes, color='r', linestyle='--', label='True K')
    plt.legend()
    plt.title(f'BIC/AIC vs Number of Components (Fixed at Dim={best_dim})')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.show()

results = run_experiment_2(X_train, y_train)


# %% [markdown]
# ## 5) K-Means after Autoencoder

# %%
ae_bottlenecks = [2, 5, 10, 15, 20]
k_values = range(2, 11)
k_inits = ["random", "kmeans++"]

ae_k_n_runs = 10
ae_k_seed_base = 42

ae_epochs = 100
ae_batch_size = 32
ae_learning_rate = 0.01
ae_l2_lambda = 0.0
ae_lr_decay = 0.0
ae_activation = "sigmoid"

ae_k_results = {}

for bottleneck in ae_bottlenecks:
    print(f"\nTraining Autoencoder with bottleneck size = {bottleneck}")

    input_dim = X.shape[1]
    layer_sizes = [
        input_dim,
        bottleneck * 2,
        bottleneck,
        bottleneck * 2,
        input_dim
    ]

    autoencoder = Autoencoder(
        layer_sizes=layer_sizes,
        activation=ae_activation,
        learning_rate=ae_learning_rate,
        l2_lambda=ae_l2_lambda,
        lr_decay=ae_lr_decay
    )

    autoencoder.train(X, epochs=ae_epochs, batch_size=ae_batch_size)

    X_encoded = autoencoder.encode(X)

    ae_k_results[bottleneck] = {}

    for init in k_inits:
        ae_k_results[bottleneck][init] = {}

        for k in k_values:
            ae_k_results[bottleneck][init][k] = {
                "runs": [],
                "best_run": None
            }

            best_silhouette = -np.inf
            best_run = None

            for run in range(ae_k_n_runs):
                km = KMeans(
                    n_clusters=k,
                    init=init,
                    max_iter=300,
                    tol=1e-4,
                    random_state=ae_k_seed_base + run
                )

                labels = km.fit_predict(X_encoded)

                if len(np.unique(labels)) < 2:
                    continue

                sil = silhouette_score(X_encoded, labels)
                dbi = davies_bouldin_index(X_encoded, labels, km.centroids_)
                chi = calinski_harabasz_index(X_encoded, labels, km.centroids_)

                run_result = {
                    "labels": labels,
                    "centroids": km.centroids_,
                    "inertia": km.inertia_,
                    "silhouette": sil,
                    "dbi": dbi,
                    "chi": chi,
                    "n_iter": km.n_iter_
                }

                ae_k_results[bottleneck][init][k]["runs"].append(run_result)

                if sil > best_silhouette:
                    best_silhouette = sil
                    best_run = run_result

            if best_run is None:
                all_runs = ae_k_results[bottleneck][init][k]["runs"]
                if len(all_runs) > 0:
                    best_run = min(all_runs, key=lambda r: r["inertia"])
                else:
                    best_run = {
                        "labels": labels,
                        "centroids": km.centroids_,
                        "inertia": km.inertia_,
                        "silhouette": -1,
                        "dbi": np.inf,
                        "chi": 0,
                        "n_iter": km.n_iter_
                    }

            ae_k_results[bottleneck][init][k]["best_run"] = best_run

for bottleneck in ae_bottlenecks:
    for init in k_inits:
        plot_elbow(ae_k_results[bottleneck], init, k_values)
        plot_silhouette(ae_k_results[bottleneck], init, k_values)
        plot_convergence_speed(ae_k_results[bottleneck], init, k_values)
        plot_davies_bouldin(ae_k_results[bottleneck], init, k_values)
        plot_calinski_harabasz(ae_k_results[bottleneck], init, k_values)

ae_best_per_bottleneck = {}

for bottleneck in ae_bottlenecks:
    best_score = -np.inf
    best_config = None

    for init in k_inits:
        for k in k_values:
            sil = ae_k_results[bottleneck][init][k]["best_run"]["silhouette"]
            if sil > best_score:
                best_score = sil
                best_config = (init, k)

    ae_best_per_bottleneck[bottleneck] = best_config

ae_best_score = -np.inf
ae_best_config = None

for bottleneck in ae_bottlenecks:
    init, k = ae_best_per_bottleneck[bottleneck]
    sil = ae_k_results[bottleneck][init][k]["best_run"]["silhouette"]

    if sil > ae_best_score:
        ae_best_score = sil
        ae_best_config = (bottleneck, init, k)

best_bottleneck, best_init, best_k = ae_best_config
final_ae_run = ae_k_results[best_bottleneck][best_init][best_k]["best_run"]

print(f"\nBest Autoencoder bottleneck: {best_bottleneck}")
print(f"Best init: {best_init}, Best k: {best_k}")

print(f"ARI: {adjusted_rand_index(y, final_ae_run['labels'])}")
print(f"NMI: {normalized_mutual_information(y, final_ae_run['labels'])}")
print(f"Purity: {purity_score(y, final_ae_run['labels'])}")

plot_contingency_table(y, final_ae_run["labels"])

mapping, mapped_labels = majority_vote_mapping(
    y,
    final_ae_run["labels"]
)

plot_confusion_matrix(y, mapped_labels)


# %% [markdown]
# ## 6) GMM after Autoencoder

# %%
dims = [2, 5, 10, 15, 20]
results = []
history_dict = {} # To store loss curves

print("\n--- Starting Experiment 6: GMM with AE vs PCA ---")

for dim in dims:
    print(f"\nProcessing Dimension: {dim}")
    
    start_time = time.time()
    pca = PCA(n_components=dim)
    pca.fit(X)            
    X_pca = pca.transform(X)
    X_pca_recon = pca.inverse_transform(X_pca)
    
    # GMM on PCA
    gmm_pca = GMM(n_components=2, covariance_type='full')

    gmm_pca.fit(X_pca)
    gmm_pca_labels = gmm_pca.predict(X_pca)
    
    pca_time = time.time() - start_time
    
    # Metrics for PCA
    rec_error_pca = pca.reconstruction_error(X)
    resp, likelihoods = gmm_pca._e_step(X_pca)
    log_likelihood = likelihoods * len(X_pca)
    # Store PCA Results
    results.append({
        'Method': 'PCA-GMM',
        'Dim': dim,
        'Silhouette': silhouette_score(X_pca, gmm_pca_labels),
        'Davies-Bouldin': davies_bouldin_index(X_pca, gmm_pca_labels, gmm_pca.means_),
        'Calinski-Harabasz': calinski_harabasz_index(X_pca, gmm_pca_labels, gmm_pca.means_),
        'WCSS': wcss(X_pca, gmm_pca_labels, gmm_pca.means_),
        'BIC': gmm_pca.bic(X_pca),
        'AIC': gmm_pca.aic(X_pca),
        'Log-Likelihood': log_likelihood,
        'ARI': adjusted_rand_index(y, gmm_pca_labels),
        'NMI': normalized_mutual_information(y, gmm_pca_labels),
        'Purity': purity_score(y, gmm_pca_labels),
        'Reconstruction_MSE': rec_error_pca,
        'Time': pca_time
    })

    # ---------------------------
    # B. Autoencoder + GMM (Experiment 6)
    # ---------------------------
    start_time = time.time()
    input_dim = X_scaled.shape[1]
    
    # Define Architecture: [Input -> 64 -> 32 -> Bottleneck -> 32 -> 64 -> Output]
    layer_sizes = [input_dim, 64, 32, dim, 32, 64, input_dim]
    
    ae = Autoencoder(layer_sizes=layer_sizes, activation="sigmoid", learning_rate=0.001)
    
    # Train
    loss_history = ae.train(X_scaled, epochs=50, batch_size=128)
    history_dict[dim] = loss_history
    
    # Compress & Reconstruct
    X_ae = ae.encode(X_scaled)
    X_ae_recon = ae.reconstruct(X_scaled)
    
    # GMM on AE Latent Space
    gmm_ae = GMM(n_components=2, covariance_type='full')
    gmm_ae.fit(X_ae)
    gmm_ae_labels = gmm_ae.predict(X_ae)
    ae_time = time.time() - start_time
    
    results.append({
        'Method': 'AE-GMM',
        'Dim': dim,
        'Silhouette': silhouette_score(X_ae, gmm_ae_labels),
        'Davies-Bouldin': davies_bouldin_index(X_ae, gmm_ae_labels, gmm_ae.means_),
        'Calinski-Harabasz': calinski_harabasz_index(X_ae, gmm_ae_labels, gmm_ae.means_),
        'BIC': gmm_ae.bic(X_ae),
        'ARI': adjusted_rand_index(y, gmm_ae_labels),
        'NMI': normalized_mutual_information(y, gmm_ae_labels),
        'Purity': purity_score(y, gmm_ae_labels),
        'Reconstruction_MSE': manual_mse(X_scaled, X_ae_recon),
        'Time': ae_time
    })

# Convert results to DataFrame
df_res = pd.DataFrame(results)

# ==========================================
# 4. VISUALIZATION & ANALYSIS
# ==========================================

# A. Comparison Heatmap
plt.figure(figsize=(14, 8))
# Pivot for heatmap: Index=Dim, Columns=Metric, Split by Method
# We will just normalize metrics to 0-1 for heatmap visualization
numeric_cols = df_res.columns.drop(['Method', 'Dim'])
df_norm = df_res.copy()
for col in numeric_cols:
    df_norm[col] = (df_res[col] - df_res[col].min()) / (df_res[col].max() - df_res[col].min())

# Create a pivot table for the heatmap (Method+Dim vs Metrics)
pivot_heatmap = df_norm.set_index(['Method', 'Dim'])[numeric_cols]
sns.heatmap(pivot_heatmap, cmap='viridis', annot=False)
plt.title("Heatmap of Normalized Metrics (AE-GMM vs PCA-GMM)")
plt.tight_layout()
plt.show()

# B. Internal Validation Metrics Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics_to_plot = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']

for i, metric in enumerate(metrics_to_plot):
    sns.lineplot(data=df_res, x='Dim', y=metric, hue='Method', marker='o', ax=axes[i])
    axes[i].set_title(f'{metric} vs Dimensions')
    axes[i].grid(True)
plt.show()

# C. External Validation Metrics Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics_to_plot = ['ARI', 'NMI', 'Purity']

for i, metric in enumerate(metrics_to_plot):
    sns.lineplot(data=df_res, x='Dim', y=metric, hue='Method', marker='o', ax=axes[i])
    axes[i].set_title(f'{metric} vs Dimensions')
    axes[i].grid(True)
plt.show()

# D. Reconstruction Error Comparison
plt.figure(figsize=(8, 5))
sns.barplot(data=df_res, x='Dim', y='Reconstruction_MSE', hue='Method')
plt.title("Reconstruction Error (MSE): AE vs PCA")
plt.ylabel("MSE (Lower is better)")
plt.show()

# E. AE Training Loss Curves
plt.figure(figsize=(10, 6))
for dim, losses in history_dict.items():
    plt.plot(losses, label=f'Bottleneck Dim {dim}')
plt.title("Autoencoder Training Loss per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.show()

# F. 2D Projections
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. PCA Plot (Left)
# ------------------
pca_viz = PCA(n_components=2)
pca_viz.fit(X_scaled)
pca_2d = pca_viz.transform(X_scaled)

sns.scatterplot(
    x=pca_2d[:,0], y=pca_2d[:,1], 
    hue=y, palette='viridis', alpha=0.5, ax=axes[0]
)
axes[0].set_title("PCA 2D Projection")

# 2. Autoencoder Plot (Right) - THIS WAS MISSING
# ---------------------------
# Note: Ideally we retrain a dim=2 AE here, but slicing the 
# current high-dim model is acceptable for a quick check.
X_latent = ae.encode(X_scaled) 
ae_2d = X_latent[:, :2]

sns.scatterplot(
    x=ae_2d[:,0], y=ae_2d[:,1], 
    hue=y, palette='viridis', alpha=0.5, ax=axes[1]
)
axes[1].set_title("Autoencoder Latent Projection (First 2 Dims)")

plt.show()

# ==========================================
# 5. SUMMARY TABLES
# ==========================================
print("\n--- Summary Table (Averages) ---")
print(df_res.groupby('Method')[['Silhouette', 'ARI', 'Reconstruction_MSE', 'Time']].mean())

print("\n--- Best Configuration by ARI ---")
best_row = df_res.loc[df_res['ARI'].idxmax()]
print(best_row)


