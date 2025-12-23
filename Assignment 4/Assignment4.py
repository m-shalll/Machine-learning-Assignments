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
import numpy as np


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

# %% [markdown]
# # Internal Metrics

# %% [markdown]
# # External Metrics
