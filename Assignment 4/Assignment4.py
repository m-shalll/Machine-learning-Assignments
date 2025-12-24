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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
import numpy as np
import time


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
# # Experiments

# %% [markdown]
# ## 1) K-Means on original data

# %%
def experiment_1_kmeans_original(X, k_values):
    results = []

    for k in k_values:
        for init in ["random", "kmeans++"]:
            start = time.time()

            km = KMeans(
                n_clusters=k,
                init=init,
                max_iter=300,
                tol=1e-4,
                random_state=42
            )
            km.fit(X)

            elapsed = time.time() - start

            results.append({
                "k": k,
                "init": init,
                "wcss": km.inertia_,
                "silhouette": silhouette_score(X, km.labels_),
                "dbi": davies_bouldin_index(X, km.labels_, km.centroids_),
                "chi": calinski_harabasz_index(X, km.labels_, km.centroids_),
                "iterations": km.n_iter_,
                "time": elapsed
            })

    return results



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

