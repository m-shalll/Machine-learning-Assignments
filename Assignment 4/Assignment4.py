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

