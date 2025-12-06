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

# %% [markdown] id="8FpQk1OWs2CG"
# # Imports

# %% id="OJi9h1PqtBbL"
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from numpy.linalg import slogdet, inv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import matplotlib.pyplot as plt

# %% [markdown] id="0v2yiieTs16d"
# # Part A

# %% [markdown] id="zBOt-O0tvbh-"
# ## A1. Dataset and Setup

# %% id="XUWgbn29vcNl"
digits = load_digits()
X = digits.data    # shape (1797, 64)
y = digits.target  # labels 0–9

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# %% [markdown] id="D7yzH8t5vf9n"
# ## A2. Gaussian Generative Model

# %% id="U-9MsynCvglp"
def estimate_parameters(X, y, lam):
    N, D = X.shape #Number of samples in training data, their 64 pixel features
    K = len(np.unique(y)) # Our labels (numbers from 0 to 9) = 10

    # Priors π_k, the frequency of each label
    pi = np.zeros(K)
    for k in range(K):
        pi[k] = np.sum(y == k) / N

    # Means μ_k, calculates average of each feature that belong to that class (64 features so 64 averages)
    mu = np.zeros((K, D))
    for k in range(K):
        mu[k] = X[y == k].mean(axis=0)

    # Shared covariance Σ
    Sigma = np.zeros((D, D))
    for i in range(N):
        k = y[i]
        # this line subtracts from all the 64 training features the means we calculated of these features
        # subtract 2 rows then reshape them to column vectors
        diff = (X[i] - mu[k]).reshape(D, 1)
        # matrix multiplication that produces 64x64 matrix
        Sigma += diff @ diff.T

    Sigma /= N
    Sigma += lam * np.eye(D) # add Lambda λ only to the diagonal matrix

    return pi, mu, Sigma


# %% id="LG8uzW58vj0z"
def compute_scores(X, pi, mu, Sigma_reg):
    N = X.shape[0]
    K = mu.shape[0]
    D = mu.shape[1]

    # Precompute inverse + log(det) for the multivariate Gaussian formula
    # logN(x;μ,Σ)= −(D/2)​log(2π) −(1/2)​log∣Σ∣ -(1/2)*(x−μ)^T * Σ^−1 * (x−μ)
    Sigma_inv = inv(Sigma_reg)
    # calculate the log of the determinant of the sigma (​log∣Σ∣)
    sign, logdet = slogdet(Sigma_reg)
    const = -0.5 * (D * np.log(2*np.pi) + logdet) #( −(D/2)​log(2π) −(1/2)​log∣Σ∣ )

    scores = np.zeros((N, K)) # liklehood of all the samples belonging to which of the 10 classes

    for k in range(K):
        diff = X - mu[k]                      # shape (N, D)
        tmp  = diff @ Sigma_inv               # shape (N, D)
        maha = np.sum(tmp * diff, axis=1)     # quadratic form
        log_gauss = const - 0.5 * maha
        scores[:, k] = np.log(pi[k]) + log_gauss

    return scores

def predict(X, pi, mu, Sigma_reg):
    scores = compute_scores(X, pi, mu, Sigma_reg)
    return np.argmax(scores, axis=1)


# %% [markdown]
# ## A3. Hyperparameter Tuning and Evaluation

# %%
lambdas = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
best_lambda = None
best_val_acc = -1

for lam in lambdas:
    pi, mu, Sigma = estimate_parameters(X_train, y_train,lam)
    y_val_pred = predict(X_val, pi, mu, Sigma)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"lambda={lam:.0e} | val accuracy={acc:.4f}")

    if acc > best_val_acc:
        best_val_acc = acc
        best_lambda = lam

print("\nBest lambda:", best_lambda)
print("Best validation accuracy:", best_val_acc)

# %% [markdown]
# # A3. Train on (train + validation) using best λ

# %%
X_final_train = np.vstack((X_train, X_val))
y_final_train = np.concatenate((y_train, y_val))

pi_f, mu_f, Sigma_f = estimate_parameters(X_final_train, y_final_train, best_lambda)
y_test_pred = predict(X_test, pi_f, mu_f, Sigma_f)

test_acc = accuracy_score(y_test, y_test_pred)
print("Test Accuracy: \n", test_acc)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Gaussian Generative Classifier — Confusion Matrix")
plt.show()

# %% [markdown]
# # A4. Report

# %% [markdown]
# ## 1. Explanation of the Generative Model
#
# ### Assumptions for the Generative Model
#
# In a Gaussian generative classifier, we assume that classification works by modeling how the data is generated for each class.  
# We make two key assumptions:
#
# - **Prior over labels**  
#   \( p(y = k) = \pi_k \)  
#   This represents how likely each digit class is before seeing any feature values.
#
# - **Class-conditional distribution**  
#   \( p(x \mid y = k) \sim \mathcal{N}(\mu_k, \Sigma) \)  
#   We assume that the feature vectors of all classes follow a multivariate Gaussian distribution with:
#   - a **class-specific mean vector** \( \mu_k \)
#   - a **shared covariance matrix** \( \Sigma \) across all classes  
#     (this is the LDA assumption)
#
# ### Estimating the Parameters
#
# From the training set:
#
# - **Class prior**  
#   \[
#   \pi_k = \frac{\text{number of samples in class } k}{\text{total number of samples}}
#   \]
#
# - **Class mean vector**  
#   \[
#   \mu_k = \frac{1}{N_k} \sum_{i: y_i = k} x_i
#   \]
#
# - **Shared covariance matrix**  
#   \[
#   \Sigma = \frac{1}{N} \sum_{k} \sum_{i: y_i = k} (x_i - \mu_k)(x_i - \mu_k)^T
#   \]
#
# ### Why We Regularize the Covariance
#
# High-dimensional data often makes the covariance matrix nearly singular.  
# To fix this, we apply:
#
# \[
# \Sigma_\lambda = \Sigma + \lambda I
# \]
#
# This **regularization**:
#
# - prevents the covariance matrix from becoming non-invertible  
# - stabilizes the model  
# - reduces overfitting by shrinking the covariance  
# - controls how smooth the decision boundaries are
#
# Smaller λ → more flexible model, may overfit  
# Larger λ → smoother model, may underfit
#
# ---
#
# ## 2. Table of Validation Accuracy for Different λ Values
#
# | λ value | Validation Accuracy |
# |--------|---------------------|
# | 1e-4   | … |
# | 1e-3   | … |
# | 1e-2   | … |
# | 1e-1   | … |
#
# (You will fill these in after running your code.)
#
# ---
#
# ## 3. Final Test Results
#
# Using the selected λ (the one with the highest validation accuracy), we retrain on the combined training + validation sets and evaluate on the test set.
#
# ### Performance Metrics
#
# - **Test accuracy:** …  
# - **Macro-averaged precision:** …  
# - **Macro-averaged recall:** …  
# - **Macro-averaged F1-score:** …
#
# ### Confusion Matrix
#
# (Insert your plotted confusion matrix as an image, or paste the numerical table.)
#
# ---
#
# ## 4. Discussion
#
# ### Digit Confusions
#
# Some digits are visually similar and therefore often misclassified.  
# Typical confusions might include:
#
# - **3 vs 5** — similar curved structure  
# - **4 vs 9** — similar upper sections  
# - **7 vs 9** — overlapping shapes in pixel space  
#
# Your own results will show which pairs were most confused in the confusion matrix.
#
# ### Effect of λ on Performance
#
# The choice of λ significantly affects classification:
#
# - Very small λ sometimes leads to instability or overfitting.
# - Larger λ smooths the covariance, improving generalization.
# - In our experiments, the best λ was …, which gave the highest validation accuracy.
#
# ### Overall Observations
#
# - The Gaussian generative classifier (LDA) performs well when each class forms a roughly Gaussian cluster.  
# - However, MNIST digits are not perfectly Gaussian and classes overlap in ways that the model cannot fully capture.
# - The model is efficient, interpretable, and relatively robust, but more advanced discriminative models (e.g., logistic regression, neural nets) usually outperform it on handwritten digits.
#

# %% [markdown] id="vyKmdrLqs1wu"
# # Part B

# %% id="THN-DXSMvf63"
path = kagglehub.dataset_download("uciml/adult-census-income")

csv_file = "adult.csv"
df = pd.read_csv(os.path.join(path, csv_file))

categorical_features = [
    'workclass',  'education', 'marital.status', 'occupation',
    'relationship', 'race', 'sex'
]
target = 'income'

X_arr = np.empty((len(df), len(categorical_features)), dtype=int)
label_encoders = {}

for i, col in enumerate(categorical_features):
    le = LabelEncoder()
    X_arr[:, i] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

y_arr = LabelEncoder().fit_transform(df[target].astype(str))

X_train, X_temp, y_train, y_temp = train_test_split(X_arr, y_arr, test_size=0.3, stratify=y_arr, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Data Loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")


# %% id="1oTe9bMzJ5Ax"
class NaiveBayes():
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        self.total_samples = len(y)
        self.num_classes = len(self.classes)

        for i, c in enumerate(self.classes):
            X_where_c = X[np.where(y == c)]
            class_count = X_where_c.shape[0]

            self.parameters.append([])

            for col_idx, col in enumerate(X_where_c.T):
                all_feature_values = np.unique(X[:, col_idx])
                num_feature_values = len(all_feature_values)

                values, counts = np.unique(col, return_counts=True)
                counts_dict = dict(zip(values, counts))

                denominator = class_count + (self.alpha * num_feature_values)

                probs = {}
                for val in all_feature_values:
                    count = counts_dict.get(val, 0)
                    probs[val] = (count + self.alpha) / denominator

                unseen_prob = (0 + self.alpha) / denominator

                parameters = {"probs": probs, "unseen_prob": unseen_prob}
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self, params, x):
        probs = params["probs"]
        unseen_prob = params["unseen_prob"]

        return probs.get(x, unseen_prob)

    def _calculate_prior(self, c):
        count_class_k = np.sum(self.y == c)
        numerator = count_class_k + self.alpha
        denominator = self.total_samples + (self.alpha * self.num_classes)

        return numerator / denominator

    def _classify(self, sample):
        posteriors = []
        for i, c in enumerate(self.classes):
            posterior = self._calculate_prior(c)

            for feature_idx, (feature_value, params) in enumerate(zip(sample, self.parameters[i])):
                likelihood = self._calculate_likelihood(params, feature_value)
                posterior *= likelihood

            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._classify(sample) for sample in X]
        return np.array(y_pred)


# %% [markdown] id="Qg-9EzmDs1ne"
# # Part C

# %% [markdown] id="P8cvmH5Ttm8m"
# ## Decision Trees

# %% [markdown] id="Pt-R_3h2tacV"
# ### Loading & Splitting Dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="Ul72e1Fhtjk8" outputId="13bbdfb8-6129-4961-b1d2-301031ff4560"
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)


# %% [markdown] id="0bdIrpt8tq7l"
# ### Entropy calculation

# %% id="ydGCEdsltwUe"
def entropy(y):
    vals, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))


# %% [markdown] id="Ud9EWWH0t7ZF"
# ### Information Gain Calculation

# %% id="P0SONj7Ct9Cd"
def information_gain(y_parent, y_left, y_right):
    n = len(y_parent)
    if len(y_left) == 0 or len(y_right) == 0:
        return 0
    return entropy(y_parent) - (len(y_left)/n)*entropy(y_left) - (len(y_right)/n)*entropy(y_right)


# %% [markdown] id="EdFjDdgRt_9f"
# ### Finding The Best Threshold to Split

# %% [markdown] id="SAn3-cGquCV2"
# #### Finding the best threshold for a given feature

# %% id="5tOs7vJEuFRd"
def best_split_for_feature(X_column, y):
    sorted_idx = np.argsort(X_column)
    X_sorted = X_column[sorted_idx]
    y_sorted = y[sorted_idx]

    distinct = np.where(np.diff(X_sorted) != 0)[0]
    if len(distinct) == 0:
        return None, 0

    thresholds = (X_sorted[distinct] + X_sorted[distinct + 1]) / 2

    best_gain = -1
    best_threshold = None

    for thr in thresholds:
        left_mask = X_column <= thr
        y_left = y[left_mask]
        y_right = y[~left_mask]
        gain = information_gain(y, y_left, y_right)
        if gain > best_gain:
            best_gain = gain
            best_threshold = thr

    return best_threshold, best_gain


# %% [markdown] id="ZxQOReF9uI3l"
# #### Finding the best feature to split on

# %% id="TN_5IbcnuLVM"
def best_split_overall(X, y):
    n_features = X.shape[1]
    best = {"feature": None, "threshold": None, "gain": -1}

    for j in range(n_features):
        thr, gain = best_split_for_feature(X[:, j], y)
        if gain > best["gain"]:
            best["gain"] = gain
            best["feature"] = j
            best["threshold"] = thr

    return best


# %% [markdown] id="NQ5wBR5xuNRm"
# ### Decision Tree Implementation

# %% [markdown] id="ovqt6lpbuRnv"
# #### Node Class Implementation

# %% id="qRaURTbXuPRO"
@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Any = None
    right: Any = None
    is_leaf: bool = False
    prediction: Optional[int] = None
    n_samples: int = 0
    class_counts: Dict[int, int] = field(default_factory=dict)


# %% [markdown] id="2HH1cuYVuV6t"
# #### DecisionTree Class Implementation

# %% id="W-41Zt-yun5V"
class DecisionTreeClassifier:
    def __init__(self, max_depth=8, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_importances_ = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)
        self.root = self._build_tree(X, y, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _build_tree(self, X, y, depth):
        node = Node()
        node.n_samples = len(y)

        vals, counts = np.unique(y, return_counts=True)
        node.class_counts = {int(v): int(c) for v, c in zip(vals, counts)}
        node.prediction = vals[np.argmax(counts)]

        if (depth >= self.max_depth or len(y) < self.min_samples_split or len(vals) == 1):
            node.is_leaf = True
            return node

        best = best_split_overall(X, y)
        if (best["gain"] <= 0 or best["feature"] is None):
            node.is_leaf = True
            return node

        feature = best["feature"]
        threshold = best["threshold"]
        node.feature = feature
        node.threshold = threshold

        self.feature_importances_[feature] += best["gain"]

        left_mask = X[:, feature] <= threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            node.is_leaf = True
            return node

        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)

        return node

    def _predict_one(self, x, node):
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


# %% [markdown]
# ### Hyperparameter Tuning

# %% [markdown]
# #### Maximum Grid Depth Tuning

# %%
depths = [2, 4, 6, 8, 10]
depth_accuracies = []
best_depth = -1
best_depth_acc = -1

for depth in depths:
    dec_tree = DecisionTreeClassifier(max_depth=depth)
    dec_tree.fit(X_train, y_train)
    predictions = dec_tree.predict(X_val)
    acc = accuracy_score(y_val, predictions)
    depth_accuracies.append(acc)
    
    print(f"Depth: {depth}, Validation Accuracy: {acc:.4f}")

    if acc > best_depth_acc:
        best_depth_acc = acc
        best_depth = depth

print(f"Best Depth: {best_depth}, Validation Accuracy: {best_depth_acc:.4f}")


# %%
plt.plot(depths, depth_accuracies)
plt.xlabel("Max Depth")
plt.ylabel("Validation Accuracy")
plt.title("Decision Tree Depth vs Accuracy")
plt.grid(True)
plt.show()

# %% [markdown]
# #### Minimum Samples For Splitting Tuning

# %%
min_samples = [2, 5, 10]
min_samples_accuracies = []
best_min_samples = -1
best_min_samples_acc = -1

for min_sample in min_samples:
    dec_tree = DecisionTreeClassifier(min_samples_split=min_sample)
    dec_tree.fit(X_train, y_train)
    predictions = dec_tree.predict(X_val)
    acc = accuracy_score(y_val, predictions)
    min_samples_accuracies.append(acc)
    
    print(f"Min Samples Split: {min_sample}, Validation Accuracy: {acc:.4f}")

    if acc > best_min_samples_acc:
        best_min_samples_acc = acc
        best_min_samples = min_sample

print(f"Best Min Samples Split: {best_min_samples}, Validation Accuracy: {best_min_samples_acc:.4f}")

# %%
plt.plot(min_samples, min_samples_accuracies)
plt.xlabel("Minimum Samples Split")
plt.ylabel("Validation Accuracy")
plt.title("Decision Tree Minimum Samples Split vs Accuracy")
plt.grid(True)
plt.show()

# %% [markdown]
# #### Combined Hyperparameter Tuning

# %%
best_combined_depth = -1
best_combined_min_samples = -1
best_combined_acc = -1

for depth in depths:
    for min_sample in min_samples:
        dec_tree = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_sample)
        dec_tree.fit(X_train, y_train)
        predictions = dec_tree.predict(X_val)
        acc = accuracy_score(y_val, predictions)
        
        print(f"Depth: {depth}, Min Samples Split: {min_sample}, Validation Accuracy: {acc:.4f}")

        if acc > best_combined_acc:
            best_combined_acc = acc
            best_combined_depth = depth
            best_combined_min_samples = min_sample

print(f"Best Combined Hyperparameters:\nDepth: {best_combined_depth}, Min Samples Split: {best_combined_min_samples}, Validation Accuracy: {best_combined_acc:.4f}")   

# %% [markdown]
# ### Evaluation On Test Dataset

# %%
X_final_train = np.vstack((X_train, X_val))
y_final_train = np.hstack((y_train, y_val))

final_dec_tree = DecisionTreeClassifier(max_depth=best_combined_depth, min_samples_split=best_combined_min_samples)
final_dec_tree.fit(X_final_train, y_final_train)
final_predictions = final_dec_tree.predict(X_test)

print("Test Set Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, final_predictions):.4f}")

print("Classification Report:")
print(classification_report(y_test, final_predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, final_predictions))

# %% [markdown] id="unhx9Ykds1e_"
# # Part D
