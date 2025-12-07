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

# %% [markdown] id="8FpQk1OWs2CG"
# # Imports

# %% id="OJi9h1PqtBbL"
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import LabelEncoder
from numpy.linalg import slogdet, inv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import kagglehub 
from kagglehub import KaggleDatasetAdapter
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

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

                unseen_prob = (self.alpha) / denominator

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

    def _calculate_posteriors(self, sample):
        log_posteriors = []
        for i, c in enumerate(self.classes):
            posterior = np.log(self._calculate_prior(c))

            for feature_idx, (feature_value, params) in enumerate(zip(sample, self.parameters[i])):
                likelihood = self._calculate_likelihood(params, feature_value)
                posterior += np.log(likelihood)

            log_posteriors.append(posterior)

        return log_posteriors

    def _predict_label(self, sample):
        log_posteriors = self._calculate_posteriors(sample)
        return self.classes[np.argmax(log_posteriors)]

    def predict(self, X):
        y_pred = [self._predict_label(sample) for sample in X]
        return np.array(y_pred)


# %%
def evaluate_model(model, X, y):
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    return accuracy


# %%
print("\nSmoothing Parameter Analysis:")
alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
best_alpha = 1.0
best_accuracy = 0.0

alpha_results = []
for alpha in alpha_values:
    nb_model = NaiveBayes(alpha=alpha)
    acc = evaluate_model(nb_model, X_val, y_val)
    alpha_results.append((alpha, acc))
    print(f"Alpha = {alpha}: Validation Accuracy = {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_alpha = alpha

print(f"\nOptimal alpha found on validation set: {best_alpha} with Accuracy: {best_accuracy:.4f}")

optimal_nb = NaiveBayes(alpha=best_alpha)
optimal_nb.fit(X_train, y_train)

# %%
print("Feature Selection Analysis:")
feature_names = np.array(categorical_features)
feature_indices = {name: i for i, name in enumerate(feature_names)}

# 1-Base Model (All Features)
base_accuracy = evaluate_model(optimal_nb, X_val, y_val)
print(f"All Features: Accuracy = {base_accuracy:.4f}")

# 2-Removing 'race'
race_idx = feature_indices['race']
X_train_no_race = np.delete(X_train, race_idx, axis=1)
X_val_no_race = np.delete(X_val, race_idx, axis=1)

nb_no_race = NaiveBayes(alpha=best_alpha)
nb_no_race.fit(X_train_no_race, y_train)
acc_no_race = evaluate_model(nb_no_race, X_val_no_race, y_val)
print(f"Excluding 'race': Accuracy = {acc_no_race:.4f} (Change: {acc_no_race - base_accuracy:.4f})")

# 3-Removing 'sex'
sex_idx = feature_indices['sex']
X_train_no_sex = np.delete(X_train, sex_idx, axis=1)
X_val_no_sex = np.delete(X_val, sex_idx, axis=1)

nb_no_sex = NaiveBayes(alpha=best_alpha)
nb_no_sex.fit(X_train_no_sex, y_train)
acc_no_sex = evaluate_model(nb_no_sex, X_val_no_sex, y_val)
print(f"Excluding 'sex': Accuracy = {acc_no_sex:.4f} (Change: {acc_no_sex - base_accuracy:.4f})")

# 3-Removing 'sex', 'race' and 'relationship'
features_to_remove = ['sex', 'race', 'relationship']
remove_idx = [feature_indices[feat] for feat in features_to_remove]
X_train_no_subset = np.delete(X_train, remove_idx, axis=1)
X_val_no_subset = np.delete(X_val, remove_idx, axis=1)

nb_no_subset = NaiveBayes(alpha=best_alpha)
nb_no_subset.fit(X_train_no_subset, y_train)
acc_no_subset = evaluate_model(nb_no_subset, X_val_no_subset, y_val)
print(f"Excluding 'sex', 'race' and 'relationship': Accuracy = {acc_no_subset:.4f} (Change: {acc_no_subset - base_accuracy:.4f})")

# 4-Only 'workclass' and 'occupation' features
workclass_idx = feature_indices['workclass']
occupation_idx = feature_indices['occupation']
subset_indices = [workclass_idx, occupation_idx]

X_train_subset = X_train[:, subset_indices]
X_val_subset = X_val[:, subset_indices]

nb_subset = NaiveBayes(alpha=best_alpha)
nb_subset.fit(X_train_subset, y_train)
acc_subset = evaluate_model(nb_subset, X_val_subset, y_val)
print(f"'workclass' and 'occupation' Only: Accuracy = {acc_subset:.4f} (Change: {acc_subset - base_accuracy:.4f})")

optmal_nb = nb_no_subset

# %%
# Probability Analysis:
print("\nProbability Analysis: ")

log_posteriors_val = np.array([optimal_nb._calculate_posteriors(sample) for sample in X_val])

scores_class_1 = log_posteriors_val[:, 1]

scores_true_0 = scores_class_1[y_val == 0] # Scores for actual <=50K
scores_true_1 = scores_class_1[y_val == 1] # Scores for actual >50K

plt.figure(figsize=(10, 6))

# Histogram for Class 0 (<=50k)
plt.hist(scores_true_0, bins=50, alpha=0.5, label='True Class <=50K', color='blue', density=True)

# Histogram for Class 1 (>50k)
plt.hist(scores_true_1, bins=50, alpha=0.5, label='True Class >50K', color='orange', density=True)

plt.title('Distribution of Log Posterior Scores for Class >50K')
plt.xlabel('Log Posterior Score (Higher = More likely >50K)')
plt.ylabel('Density (Frequency)')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()

print("Analysis:")
print("This plot shows how confident the model is for each class.")
print("The blue bars shows scores for people who earn <=50K.")
print("The orange distribution shows scores for people who earn >50K.")
print("These two distributions should be far apart with little overlap.")
print("The overlap area represents where the model is confused and making errors.")

# %%
# Independence Assumption
print("\nIndependence Assumption:")
print("The Naive Bayes classifier assumes that all features are conditionally independent given the class.")
print("In this dataset, this assumption is likely violated in several places:")
print("- Marital Status and Relationship: These are highly correlated.")
print("- Education and Occupation: These are often correlated as higher education restricts certain occupations.")
print("- Sex and Relationship: These are highly correlated as 'Husband' implies 'Male' and 'Wife' implies 'Female'.")
print("When features are dependent, the model 'double counts', leading to overconfident probabilities , though the ranking often but not always remains correct.")

# %%
# Performance Comparison with MultinomialNB
print("\nPerformance Comparison on Test Set:")

y_pred_custom = optimal_nb.predict(X_test)
custom_acc_test = accuracy_score(y_test, y_pred_custom)
print(f"Custom NaiveBayes: Test Accuracy = {custom_acc_test:.4f}")

# MultinomialNB
sklearn_nb = MultinomialNB(alpha=best_alpha)
sklearn_nb.fit(X_train, y_train)
y_pred_sklearn = sklearn_nb.predict(X_test)
sklearn_acc_test = accuracy_score(y_test, y_pred_sklearn)
print(f"Scikit MultinomialNB: Test Accuracy = {sklearn_acc_test:.4f}")

# Conclusion
diff = custom_acc_test - sklearn_acc_test
print(f"Difference (Custom - Sklearn): {diff:.4f}")

# %% [markdown] id="Qg-9EzmDs1ne"
# # Part C

# %% [markdown] id="P8cvmH5Ttm8m"
# ## Decision Trees

# %% [markdown] id="Pt-R_3h2tacV"
# ## Loading & Splitting Dataset

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
# ## Entropy calculation

# %% id="ydGCEdsltwUe"
def entropy(y):
    vals, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))


# %% [markdown] id="Ud9EWWH0t7ZF"
# ## Information Gain Calculation

# %% id="P0SONj7Ct9Cd"
def information_gain(y_parent, y_left, y_right):
    n = len(y_parent)
    if len(y_left) == 0 or len(y_right) == 0:
        return 0
    return entropy(y_parent) - (len(y_left)/n)*entropy(y_left) - (len(y_right)/n)*entropy(y_right)


# %% [markdown] id="EdFjDdgRt_9f"
# ## Finding The Best Threshold to Split

# %% [markdown] id="SAn3-cGquCV2"
# ### Finding the best threshold for a given feature

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
# ### Finding the best feature to split on

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
# ## Decision Tree Implementation

# %% [markdown] id="ovqt6lpbuRnv"
# ### Node Class Implementation

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
# ### DecisionTree Class Implementation

# %% id="W-41Zt-yun5V"
class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
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

# %% [markdown] id="unhx9Ykds1e_"
# # Part D
