# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:07:39 2024

@author: m_pan
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Step A: Retrieve and load the Olivetti faces dataset
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target
n_samples, n_features = X.shape
print(f"Number of samples: {n_samples}, Number of features: {n_features}")

# Step B: Split the dataset using stratified sampling
# Using a 70-15-15 split for training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Rationale for split ratio:
# A common split ratio is 70% training, 15% validation, and 15% test. 
# This allows sufficient data for training while reserving enough data for tuning and testing.

# Step C: Use K-Means to reduce dimensionality and determine the number of clusters
# We can use PCA for initial dimensionality reduction
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_train)

# Determine the optimal number of clusters using silhouette score
silhouette_scores = []
range_n_clusters = range(2, 15)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the silhouette score is: {silhouette_avg}")

# Plotting the silhouette scores
plt.figure()
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Optimal number of clusters can be selected based on the highest silhouette score
optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters based on silhouette score: {optimal_n_clusters}")

kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
kmeans.fit(X_pca)
X_kmeans = kmeans.transform(X_pca)

# Step D: Train a classifier to predict which person is represented
# Using Logistic Regression and k-fold cross-validation
classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5)

# Store metrics for averaging later
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, val_index in cv.split(X_train, y_train):
    X_cv_train, X_cv_val = X_train[train_index], X_train[val_index]
    y_cv_train, y_cv_val = y_train[train_index], y_train[val_index]
    classifier.fit(X_cv_train, y_cv_train)
    y_pred = classifier.predict(X_cv_val)

    # Calculate metrics for the current fold
    report = classification_report(y_cv_val, y_pred, output_dict=True)
    accuracy_scores.append(report['accuracy'])
    precision_scores.append(report['macro avg']['precision'])
    recall_scores.append(report['macro avg']['recall'])
    f1_scores.append(report['macro avg']['f1-score'])

# Print average metrics across all folds
print("Cross-Validation Results:")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")

# Evaluate on the validation set
classifier.fit(X_train, y_train)
y_val_pred = classifier.predict(X_val)
print("Validation set classification report:")
print(classification_report(y_val, y_val_pred))

# Step E: Apply DBSCAN algorithm to the Olivetti Faces dataset
# Preprocessing the images (using PCA)
X_pca_dbscan = pca.fit_transform(X)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as necessary
dbscan_labels = dbscan.fit_predict(X_pca_dbscan)

# Rationale for similarity measure:
# DBSCAN uses a density-based approach, which is suitable for facial images as it can identify clusters of varying shapes and sizes, allowing for the presence of noise.

# # Visualize the results of DBSCAN
# plt.figure(figsize=(10, 8))
# unique_labels = set(dbscan_labels)
# colors = [plt.cm.Spectral(i / len(unique_labels)) for i in range(len(unique_labels))]

# for k, col in zip(unique_labels, colors):
#     class_member_mask = (dbscan_labels == k)
#     xy = X_pca_dbscan[class_member_mask]
#     plt.scatter(xy[:, 0], xy[:, 1], c=col, label=f'Cluster {k}', edgecolor='k', s=50)

# # Highlight noise points
# noise_mask = (dbscan_labels == -1)
# plt.scatter(X_pca_dbscan[noise_mask, 0], X_pca_dbscan[noise_mask, 1], c='k', label='Noise', s=50)

# plt.title("DBSCAN Clustering of Olivetti Faces Dataset")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.legend()
# plt.colorbar(label='Cluster Label')
# plt.show()
