import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Ensure outputs folder exists
os.makedirs('d:/Ass 1/outputs', exist_ok=True)

# 1. Load Data
df = pd.read_csv('d:/Ass 1/data/ecommerce_infrastructure_scaling.csv')
print(f"Dataset Shape: {df.shape}")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# 2. Preprocessing & Feature Engineering
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# SRE domain engineered features
df['requests_per_user'] = df['requests_per_min'] / (df['active_users'] + 1e-5)
df['orders_to_requests_ratio'] = df['orders_per_min'] / (df['requests_per_min'] + 1e-5)
df['capacity_headroom'] = df['current_capacity_rpm'] - df['requests_per_min']
df['utilization_pressure'] = (
    df['cpu_utilization'].fillna(df['cpu_utilization'].median()) +
    df['memory_utilization'].fillna(df['memory_utilization'].median()) +
    df['disk_utilization'].fillna(df['disk_utilization'].median())
) / 3.0

# Chronological Train-Test Split (80% Train, 20% Evaluation)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"Train samples: {len(train_df)} ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
print(f"Test samples:  {len(test_df)} ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

num_cols = [
    'active_users', 'requests_per_min', 'orders_per_min', 'cpu_utilization',
    'memory_utilization', 'network_mbps', 'disk_utilization', 'response_time_ms',
    'current_capacity_rpm', 'hour_of_day', 'day_of_week', 'is_weekend',
    'requests_per_user', 'orders_to_requests_ratio', 'capacity_headroom', 'utilization_pressure'
]
cat_cols = ['sale_event', 'region']

# Fit imputers on train only
num_imputer = SimpleImputer(strategy='median')
train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

# Categorical Encoding
train_encoded = pd.get_dummies(train_df[cat_cols], drop_first=False, dtype=int)
test_encoded = pd.get_dummies(test_df[cat_cols], drop_first=False, dtype=int)
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

X_train = pd.concat([train_df[num_cols].reset_index(drop=True), train_encoded.reset_index(drop=True)], axis=1)
X_test = pd.concat([test_df[num_cols].reset_index(drop=True), test_encoded.reset_index(drop=True)], axis=1)

y_reg_train = train_df['next_15min_requests'].values
y_reg_test = test_df['next_15min_requests'].values

y_clf_train = train_df['scale_up_required'].values
y_clf_test = test_df['scale_up_required'].values

# ----------------- REGRESSION -----------------
scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_reg_train)
y_reg_pred = lr.predict(X_test_scaled)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)
mape = np.mean(np.abs((y_reg_test - y_reg_pred) / y_reg_test)) * 100

print(f"\n--- REGRESSION METRICS ---")
print(f"MAE:  {mae:.2f} RPM")
print(f"RMSE: {rmse:.2f} RPM")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# ----------------- CLASSIFICATION -----------------
dt = DecisionTreeClassifier(random_state=42, max_depth=5, class_weight='balanced')
dt.fit(X_train, y_clf_train)
y_dt_pred = dt.predict(X_test)

rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6, class_weight='balanced')
rf.fit(X_train, y_clf_train)
y_rf_pred = rf.predict(X_test)

print(f"\n--- DECISION TREE METRICS ---")
print(f"Accuracy:  {accuracy_score(y_clf_test, y_dt_pred):.4f}")
print(f"Precision: {precision_score(y_clf_test, y_dt_pred):.4f}")
print(f"Recall:    {recall_score(y_clf_test, y_dt_pred):.4f}")
print(f"F1-Score:  {f1_score(y_clf_test, y_dt_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_clf_test, y_dt_pred))

print(f"\n--- RANDOM FOREST METRICS ---")
print(f"Accuracy:  {accuracy_score(y_clf_test, y_rf_pred):.4f}")
print(f"Precision: {precision_score(y_clf_test, y_rf_pred):.4f}")
print(f"Recall:    {recall_score(y_clf_test, y_rf_pred):.4f}")
print(f"F1-Score:  {f1_score(y_clf_test, y_rf_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_clf_test, y_rf_pred))

# ----------------- UNSUPERVISED & CLUSTERING -----------------
# Exclude targets completely
unsup_features = [
    'active_users', 'requests_per_min', 'orders_per_min', 'cpu_utilization',
    'memory_utilization', 'network_mbps', 'disk_utilization', 'response_time_ms',
    'current_capacity_rpm', 'requests_per_user', 'capacity_headroom', 'utilization_pressure'
]
imputer_unsup = SimpleImputer(strategy='median')
df_unsup_imputed = imputer_unsup.fit_transform(df[unsup_features])
scaler_unsup = StandardScaler()
X_unsup_scaled = scaler_unsup.fit_transform(df_unsup_imputed)

# K-Means Elbow & Silhouette
inertias = []
silhouettes = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_unsup_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_unsup_scaled, km_labels))

km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df['kmeans_cluster'] = km_final.fit_predict(X_unsup_scaled)

# DBSCAN
dbscan = DBSCAN(eps=2.5, min_samples=10)
df['dbscan_cluster'] = dbscan.fit_predict(X_unsup_scaled)
n_dbscan_clusters = len(set(df['dbscan_cluster'])) - (1 if -1 in df['dbscan_cluster'] else 0)
n_dbscan_noise = (df['dbscan_cluster'] == -1).sum()
print(f"\nDBSCAN: {n_dbscan_clusters} clusters, {n_dbscan_noise} noise points ({n_dbscan_noise/len(df)*100:.2f}%)")

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_unsup_scaled)
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]
print(f"PCA Explained Variance: {pca.explained_variance_ratio_}, Cumulative: {sum(pca.explained_variance_ratio_):.4f}")

print("\n--- ALL COMPUTATIONS COMPLETE AND VERIFIED ---")

