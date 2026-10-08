import os
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
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

output_dir = 'd:/Ass 1/outputs'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv('d:/Ass 1/data/ecommerce_infrastructure_scaling.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Time features
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

# 1. Correlation Heatmap
plt.figure(figsize=(12, 9), dpi=300)
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f',
            square=True, linewidths=.5, cbar_kws={'shrink': .8})
plt.title('Feature Correlation Matrix (Operational Telemetry & Targets)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_correlation_heatmap.png')
plt.close()

# Split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

num_cols = [
    'active_users', 'requests_per_min', 'orders_per_min', 'cpu_utilization',
    'memory_utilization', 'network_mbps', 'disk_utilization', 'response_time_ms',
    'current_capacity_rpm', 'hour_of_day', 'day_of_week', 'is_weekend',
    'requests_per_user', 'orders_to_requests_ratio', 'capacity_headroom', 'utilization_pressure'
]
cat_cols = ['sale_event', 'region']

num_imputer = SimpleImputer(strategy='median')
train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

train_encoded = pd.get_dummies(train_df[cat_cols], drop_first=False, dtype=int)
test_encoded = pd.get_dummies(test_df[cat_cols], drop_first=False, dtype=int)
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

X_train = pd.concat([train_df[num_cols].reset_index(drop=True), train_encoded.reset_index(drop=True)], axis=1)
X_test = pd.concat([test_df[num_cols].reset_index(drop=True), test_encoded.reset_index(drop=True)], axis=1)

y_reg_train = train_df['next_15min_requests'].values
y_reg_test = test_df['next_15min_requests'].values

y_clf_train = train_df['scale_up_required'].values
y_clf_test = test_df['scale_up_required'].values

# Regression Model
scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_reg_train)
y_reg_pred = lr.predict(X_test_scaled)

# 2. Time Series Overlay
plt.figure(figsize=(15, 6), dpi=300)
test_timestamps = test_df['timestamp'].values
plt.plot(test_timestamps, y_reg_test, label='Actual Demand (next_15min_requests)', color='#1f77b4', linewidth=1.5, alpha=0.85)
plt.plot(test_timestamps, y_reg_pred, label='Forecasted Demand (Linear Regression)', color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.9)
plt.fill_between(test_timestamps, y_reg_test, y_reg_pred, color='gray', alpha=0.15, label='Residual Gap (Error)')
plt.title('Chronological Demand Forecast Overlay (Evaluation Window: July 17 - 21)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Timestamp (10-minute intervals)', fontsize=11, fontweight='bold')
plt.ylabel('Requests Per Minute (RPM)', fontsize=11, fontweight='bold')
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_demand_forecast_timeseries.png')
plt.close()

# 3. Actual vs Predicted Scatter
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(y_reg_test, y_reg_pred, alpha=0.6, color='#2ca02c', edgecolors='k', linewidth=0.3, s=40)
min_val = min(y_reg_test.min(), y_reg_pred.min())
max_val = max(y_reg_test.max(), y_reg_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y = x)')
plt.title(f'Actual vs Predicted Demand Scatter (R² = {r2_score(y_reg_test, y_reg_pred):.4f})', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Actual Demand (RPM)', fontsize=11, fontweight='bold')
plt.ylabel('Predicted Demand (RPM)', fontsize=11, fontweight='bold')
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_demand_regression_scatter.png')
plt.close()

# Classification Models
dt = DecisionTreeClassifier(random_state=42, max_depth=5, class_weight='balanced')
dt.fit(X_train, y_clf_train)
y_dt_pred = dt.predict(X_test)

rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6, class_weight='balanced')
rf.fit(X_train, y_clf_train)
y_rf_pred = rf.predict(X_test)

# 4. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
cm_dt = confusion_matrix(y_clf_test, y_dt_pred)
cm_rf = confusion_matrix(y_clf_test, y_rf_pred)

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['No Scale-Up (0)', 'Scale-Up (1)'],
            yticklabels=['No Scale-Up (0)', 'Scale-Up (1)'])
axes[0].set_title(f'Decision Tree Classifier\nRecall: {recall_score(y_clf_test, y_dt_pred):.2%}, F1: {f1_score(y_clf_test, y_dt_pred):.2%}', fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontweight='bold')
axes[0].set_ylabel('Actual Label', fontweight='bold')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=['No Scale-Up (0)', 'Scale-Up (1)'],
            yticklabels=['No Scale-Up (0)', 'Scale-Up (1)'])
axes[1].set_title(f'Random Forest Classifier (Recommended)\nRecall: {recall_score(y_clf_test, y_rf_pred):.2%}, F1: {f1_score(y_clf_test, y_rf_pred):.2%}', fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontweight='bold')
axes[1].set_ylabel('Actual Label', fontweight='bold')

plt.suptitle('Proactive Scale-Up Decisioning: Confusion Matrix Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_classification_confusion_matrices.png')
plt.close()

# 5. Random Forest Feature Importance
importances = rf.feature_importances_
feature_names = X_train.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(10, 7), dpi=300)
feat_imp.plot(kind='barh', color='#4a90e2', edgecolor='black', linewidth=0.5)
plt.title('Random Forest Feature Importance (Scale-Up Trigger)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Gini Importance', fontsize=11, fontweight='bold')
plt.ylabel('Features', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_rf_feature_importance.png')
plt.close()

# 6. Unsupervised Clustering & Elbow/Silhouette
unsup_features = [
    'active_users', 'requests_per_min', 'orders_per_min', 'cpu_utilization',
    'memory_utilization', 'network_mbps', 'disk_utilization', 'response_time_ms',
    'current_capacity_rpm', 'requests_per_user', 'capacity_headroom', 'utilization_pressure'
]
imputer_unsup = SimpleImputer(strategy='median')
df_unsup_imputed = imputer_unsup.fit_transform(df[unsup_features])
scaler_unsup = StandardScaler()
X_unsup_scaled = scaler_unsup.fit_transform(df_unsup_imputed)

inertias = []
silhouettes = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_unsup_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_unsup_scaled, km_labels))

fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
ax1.set_ylabel('Inertia (WCSS)', color=color, fontweight='bold')
ax1.plot(k_range, inertias, 'o-', color=color, linewidth=2, markersize=7)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')
ax2.plot(k_range, silhouettes, 's--', color=color, linewidth=2, markersize=7)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('K-Means Cluster Evaluation: Elbow Method & Silhouette Scores', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_kmeans_elbow_silhouette.png')
plt.close()

km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df['kmeans_cluster'] = km_final.fit_predict(X_unsup_scaled)

# 7. Hierarchical Dendrogram
plt.figure(figsize=(12, 6), dpi=300)
sample_indices = np.random.RandomState(42).choice(len(X_unsup_scaled), size=150, replace=False)
X_sample = X_unsup_scaled[sample_indices]
link_matrix = linkage(X_sample, method='ward')
dendrogram(link_matrix, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=9., show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage, Representative Operational Snapshots)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Cluster Sample Index / Size', fontsize=11, fontweight='bold')
plt.ylabel('Ward Linkage Distance', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/07_hierarchical_dendrogram.png')
plt.close()

# 8. PCA 2D Visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_unsup_scaled)
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]

plt.figure(figsize=(10, 7), dpi=300)
cluster_names = {0: 'Steady-State Normal', 1: 'Elevated Promo/Sale', 2: 'Critical Flash Surge'}
df['cluster_label'] = df['kmeans_cluster'].map(cluster_names)
sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='cluster_label', palette='Set1', alpha=0.7, s=45, edgecolor='none')
plt.title(f'PCA 2D Projection Colored by Infrastructure States (Variance Explained: {sum(pca.explained_variance_ratio_):.1%})', fontsize=14, fontweight='bold', pad=12)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} Variance)', fontsize=11, fontweight='bold')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} Variance)', fontsize=11, fontweight='bold')
plt.legend(title='Infrastructure Operating State', frameon=True)
plt.tight_layout()
plt.savefig(f'{output_dir}/08_pca_2d_clusters.png')
plt.close()

# 9. PCA Colored by Sale Event
plt.figure(figsize=(10, 7), dpi=300)
sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='sale_event', palette={'Normal': '#2ca02c', 'Sale': '#1f77b4', 'Flash Sale': '#d62728'}, alpha=0.7, s=45, edgecolor='none')
plt.title('PCA 2D Projection Colored by Business Sale Event', fontsize=14, fontweight='bold', pad=12)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} Variance)', fontsize=11, fontweight='bold')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} Variance)', fontsize=11, fontweight='bold')
plt.legend(title='Sale Event Context', frameon=True)
plt.tight_layout()
plt.savefig(f'{output_dir}/09_pca_2d_sale_event.png')
plt.close()

# 10. Demand vs Response Time & Headroom
plt.figure(figsize=(10, 6), dpi=300)
scatter = plt.scatter(df['requests_per_min'], df['response_time_ms'], c=df['capacity_headroom'], cmap='coolwarm_r', alpha=0.7, s=40, edgecolors='none')
cbar = plt.colorbar(scatter)
cbar.set_label('Capacity Headroom (RPM Buffer)', fontweight='bold')
plt.axhline(200, color='red', linestyle='--', linewidth=1.5, label='SRE Critical SLA Threshold (200ms)')
plt.title('Throughput vs Response Latency (Colored by Capacity Headroom)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Current Requests Per Minute (RPM)', fontsize=11, fontweight='bold')
plt.ylabel('Response Time (ms)', fontsize=11, fontweight='bold')
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig(f'{output_dir}/10_demand_vs_response_time.png')
plt.close()

print('All 10 figures successfully generated and saved to d:/Ass 1/outputs!')

