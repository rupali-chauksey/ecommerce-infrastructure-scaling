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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Set clean aesthetic styling matching professional benchmarks
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

output_dir = 'd:/Ass 1/outputs'
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
df = pd.read_csv('d:/Ass 1/data/ecommerce_infrastructure_scaling.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Feature engineering
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

df['requests_per_user'] = df['requests_per_min'] / (df['active_users'] + 1e-5)
df['orders_to_requests_ratio'] = df['orders_per_min'] / (df['requests_per_min'] + 1e-5)
df['capacity_headroom'] = df['current_capacity_rpm'] - df['requests_per_min']
df['utilization_pressure'] = (
    df['cpu_utilization'].fillna(df['cpu_utilization'].median()) +
    df['memory_utilization'].fillna(df['memory_utilization'].median()) +
    df['disk_utilization'].fillna(df['disk_utilization'].median())
) / 3.0

# -------------------------------------------------------------
# PLOT 1: Task 1 - EDA Distributions & Target Patterns
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300)

# 1a. Demand Target Distribution
sns.histplot(df['next_15min_requests'], kde=True, ax=ax1, color='#3498db', bins=30, alpha=0.55, edgecolor='black', linewidth=0.5)
ax1.axvline(df['next_15min_requests'].mean(), color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Mean ({df["next_15min_requests"].mean():.0f} RPM)')
ax1.axvline(df['next_15min_requests'].median(), color='#27ae60', linestyle=':', linewidth=1.5, label=f'Median ({df["next_15min_requests"].median():.0f} RPM)')
ax1.set_title('Distribution of Upcoming Demand (15-min Ahead)', fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel('Requests Per Minute (RPM)', fontsize=9, fontweight='bold')
ax1.set_ylabel('Telemetry Snapshot Count', fontsize=9, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=8.5)

# 1b. Target Classification Distribution
scale_counts = df['scale_up_required'].value_counts()
bars = ax2.bar(['Capacity Sufficient (0)', 'Scale-Up Required (1)'],
               [scale_counts.get(0, 0), scale_counts.get(1, 0)],
               color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.6, width=0.45)
ax2.set_title('Proactive Scale-Up Class Distribution', fontsize=11, fontweight='bold', pad=10)
ax2.set_ylabel('Snapshot Count', fontsize=9, fontweight='bold')
ax2.set_ylim(0, len(df) * 1.1)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 40,
             f'{int(height)} ({height/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/01_eda_and_targets.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# PLOT 2: Task 2 - Chronological Split & Missing Telemetry
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300)
split_idx = int(len(df) * 0.8)

# 2a. Chronological Split
ax1.plot(df['timestamp'].iloc[:split_idx], df['requests_per_min'].iloc[:split_idx], color='#3498db', lw=1.2, alpha=0.85, label='Train (80% = 2,418)')
ax1.plot(df['timestamp'].iloc[split_idx:], df['requests_per_min'].iloc[split_idx:], color='#e74c3c', lw=1.3, alpha=0.9, label='Eval (20% = 605)')
ax1.axvline(df['timestamp'].iloc[split_idx], color='black', linestyle='--', linewidth=1, label='Split (July 17)')
ax1.set_title('Chronological Time-Aware Split (No Shuffling)', fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel('Timestamp (July 2026)', fontsize=9, fontweight='bold')
ax1.set_ylabel('Requests / Min (RPM)', fontsize=9, fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fontsize=8)

# 2b. Missing Values Bar Chart
missing_counts = df.isnull().sum()
missing_filtered = missing_counts[missing_counts > 0]
bars2 = ax2.barh(missing_filtered.index, missing_filtered.values, color='#9b59b6', edgecolor='black', linewidth=0.6, height=0.45)
ax2.set_title('Missing Telemetry Counts (Imputed via Train Median/Mode)', fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel('Missing Observations Count', fontsize=9, fontweight='bold')
ax2.set_xlim(0, 25)
for bar in bars2:
    width = bar.get_width()
    ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
             f'{int(width)} ({width/len(df)*100:.2f}%)',
             ha='left', va='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/02_preprocessing_and_split.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# PLOT 3: Task 3 - Feature Correlation Matrix
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
corr_features = ['active_users', 'requests_per_min', 'orders_per_min', 'cpu_utilization',
                 'memory_utilization', 'response_time_ms', 'current_capacity_rpm',
                 'requests_per_user', 'capacity_headroom', 'utilization_pressure',
                 'next_15min_requests', 'scale_up_required']
corr = df[corr_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f',
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax, annot_kws={'size': 7.5})
ax.set_title('Operational Telemetry & Target Correlation Matrix', fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_feature_correlation.png', bbox_inches='tight')
plt.close()

# Train/Test preproc
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

# Regression
scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_reg_train)
y_reg_pred = lr.predict(X_test_scaled)

# -------------------------------------------------------------
# PLOT 4: Task 4 - Demand Forecasting (Regression)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300)

# 4a. Time series overlay
test_times = test_df['timestamp'].values
ax1.plot(test_times, y_reg_test, label='Actual Demand', color='#1f77b4', lw=1.3, alpha=0.85)
ax1.plot(test_times, y_reg_pred, label='Forecast (Linear Reg)', color='#ff7f0e', ls='--', lw=1.3, alpha=0.9)
ax1.set_title(f'Demand Forecast Overlay (MAE: {mean_absolute_error(y_reg_test, y_reg_pred):.0f} RPM)', fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel('Timestamp (Eval Window)', fontsize=9, fontweight='bold')
ax1.set_ylabel('Requests / Min (RPM)', fontsize=9, fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fontsize=8)

# 4b. Scatter
ax2.scatter(y_reg_test, y_reg_pred, color='#2ecc71', alpha=0.55, edgecolors='k', linewidth=0.3, s=28)
min_v, max_v = min(y_reg_test.min(), y_reg_pred.min()), max(y_reg_test.max(), y_reg_pred.max())
ax2.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label='Ideal (y = x)')
ax2.set_title(f'Actual vs Predicted Demand (R²: {r2_score(y_reg_test, y_reg_pred):.4f})', fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel('Actual Demand (RPM)', fontsize=9, fontweight='bold')
ax2.set_ylabel('Predicted Demand (RPM)', fontsize=9, fontweight='bold')
ax2.legend(loc='upper left', frameon=True, fontsize=8)

plt.tight_layout()
plt.savefig(f'{output_dir}/04_demand_forecasting.png', bbox_inches='tight')
plt.close()

# Classification
dt = DecisionTreeClassifier(random_state=42, max_depth=5, class_weight='balanced')
dt.fit(X_train, y_clf_train)
y_dt_pred = dt.predict(X_test)

rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6, class_weight='balanced')
rf.fit(X_train, y_clf_train)
y_rf_pred = rf.predict(X_test)

# -------------------------------------------------------------
# PLOT 5: Task 5 - Proactive Scale-Up Decisioning (Classification)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300)

# 5a. Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_clf_test, y_rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax1,
            xticklabels=['No Scale (0)', 'Scale (1)'], yticklabels=['No Scale (0)', 'Scale (1)'],
            annot_kws={'size': 11, 'fontweight': 'bold'})
ax1.set_title(f'Random Forest Confusion Matrix\nRecall: {recall_score(y_clf_test, y_rf_pred):.1%}, F1: {f1_score(y_clf_test, y_rf_pred):.1%}', fontsize=10.5, fontweight='bold', pad=10)
ax1.set_xlabel('Predicted Action', fontsize=9, fontweight='bold')
ax1.set_ylabel('Actual Requirement', fontsize=9, fontweight='bold')

# 5b. Top Feature Importance
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=True).tail(8)
ax2.barh(importances.index, importances.values, color='#3498db', edgecolor='black', linewidth=0.5, height=0.55)
ax2.set_title('Top Decision Drivers (Gini Importance)', fontsize=10.5, fontweight='bold', pad=10)
ax2.set_xlabel('Importance Score', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/05_proactive_classification.png', bbox_inches='tight')
plt.close()

# Unsupervised Clustering
unsup_cols = [
    'active_users', 'requests_per_min', 'orders_per_min', 'cpu_utilization',
    'memory_utilization', 'network_mbps', 'disk_utilization', 'response_time_ms',
    'current_capacity_rpm', 'requests_per_user', 'capacity_headroom', 'utilization_pressure'
]
imputer_unsup = SimpleImputer(strategy='median')
df_unsup_imputed = imputer_unsup.fit_transform(df[unsup_cols])
scaler_unsup = StandardScaler()
X_unsup_scaled = scaler_unsup.fit_transform(df_unsup_imputed)

inertias = []
silhouettes = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_unsup_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_unsup_scaled, labels))

km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df['kmeans_cluster'] = km_final.fit_predict(X_unsup_scaled)

# -------------------------------------------------------------
# PLOT 6: Task 6 - Traffic & Infrastructure States (Clustering)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300)

# 6a. Elbow & Silhouette
color = '#1f77b4'
ax1.plot(k_range, inertias, 'o-', color=color, lw=1.8, markersize=5.5, label='Inertia')
ax1.set_xlabel('Number of Clusters (K)', fontsize=9, fontweight='bold')
ax1.set_ylabel('Inertia (WCSS)', color=color, fontsize=9, fontweight='bold')
ax1_twin = ax1.twinx()
color2 = '#e74c3c'
ax1_twin.plot(k_range, silhouettes, 's--', color=color2, lw=1.8, markersize=5.5, label='Silhouette')
ax1_twin.set_ylabel('Silhouette Score', color=color2, fontsize=9, fontweight='bold')
ax1.set_title('K-Means Optimization (Elbow & Silhouette)', fontsize=10.5, fontweight='bold', pad=10)

# 6b. Hierarchical Dendrogram
sample_idx = np.random.RandomState(42).choice(len(X_unsup_scaled), size=120, replace=False)
link_matrix = linkage(X_unsup_scaled[sample_idx], method='ward')
dendrogram(link_matrix, truncate_mode='lastp', p=25, leaf_rotation=90., leaf_font_size=8., show_contracted=True, ax=ax2)
ax2.set_title('Hierarchical Clustering (Ward Linkage)', fontsize=10.5, fontweight='bold', pad=10)
ax2.set_xlabel('Subtree Cluster Size', fontsize=9, fontweight='bold')
ax2.set_ylabel('Linkage Distance', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/06_infrastructure_clustering.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# PLOT 7: Task 7 - Dimensionality Reduction (PCA & State Landscape)
# -------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_unsup_scaled)
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300)

# 7a. PCA 2D Scatter
cluster_names = {0: 'Steady-State Normal', 1: 'Elevated Promo/Sale', 2: 'Critical Flash Surge'}
df['state_name'] = df['kmeans_cluster'].map(cluster_names)
palette_clusters = {'Steady-State Normal': '#2ecc71', 'Elevated Promo/Sale': '#3498db', 'Critical Flash Surge': '#e74c3c'}
sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='state_name', palette=palette_clusters,
                alpha=0.65, s=25, edgecolor='none', ax=ax1)
ax1.set_title(f'PCA 2D Operating States ({sum(pca.explained_variance_ratio_):.1%} Variance)', fontsize=10.5, fontweight='bold', pad=10)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=8.5, fontweight='bold')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=8.5, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=7.5)

# 7b. Throughput vs Latency & Headroom
scatter = ax2.scatter(df['requests_per_min'], df['response_time_ms'], c=df['capacity_headroom'],
                      cmap='coolwarm_r', alpha=0.65, s=25, edgecolors='none')
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Headroom Buffer (RPM)', fontsize=8.5, fontweight='bold')
ax2.axhline(200, color='red', linestyle='--', linewidth=1.2, label='200ms SLA Limit')
ax2.set_title('Throughput vs Latency (SLA Stress Zone)', fontsize=10.5, fontweight='bold', pad=10)
ax2.set_xlabel('Requests / Min (RPM)', fontsize=8.5, fontweight='bold')
ax2.set_ylabel('Response Latency (ms)', fontsize=8.5, fontweight='bold')
ax2.legend(loc='upper left', frameon=True, fontsize=7.5)

plt.tight_layout()
plt.savefig(f'{output_dir}/07_pca_and_capacity_states.png', bbox_inches='tight')
plt.close()

print('All 7 clean, neat 2-panel visualizations successfully generated!')
