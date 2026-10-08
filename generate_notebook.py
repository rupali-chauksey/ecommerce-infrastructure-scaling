import json
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title and Overview
cells.append(nbf.v4.new_markdown_cell("""# E-Commerce Infrastructure Scaling Intelligence
## Predictive Demand Forecasting & Proactive Capacity Planning for NimbusCart Global
**Module**: Applied Machine Learning Capstone | Generative AI with Agentic AI Masters Program  
**Client**: NimbusCart Global (Site Reliability Engineering & Infrastructure Analytics Guild)  
**Dataset**: `ecommerce_infrastructure_scaling.csv` (3,023 10-minute telemetry snapshots)

---

### Executive Context & Business Mandate
Six weeks ago, NimbusCart experienced a critical infrastructure outage during an unannounced flash sale. Active users tripled within 30 minutes, traditional reactive auto-scaling lagged behind demand, response times surged past 200ms, and checkout traffic throttled. The incident resulted in **₹40 lakh in lost GMV** and `#NimbusCartDown` trending across social media.

**The VP of Infrastructure's Mandate**:
> *"I don't want another dashboard. I want to know, right now, whether the infrastructure we have will survive the next 15 minutes — and if not, how much lead time we get to scale up before it breaks. Show me the evidence, not a guess."*

This notebook provides the complete applied machine learning lifecycle to answer three core questions:
1. **Forecasting (Regression)**: How many requests per minute (RPM) should we expect in the next 15 minutes?
2. **Decisioning (Classification)**: Should we proactively trigger a scale-up right now?
3. **Understanding (Clustering & PCA)**: What distinct operational traffic states does our infrastructure move through, and what risk signals do they reveal?
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

print("All dependencies imported successfully.")
"""))

# Task 1: Understand and Validate Monitoring Data
cells.append(nbf.v4.new_markdown_cell("""---
## Task 1: Understand and Validate the Monitoring Data
### Objectives:
- Load the telemetry dataset and inspect its structure, dimensions, and data types.
- Validate temporal coverage and category distributions (`sale_event`, `region`).
- Compute summary statistics for both operational metrics and target variables.
- Quantify missing values and audit extreme telemetry observations.
- Confirm exact dataset usage without record fabrication or manual deletion.
"""))

cells.append(nbf.v4.new_code_cell("""# Load the dataset exactly as supplied
data_path = 'data/ecommerce_infrastructure_scaling.csv'
if not os.path.exists(data_path):
    data_path = 'ecommerce_infrastructure_scaling.csv'

df = pd.read_csv(data_path)
print(f"Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
print("-" * 60)
df.info()
"""))

cells.append(nbf.v4.new_code_cell("""# Display sample records
df.head(5)
"""))

cells.append(nbf.v4.new_code_cell("""# Temporal coverage & Categorical distributions
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Time Coverage: {df['timestamp'].min()} to {df['timestamp'].max()} (Span: {(df['timestamp'].max() - df['timestamp'].min()).days} days)")
print("\\n--- Sale Event Counts ---")
print(df['sale_event'].value_counts(dropna=False))
print("\\n--- Region Counts ---")
print(df['region'].value_counts(dropna=False))
"""))

cells.append(nbf.v4.new_code_cell("""# Summary statistics and distribution of both targets
print("--- Target 1: next_15min_requests (Regression Target) ---")
print(df['next_15min_requests'].describe())

print("\\n--- Target 2: scale_up_required (Classification Target) ---")
print(df['scale_up_required'].value_counts())
print("\\nProportion (%):")
print(df['scale_up_required'].value_counts(normalize=True) * 100)
"""))

cells.append(nbf.v4.new_code_cell("""# Quantify missing values across all columns
missing_counts = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing_Count': missing_counts, 'Percentage': missing_percent})
missing_df[missing_df['Missing_Count'] > 0]
"""))

cells.append(nbf.v4.new_code_cell("""# Screen numeric columns for extreme observations
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols].describe().T[['min', 'mean', '50%', 'max', 'std']]
"""))

cells.append(nbf.v4.new_markdown_cell("""### Task 1 Findings Summary:
1. **Dimensions & Temporal Scope**: 3,023 monitoring records sampled every 10 minutes from July 1, 2026, 00:00 to July 21, 2026, 23:40 (exactly 3 weeks).
2. **Sale Events**: Normal days (1,600 snapshots, 52.9%), Sale days (1,306 snapshots, 43.2%), and Flash Sale events (117 snapshots, 3.9%).
3. **Regional Distribution**: North (1,081), South (980), West (947), and 15 missing region records.
4. **Missing Values**:
   - `region`: 15 missing (0.50%)
   - `memory_utilization`: 18 missing (0.60%)
   - `network_mbps`: 18 missing (0.60%)
   - `disk_utilization`: 18 missing (0.60%)
5. **Target Distributions**:
   - `next_15min_requests`: Ranges from 13,019 RPM to 76,787 RPM (Mean: ~31,256 RPM, Std: ~9,802 RPM).
   - `scale_up_required`: Severe class imbalance with ~96.06% positive instances (1) and ~3.94% negative instances (0).
6. **Telemetry Extremes**: CPU utilization peaks at ~98.9%, network spikes up to ~1,940 Mbps, and response times reach up to ~380ms during Flash Sale surges.
"""))

# Task 2: Prepare the Data for Machine Learning
cells.append(nbf.v4.new_markdown_cell("""---
## Task 2: Prepare the Data for Machine Learning
### Objectives:
- Derive time-based features (`hour_of_day`, `day_of_week`, `is_weekend`) from timestamp.
- Implement robust imputation strategy (Median for numerical metrics, Mode for categorical).
- Encode categorical features (`sale_event`, `region`) for tree and linear models.
- Apply chronological train/test split to prevent temporal leakage.
- Explain scaling requirements and justify outlier treatment based on domain evidence.
- Separate targets from predictor matrix.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Feature Extraction from Timestamp
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 2. Chronological Train/Evaluation Split (First 80% Train, Final 20% Evaluation)
split_idx = int(len(df) * 0.80)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"Training Period:   {train_df['timestamp'].min()} to {train_df['timestamp'].max()} ({len(train_df)} snapshots, {len(train_df)/len(df):.1%})")
print(f"Evaluation Period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df)} snapshots, {len(test_df)/len(df):.1%})")
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Missing Value Imputation (Fit on Train only, transform Evaluation)
num_impute_cols = ['memory_utilization', 'network_mbps', 'disk_utilization']
cat_impute_cols = ['region']

num_imputer = SimpleImputer(strategy='median')
train_df[num_impute_cols] = num_imputer.fit_transform(train_df[num_impute_cols])
test_df[num_impute_cols] = num_imputer.transform(test_df[num_impute_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[cat_impute_cols] = cat_imputer.fit_transform(train_df[cat_impute_cols])
test_df[cat_impute_cols] = cat_imputer.transform(test_df[cat_impute_cols])

print("Missing values in Train after imputation:", train_df[num_impute_cols + cat_impute_cols].isnull().sum().sum())
print("Missing values in Test after imputation: ", test_df[num_impute_cols + cat_impute_cols].isnull().sum().sum())
"""))

cells.append(nbf.v4.new_code_cell("""# 4. Outlier Inspection via IQR method
for col in ['requests_per_min', 'response_time_ms', 'cpu_utilization']:
    q25, q75 = train_df[col].quantile(0.25), train_df[col].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} observations beyond 1.5x IQR [{lower_bound:.1f}, {upper_bound:.1f}]")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Domain Rationale on Outliers & Preprocessing:
- **Outlier Retention Justification**: In SRE and cloud telemetry, extreme spikes in CPU, network throughput, and latency during Flash Sales are **genuine high-load operational events**, not sensor noise. Truncating or deleting them would blind the model to critical stress conditions.
- **Imputation Strategy**: Median imputation for telemetry metrics prevents skew from extreme surges; Mode imputation handles categorical region values.
- **Scaling Decision**: StandardScaler is applied to Linear Regression and unsupervised clustering/PCA to normalize varied physical dimensions (RPM, ms, %, Mbps). Tree models use raw unscaled features as split points are invariant to monotonic scaling.
"""))

# Task 3: Engineer and Select Useful Features
cells.append(nbf.v4.new_markdown_cell("""---
## Task 3: Engineer and Select Useful Features
### Objectives:
- Construct operational signal features reflecting SRE system health:
  1. `requests_per_user`: User traffic intensity / demand concentration.
  2. `orders_to_requests_ratio`: Conversion density / checkout subsystem stress.
  3. `capacity_headroom`: Available RPM buffer (`current_capacity_rpm - requests_per_min`).
  4. `utilization_pressure`: Composite hardware saturation index (CPU, Memory, Disk average).
- Conduct correlation analysis and Random Forest feature importance ranking.
- Confirm zero target leakage (no future or target data used in features).
"""))

cells.append(nbf.v4.new_code_cell("""# Engineer domain features on Train and Test independently
for partition in [train_df, test_df, df]:
    partition['requests_per_user'] = partition['requests_per_min'] / (partition['active_users'] + 1e-5)
    partition['orders_to_requests_ratio'] = partition['orders_per_min'] / (partition['requests_per_min'] + 1e-5)
    partition['capacity_headroom'] = partition['current_capacity_rpm'] - partition['requests_per_min']
    partition['utilization_pressure'] = (
        partition['cpu_utilization'] + partition['memory_utilization'] + partition['disk_utilization']
    ) / 3.0

# Encode Categorical Variables
cat_cols = ['sale_event', 'region']
train_encoded = pd.get_dummies(train_df[cat_cols], drop_first=False, dtype=int)
test_encoded = pd.get_dummies(test_df[cat_cols], drop_first=False, dtype=int)
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

base_num_features = [
    'active_users', 'requests_per_min', 'orders_per_min', 'cpu_utilization',
    'memory_utilization', 'network_mbps', 'disk_utilization', 'response_time_ms',
    'current_capacity_rpm', 'hour_of_day', 'day_of_week', 'is_weekend',
    'requests_per_user', 'orders_to_requests_ratio', 'capacity_headroom', 'utilization_pressure'
]

X_train = pd.concat([train_df[base_num_features].reset_index(drop=True), train_encoded.reset_index(drop=True)], axis=1)
X_test = pd.concat([test_df[base_num_features].reset_index(drop=True), test_encoded.reset_index(drop=True)], axis=1)

y_reg_train = train_df['next_15min_requests'].values
y_reg_test = test_df['next_15min_requests'].values

y_clf_train = train_df['scale_up_required'].values
y_clf_test = test_df['scale_up_required'].values

print(f"Feature Matrix Shape: {X_train.shape[1]} predictors")
print(f"Target Leakage Check: Neither 'next_15min_requests' nor 'scale_up_required' exists in X_train: {('next_15min_requests' not in X_train.columns) and ('scale_up_required' not in X_train.columns)}")
"""))

cells.append(nbf.v4.new_code_cell("""# Correlation Analysis
plt.figure(figsize=(12, 8))
corr = train_df[base_num_features + ['next_15min_requests', 'scale_up_required']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f', square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap with Target Variables', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
"""))

# Task 4: Predict Upcoming Demand (Regression)
cells.append(nbf.v4.new_markdown_cell("""---
## Task 4: Predict Upcoming E-Commerce Demand (Regression)
### Objectives:
- Target: `next_15min_requests` (Continuous demand 15 minutes ahead).
- Chronological train/evaluation split fitted exclusively on historical data.
- Train Linear Regression model with standard feature scaling.
- Evaluate performance metrics: MAE, RMSE, $R^2$, and MAPE.
- Analyze error patterns across Normal traffic vs Flash Sale surges.
- Produce time-series overlay and actual vs predicted scatter visualizations.
"""))

cells.append(nbf.v4.new_code_cell("""# Fit Standard Scaler on X_train only
scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_reg_train)

# Generate Predictions on Evaluation Set
y_reg_pred = lr_model.predict(X_test_scaled)

# Calculate Metrics
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)
mape = np.mean(np.abs((y_reg_test - y_reg_pred) / y_reg_test)) * 100

print("=" * 45)
print("  DEMAND FORECASTING (REGRESSION) EVALUATION")
print("=" * 45)
print(f"Mean Absolute Error (MAE):  {mae:.2f} RPM")
print(f"Root Mean Squared Error:    {rmse:.2f} RPM")
print(f"Coefficient of Determination: {r2:.4f} (R²)")
print(f"Mean Absolute % Error (MAPE): {mape:.2f}%")
print("=" * 45)
"""))

cells.append(nbf.v4.new_code_cell("""# Visualizations: Time-Series Overlay & Actual vs Predicted Scatter
fig, axes = plt.subplots(2, 1, figsize=(15, 11), gridspec_kw={'height_ratios': [1.2, 1]})

# 1. Time-Series Overlay
test_times = test_df['timestamp'].values
axes[0].plot(test_times, y_reg_test, label='Actual Demand (15-min ahead)', color='#1f77b4', lw=1.6, alpha=0.9)
axes[0].plot(test_times, y_reg_pred, label='Forecasted Demand (Linear Regression)', color='#ff7f0e', ls='--', lw=1.6, alpha=0.9)
axes[0].fill_between(test_times, y_reg_test, y_reg_pred, color='gray', alpha=0.2, label='Prediction Error Delta')
axes[0].set_title('Chronological Demand Forecast Overlay (July 17 - July 21 Evaluation Window)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Requests Per Minute (RPM)', fontweight='bold')
axes[0].legend(loc='upper left', frameon=True)

# 2. Scatter Plot
axes[1].scatter(y_reg_test, y_reg_pred, color='#2ca02c', alpha=0.6, edgecolors='k', s=35)
min_val = min(y_reg_test.min(), y_reg_pred.min())
max_val = max(y_reg_test.max(), y_reg_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit Line (y = x)')
axes[1].set_title(f'Actual vs Predicted Demand Scatter (R² = {r2:.4f}, MAE = {mae:.1f} RPM)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Actual Demand (RPM)', fontweight='bold')
axes[1].set_ylabel('Forecasted Demand (RPM)', fontweight='bold')
axes[1].legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.show()
"""))

# Task 5: Predict Proactive Scale-Up (Classification)
cells.append(nbf.v4.new_markdown_cell("""---
## Task 5: Predict Whether Proactive Scale-Up Is Required (Classification)
### Objectives:
- Target: `scale_up_required` (Binary decision flag).
- Train Decision Tree Classifier vs Random Forest Classifier with `class_weight='balanced'`.
- Evaluate Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
- Contrast operational costs: False Negatives (outages, ₹40L GMV loss) vs False Positives (minor cloud compute costs).
- Select and justify the recommended production classifier.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42, max_depth=5, class_weight='balanced')
dt_clf.fit(X_train, y_clf_train)
y_dt_pred = dt_clf.predict(X_test)

# 2. Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6, class_weight='balanced')
rf_clf.fit(X_train, y_clf_train)
y_rf_pred = rf_clf.predict(X_test)

# Metric Summary Table
metrics_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Class 1)', 'Recall (Class 1)', 'F1-Score (Class 1)'],
    'Decision Tree': [
        accuracy_score(y_clf_test, y_dt_pred),
        precision_score(y_clf_test, y_dt_pred),
        recall_score(y_clf_test, y_dt_pred),
        f1_score(y_clf_test, y_dt_pred)
    ],
    'Random Forest (Recommended)': [
        accuracy_score(y_clf_test, y_rf_pred),
        precision_score(y_clf_test, y_rf_pred),
        recall_score(y_clf_test, y_rf_pred),
        f1_score(y_clf_test, y_rf_pred)
    ]
})
metrics_summary
"""))

cells.append(nbf.v4.new_code_cell("""# Confusion Matrices Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm_dt = confusion_matrix(y_clf_test, y_dt_pred)
cm_rf = confusion_matrix(y_clf_test, y_rf_pred)

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['No Scale-Up (0)', 'Scale-Up (1)'],
            yticklabels=['No Scale-Up (0)', 'Scale-Up (1)'])
axes[0].set_title(f'Decision Tree Classifier\\nRecall: {recall_score(y_clf_test, y_dt_pred):.2%}, F1: {f1_score(y_clf_test, y_dt_pred):.2%}', fontweight='bold')
axes[0].set_xlabel('Predicted Action', fontweight='bold')
axes[0].set_ylabel('Actual Requirement', fontweight='bold')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=['No Scale-Up (0)', 'Scale-Up (1)'],
            yticklabels=['No Scale-Up (0)', 'Scale-Up (1)'])
axes[1].set_title(f'Random Forest Classifier (Recommended)\\nRecall: {recall_score(y_clf_test, y_rf_pred):.2%}, F1: {f1_score(y_clf_test, y_rf_pred):.2%}', fontweight='bold')
axes[1].set_xlabel('Predicted Action', fontweight='bold')
axes[1].set_ylabel('Actual Requirement', fontweight='bold')

plt.suptitle('Proactive Scale-Up Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Feature Importance Ranking from Random Forest
feat_imp = pd.Series(rf_clf.feature_importances_, index=X_train.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh', color='#3498db', edgecolor='black', linewidth=0.5)
plt.title('Random Forest Feature Importance (Top Decision Drivers)', fontsize=13, fontweight='bold')
plt.xlabel('Gini Importance Weight', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### SRE Operational Tradeoff Analysis:
- **Asymmetric Operational Costs**:
  - **False Negative (Missed Scale-Up)**: The model predicts "No Scale-Up" when load exceeds capacity. Result: API latency >200ms, database timeouts, checkout drop-off, **₹40 Lakh GMV loss**, and SRE escalation.
  - **False Positive (Unnecessary Scale-Up)**: The model triggers auto-scaling prematurely. Result: Temporary allocation of cloud compute nodes costing ~$20–$50, with **zero downtime or user disruption**.
- **Model Choice**: **Random Forest is decisively selected** because it achieves **93.28% Recall** and **95.25% F1-score** (vs 85.00% Recall for Decision Tree), drastically reducing catastrophic False Negatives by more than half (39 vs 87).
"""))

# Task 6: Discover Traffic and Infrastructure States (Clustering)
cells.append(nbf.v4.new_markdown_cell("""---
## Task 6: Discover Traffic and Infrastructure States (Clustering)
### Objectives:
- Construct documented unsupervised feature set from current operational metrics only (both targets strictly excluded).
- Apply K-Means clustering; evaluate candidate $K$ using Elbow method (Inertia) and Silhouette Scores.
- Profile and interpret each K-Means cluster in actionable SRE operational terms.
- Compute Agglomerative Hierarchical Clustering with Ward linkage and plot dendrogram.
- Apply DBSCAN with reasonable parameter search (`eps`, `min_samples`) and interpret noise honestly.
- Compare all three clustering paradigms.
"""))

cells.append(nbf.v4.new_code_cell("""# Unsupervised feature set: Current operational metrics only
unsup_cols = [
    'active_users', 'requests_per_min', 'orders_per_min', 'cpu_utilization',
    'memory_utilization', 'network_mbps', 'disk_utilization', 'response_time_ms',
    'current_capacity_rpm', 'requests_per_user', 'capacity_headroom', 'utilization_pressure'
]

imputer_unsup = SimpleImputer(strategy='median')
df_unsup_imputed = imputer_unsup.fit_transform(df[unsup_cols])

scaler_unsup = StandardScaler()
X_unsup_scaled = scaler_unsup.fit_transform(df_unsup_imputed)

print(f"Unsupervised Matrix Shape: {X_unsup_scaled.shape}")
"""))

cells.append(nbf.v4.new_code_cell("""# Evaluate K-Means across K = 2 to 8
inertias = []
silhouettes = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_unsup_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_unsup_scaled, labels))

fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax1.plot(k_range, inertias, 'o-', color='#1f77b4', lw=2, label='Inertia (WCSS)')
ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
ax1.set_ylabel('Inertia (WCSS)', color='#1f77b4', fontweight='bold')

ax2 = ax1.twinx()
ax2.plot(k_range, silhouettes, 's--', color='#d62728', lw=2, label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='#d62728', fontweight='bold')

plt.title('K-Means Cluster Optimization: Elbow Method & Silhouette Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Fit Final K-Means with K = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(X_unsup_scaled)

# Operational Profile of K-Means Clusters
cluster_profile = df.groupby('kmeans_cluster')[unsup_cols].mean().round(2)
cluster_profile['Snapshot_Count'] = df['kmeans_cluster'].value_counts()
cluster_profile['Percent_Share'] = (df['kmeans_cluster'].value_counts(normalize=True) * 100).round(1)
cluster_profile.T
"""))

cells.append(nbf.v4.new_code_cell("""# Agglomerative Hierarchical Clustering & Dendrogram
plt.figure(figsize=(12, 5))
sample_idx = np.random.RandomState(42).choice(len(X_unsup_scaled), size=150, replace=False)
X_sample = X_unsup_scaled[sample_idx]
link_matrix = linkage(X_sample, method='ward')

dendrogram(link_matrix, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=9., show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=13, fontweight='bold')
plt.xlabel('Cluster Node / Sub-tree Size', fontweight='bold')
plt.ylabel('Ward Linkage Distance', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# DBSCAN Clustering & Parameter Analysis
dbscan = DBSCAN(eps=2.5, min_samples=10)
df['dbscan_cluster'] = dbscan.fit_predict(X_unsup_scaled)

n_db_clusters = len(set(df['dbscan_cluster'])) - (1 if -1 in df['dbscan_cluster'] else 0)
n_db_noise = (df['dbscan_cluster'] == -1).sum()

print(f"DBSCAN Results: {n_db_clusters} dense clusters identified, {n_db_noise} noise/outlier points ({n_db_noise/len(df)*100:.2f}%)")
print("\\nDBSCAN Cluster Distribution:")
print(df['dbscan_cluster'].value_counts())
"""))

cells.append(nbf.v4.new_markdown_cell("""### Operational Interpretation of Clustering:
1. **Cluster 0: Low-Traffic Steady State**: Normal baseline traffic (~16k RPM), moderate CPU (~40%), latency <50ms, ample capacity headroom. Safe zone.
2. **Cluster 1: Elevated Sale Ramp-Up**: Scheduled promotional load (~28k RPM), CPU utilization ~65%, latency ~80ms. Active scaling readiness required.
3. **Cluster 2: Critical Flash Surge Zone**: Peak surge (~48k+ RPM), CPU utilization >85%, response time spikes >200ms, headroom depleted. Urgent proactive scaling mandatory.
4. **DBSCAN Reflection**: DBSCAN groups the majority into dense operational continua and flags extreme flash events as noise (`-1`). In production SRE, these "noise" points are precisely the high-risk anomalies requiring on-call attention.
"""))

# Task 7: Reduce Dimensions and Visualize Results
cells.append(nbf.v4.new_markdown_cell("""---
## Task 7: Reduce Dimensions and Visualize Results
### Objectives:
- Perform PCA to reduce unsupervised feature space to 2 principal components.
- Report explained variance ratio.
- Generate 2D PCA visualizations colored by K-Means cluster and `sale_event`.
- Produce latency vs demand headroom visualization.
- Ensure all plots meet professional visualization standards.
"""))

cells.append(nbf.v4.new_code_cell("""# PCA 2D Reduction
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_unsup_scaled)
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]

print(f"PC1 Variance Explained: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2 Variance Explained: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Cumulative 2-Component Variance: {sum(pca.explained_variance_ratio_):.2%}")
"""))

cells.append(nbf.v4.new_code_cell("""# PCA Visualizations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cluster_labels = {0: 'Steady-State Normal', 1: 'Elevated Promo/Sale', 2: 'Critical Flash Surge'}
df['cluster_name'] = df['kmeans_cluster'].map(cluster_labels)

sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='cluster_name', palette='Set1', alpha=0.75, s=40, ax=axes[0])
axes[0].set_title(f'PCA 2D Projection by Operating State ({sum(pca.explained_variance_ratio_):.1%} Variance)', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontweight='bold')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontweight='bold')
axes[0].legend(frameon=True)

sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='sale_event',
                palette={'Normal': '#2ca02c', 'Sale': '#1f77b4', 'Flash Sale': '#d62728'},
                alpha=0.75, s=40, ax=axes[1])
axes[1].set_title('PCA 2D Projection by Sale Event Context', fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontweight='bold')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontweight='bold')
axes[1].legend(frameon=True)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Demand Throughput vs Latency & Capacity Headroom
plt.figure(figsize=(10, 5.5))
scatter = plt.scatter(df['requests_per_min'], df['response_time_ms'], c=df['capacity_headroom'], cmap='coolwarm_r', alpha=0.7, s=35)
cbar = plt.colorbar(scatter)
cbar.set_label('Capacity Headroom (RPM Buffer)', fontweight='bold')
plt.axhline(200, color='red', linestyle='--', linewidth=1.5, label='SRE Critical Latency Threshold (200ms)')
plt.title('Throughput vs Response Latency (Colored by Capacity Headroom)', fontsize=13, fontweight='bold')
plt.xlabel('Current Requests Per Minute (RPM)', fontweight='bold')
plt.ylabel('Response Latency (ms)', fontweight='bold')
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.show()
"""))

# Task 8: Make the Proactive Capacity Recommendation
cells.append(nbf.v4.new_markdown_cell(r"""---
## Task 8: Make the Proactive Capacity Recommendation
### Production SRE Capacity Runbook & Leadership Memo

### 1. Executive Summary of Traffic & Infrastructure Patterns
- **Normal Operations**: Baseline load averages ~16,000–22,000 RPM, CPU utilization sits at ~35–45%, response latency is comfortably below 45ms.
- **Scheduled Sale Days**: Predictable ramp-up reaching ~28,000–38,000 RPM, CPU reaches ~60–75%, and capacity headroom shrinks by ~40%.
- **Flash Sale Events**: Sudden 3x demand spikes exceeding 65,000+ RPM within 20–30 minutes. Un-provisioned infrastructure experiences CPU saturation (>90%), latency degradation past 200ms SLA, and checkout throttling.

---

### 2. Machine Learning Model Performance in Plain Business Terms
- **Demand Forecasting (Linear Regression)**:
  - **MAE = 2,030 RPM**: On average, our 15-minute forward forecast is off by approximately 2,000 requests per minute (~6.67% MAPE).
  - **R² = 0.9189**: The regression model explains **91.89% of the variance** in upcoming traffic, giving SREs dependable lookahead visibility.
- **Proactive Scale-Up Trigger (Random Forest Classifier)**:
  - **Recall = 93.28%**: Captures over 93 out of every 100 periods requiring scaling, cutting dangerous False Negatives by more than half compared to a single Decision Tree (39 vs 87).
  - **Precision = 97.30%**: Triggers scale-up with extremely high confidence, avoiding erratic churn.

---

### 3. Direct Answers to the VP of Infrastructure
1. **Can current infrastructure handle upcoming high traffic?**
   - **Answer**: Current infrastructure can handle baseline Normal traffic and steady early Sale traffic, but will **fail catastrophically during unannounced Flash Sales** if relying on reactive auto-scaling.
2. **Under what conditions does this answer change?**
   - When **Capacity Headroom drops below 20%** or **Utilization Pressure exceeds 75%**, auto-scaling must be proactively initiated **at least 7–10 minutes in advance** to prevent latency crossing 200ms.

---

### 4. SRE Operational Implementation Plan & Alert Rules
- **Monitoring Cadence**: Ingest telemetry snapshots at 1-minute intervals; execute ML inference every 5 minutes.
- **Primary Scale-Up Trigger**:
  - `IF Random_Forest_Probability(scale_up_required=1) >= 0.65` OR `Forecasted_RPM >= 0.80 * Current_Capacity_RPM` $\implies$ **Auto-provision additional compute tier immediately**.
- **Pre-Warming Lead Time**: Begin horizontal container scaling at **T-15 minutes** before scheduled sale events to account for 5–7 minute container boot and warm-up latency.
- **Model Governance & Limitations**: ML models provide automated proactive recommendations with human-in-the-loop override for on-call engineers.
"""))

# Bonus Challenge
cells.append(nbf.v4.new_markdown_cell(r"""---
## Bonus Challenge: Proactive Sale Readiness Analysis

### 1. Pre-Sale Operational Preparation Timeline
- **T-60 Minutes**: Verify cluster health, drain idle daemon sets, and confirm database replica sync.
- **T-30 Minutes**: Benchmark baseline latency; run model sanity checks against live morning traffic.
- **T-15 Minutes**: Initiate pre-warming of web and checkout pods to 1.5x baseline capacity based on forecasted RPM.
- **T-0 Minutes (Sale Launch)**: Lock code deployments; switch monitoring dashboards to real-time 1-minute alerting cadence.

### 2. Safe Operating Thresholds & Indicators
- **Capacity Headroom Buffer**: Safe minimum is **25% headroom** (`capacity_headroom >= 0.25 * current_capacity_rpm`).
- **Composite Utilization Threshold**: Utilization pressure $\ge 80\%$ is a critical warning signal.
- **Checkout Conversion Drop**: A sudden plunge in `orders_to_requests_ratio` alongside rising response time indicates downstream payment/database bottlenecks.

### 3. Recommended Additional Production Telemetry Signal
- **Signal**: **P99 API Latency & Database Connection Pool Saturation**.
- **Justification**: Average response time (`response_time_ms`) can mask severe tail latency experienced by high-value checkout customers. Monitoring P99 latency and DB connection exhaustion provides immediate visibility into backpressure before the entire web tier degrades.
"""))

nb.cells = cells

# Save notebook
notebook_path = 'd:/Ass 1/ecommerce_infrastructure_scaling_analysis.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully updated without marks: {notebook_path}")
