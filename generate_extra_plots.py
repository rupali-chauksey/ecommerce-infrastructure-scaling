import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = 'd:/Ass 1/outputs'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('d:/Ass 1/data/ecommerce_infrastructure_scaling.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# 1. EDA Distributions & Targets (Task 1)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

# 1a. Sale Event Distribution
sale_counts = df['sale_event'].value_counts()
colors_sale = ['#2ca02c', '#1f77b4', '#d62728']
axes[0, 0].pie(sale_counts, labels=sale_counts.index, autopct='%1.1f%%', startangle=140, colors=colors_sale, explode=(0.02, 0.02, 0.08), shadow=True)
axes[0, 0].set_title('Sale Event Distribution (Normal vs Sale vs Flash Sale)', fontsize=12, fontweight='bold')

# 1b. Regional Distribution
region_counts = df['region'].fillna('Missing').value_counts()
sns.barplot(x=region_counts.index, y=region_counts.values, ax=axes[0, 1], palette='Blues_r', edgecolor='black', linewidth=0.5)
axes[0, 1].set_title('Regional Telemetry Distribution (North, South, West)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Snapshot Count', fontweight='bold')
for i, v in enumerate(region_counts.values):
    axes[0, 1].text(i, v + 20, str(v), ha='center', fontweight='bold')

# 1c. Target 1 Distribution: next_15min_requests
sns.histplot(df['next_15min_requests'], kde=True, ax=axes[1, 0], color='#ff7f0e', bins=30, edgecolor='black', alpha=0.6)
axes[1, 0].axvline(df['next_15min_requests'].mean(), color='red', linestyle='--', label=f'Mean: {df["next_15min_requests"].mean():.0f} RPM')
axes[1, 0].axvline(df['next_15min_requests'].median(), color='green', linestyle=':', label=f'Median: {df["next_15min_requests"].median():.0f} RPM')
axes[1, 0].set_title('Target 1 Distribution: next_15min_requests (RPM)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Requests Per Minute 15-min Ahead', fontweight='bold')
axes[1, 0].legend(frameon=True)

# 1d. Target 2 Distribution: scale_up_required
scale_counts = df['scale_up_required'].value_counts()
sns.barplot(x=['No Scale-Up (0)', 'Scale-Up Required (1)'], y=scale_counts.values, ax=axes[1, 1], palette=['#3498db', '#e74c3c'], edgecolor='black', linewidth=0.5)
axes[1, 1].set_title('Target 2 Distribution: scale_up_required (Imbalance Flag)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Snapshot Count', fontweight='bold')
for i, v in enumerate(scale_counts.values):
    axes[1, 1].text(i, v + 40, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')

plt.suptitle('Task 1: Telemetry Data Inspection, Distributions & Target Distributions', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/00_eda_distributions_and_targets.png')
plt.close()

# 2. Chronological Split & Metric Profiling (Task 2)
fig, axes = plt.subplots(2, 1, figsize=(15, 8), dpi=300)
split_idx = int(len(df) * 0.8)

# Timeline split
train_times = df['timestamp'].iloc[:split_idx]
test_times = df['timestamp'].iloc[split_idx:]
train_reqs = df['requests_per_min'].iloc[:split_idx]
test_reqs = df['requests_per_min'].iloc[split_idx:]

axes[0].plot(train_times, train_reqs, color='#1f77b4', label=f'Training Set (First 80% = {len(train_times)} snapshots)', lw=1.2, alpha=0.85)
axes[0].plot(test_times, test_reqs, color='#d62728', label=f'Evaluation Set (Final 20% = {len(test_times)} snapshots)', lw=1.4, alpha=0.9)
axes[0].axvline(df['timestamp'].iloc[split_idx], color='black', linestyle='--', label='Chronological Split Boundary (July 17, 19:00)')
axes[0].set_title('Task 2: Time-Aware Chronological Split (No Shuffling / Temporal Leakage Prevention)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Requests Per Minute (RPM)', fontweight='bold')
axes[0].legend(loc='upper left', frameon=True)

# Missing Values Summary Bar Chart
missing_counts = df.isnull().sum()
missing_filtered = missing_counts[missing_counts > 0]
axes[1].barh(missing_filtered.index, missing_filtered.values, color='#9b59b6', edgecolor='black', linewidth=0.5)
axes[1].set_title('Task 2: Missing Telemetry Values (Imputed via Median & Mode on Train Split)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Missing Observations', fontweight='bold')
for i, v in enumerate(missing_filtered.values):
    axes[1].text(v + 0.3, i, f'{v} ({v/len(df)*100:.2f}%)', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/00_task2_outliers_and_temporal_split.png')
plt.close()

print('Extra Task 1 & 2 visualizations generated successfully!')
