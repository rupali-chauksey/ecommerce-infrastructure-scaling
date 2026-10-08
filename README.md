# E-Commerce Infrastructure Scaling Intelligence
## Predictive Demand Forecasting & Proactive Capacity Planning for NimbusCart Global

**Role/Persona**: Applied Machine Learning Engineer / SRE Infrastructure Analytics  
**Module**: Applied Machine Learning Capstone Deliverable | Generative AI with Agentic AI Masters Program  
**Client**: NimbusCart Global (Site Reliability Engineering & Cloud Infrastructure Guild)  
**Deliverables**:
- Jupyter Notebook: [`ecommerce_infrastructure_scaling_analysis.ipynb`](file:///d:/Ass%201/ecommerce_infrastructure_scaling_analysis.ipynb)
- SRE Executive Summary & Documentation: [`README.md`](file:///d:/Ass%201/README.md)
- Production Visualizations: [`outputs/`](file:///d:/Ass%201/outputs/)

---

## 1. Executive Summary & Incident Context

Six weeks ago, NimbusCart experienced a severe infrastructure failure during an unscheduled flash sale. Active users tripled within 30 minutes, traditional threshold-based reactive auto-scaling lagged behind demand spikes, API response latency surged past the critical 200ms SLA, and the platform throttled checkout requests. The outage resulted in **₹40 Lakh in lost Gross Merchandise Value (GMV)** and `#NimbusCartDown` trending on social media.

### The Mandate from the VP of Infrastructure:
> *"I don't want another dashboard. I want to know, right now, whether the infrastructure we have will survive the next 15 minutes — and if not, how much lead time we get to scale up before it breaks. Show me the evidence, not a guess."*

This project delivers an end-to-end Machine Learning intelligence engine that transitions NimbusCart from **reactive fire-fighting** to **proactive, evidence-based automated capacity scaling**.

---

## 2. Project Architecture & Applied ML Workflow

```
                                  [ Production Telemetry Stream ]
                             3,023 snapshots | 10-min cadence | July 1–21
                                                  │
                ┌─────────────────────────────────┴─────────────────────────────────┐
                ▼                                                                   ▼
    [ Supervised Intelligence ]                                         [ Unsupervised Discovery ]
    1. Demand Forecasting (Regression)                                  1. K-Means Clustering (K=3)
       • Linear Regression                                                 • Steady-State Normal (52.9%)
       • MAE: 2,030 RPM | R²: 0.9189 | MAPE: 6.67%                         • Elevated Sale Ramp-Up (43.2%)
    2. Proactive Decisioning (Classification)                              • Critical Flash Surge (3.9%)
       • Random Forest vs Decision Tree                                 2. Hierarchical Dendrogram (Ward)
       • RF Recall: 93.28% | F1: 0.9525 | Precision: 97.30%             3. DBSCAN & PCA 2D Projections
                └─────────────────────────────────┬─────────────────────────────────┘
                                                  ▼
                               [ SRE 2 AM Operational Runbook ]
                     Pre-warming: T-15 min | Headroom Alert: <20% | Scale Trigger: Prob >= 0.65
```

---

## 3. Dataset Architecture & Validation (Task 1)

The dataset (`data/ecommerce_infrastructure_scaling.csv`) contains 3 weeks of high-resolution telemetry:
- **Total Records**: 3,023 snapshots (sampled every 10 minutes)
- **Time Range**: July 1, 2026, 00:00:00 to July 21, 2026, 23:40:00
- **Operational Scope**: North (1,081), South (980), West (947), and 15 missing records.
- **Traffic Regimes**:
  - `Normal`: 1,600 rows (52.93%)
  - `Sale`: 1,306 rows (43.20%)
  - `Flash Sale`: 117 rows (3.87%)

### Missing Data & Quality Audit:
- Missing fields identified: `region` (15, 0.50%), `memory_utilization` (18, 0.60%), `network_mbps` (18, 0.60%), `disk_utilization` (18, 0.60%).
- Target 1 (`next_15min_requests`): Mean = 31,256 RPM, Min = 13,019 RPM, Max = 76,787 RPM.
- Target 2 (`scale_up_required`): Class 1 (96.06%), Class 0 (3.94%) in raw snapshots.
- **Data Integrity**: Handled with zero row deletion and strict imputation; no synthetic records fabricated.

---

## 4. ML Data Preparation & Leakage Prevention (Task 2)

- **Temporal Train/Evaluation Partitioning**:
  - Strict chronological 80/20 split used to mirror production time-series deployments.
  - **Train Partition**: 2,418 snapshots (July 1, 00:00 to July 17, 18:50).
  - **Evaluation Partition**: 605 snapshots (July 17, 19:00 to July 21, 23:40).
- **Imputation Strategy**:
  - Fitted strictly on the training partition: **Median imputation** for numerical telemetry (resilient against flash sale skew) and **Mode imputation** for categorical region metadata.
- **Outlier Treatment Rationale**:
  - Telemetry spikes (CPU > 90%, response time > 250ms) during Flash Sales represent genuine system stress rather than measurement error. Retaining these outliers is essential for models to learn emergency scaling conditions.
- **Strict Target Isolation**:
  - Both targets (`next_15min_requests` and `scale_up_required`) are completely decoupled from current-state feature matrices and excluded from all unsupervised clustering.

---

## 5. Domain Feature Engineering (Task 3)

Four high-value operational signals were engineered:
1. **Demand Pressure (`requests_per_user`)**: Ratio of request throughput to active concurrent users, measuring traffic concentration per visitor.
2. **Checkout Conversion Density (`orders_to_requests_ratio`)**: Monitors checkout pipeline load and potential payment gateway bottlenecking.
3. **Capacity Headroom (`capacity_headroom`)**: `current_capacity_rpm - requests_per_min`, quantifying the absolute buffer remaining before hardware saturation.
4. **Composite Utilization Pressure (`utilization_pressure`)**: Mean saturation across CPU, Memory, and Disk resources, providing an aggregated system stress index.

---

## 6. Supervised Modeling & Evaluation Results

### A. Demand Forecasting Regression (Task 4)
- **Model**: Multi-variable Linear Regression with Standardized Features.
- **Target**: `next_15min_requests` (15-minute lookahead demand).
- **Performance**:
  - **MAE**: **2,030.47 RPM**
  - **RMSE**: **2,644.16 RPM**
  - **$R^2$ Score**: **0.9189** (Explains 91.89% of demand variance)
  - **MAPE**: **6.67%**
- **SRE Insight**: The model closely tracks normal and scheduled promotional trends with high fidelity, serving as a reliable early warning signal.

### B. Proactive Scale-Up Classification (Task 5)
- **Target**: `scale_up_required` (Binary auto-scaling action).
- **Cost-Asymmetry Analysis**:
  - **False Negative (Missed Scale-Up)**: Fatal failure mode. System throttles, latency spikes > 200ms, checkout fails, incurring ₹40L GMV loss and brand damage.
  - **False Positive (Unnecessary Scale-Up)**: Negligible failure mode. Minor temporary cloud compute cost ($20–$50), zero user disruption.
  - **North Star Metric**: **Recall on Class 1**.

| Metric | Decision Tree | Random Forest (Recommended) | Operational Advantage |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 84.13% | **91.07%** | +6.94% overall precision |
| **Precision (Class 1)** | 98.21% | **97.30%** | Consistently high trigger confidence |
| **Recall (Class 1)** | 85.00% | **93.28%** | **Cuts False Negatives from 87 to 39** |
| **F1-Score (Class 1)** | 91.13% | **95.25%** | Superior harmonic balance |

---

## 7. Traffic & Infrastructure State Discovery (Task 6 & Task 7)

### K-Means Clustering ($K=3$ Chosen via Elbow & Silhouette Analysis):
1. **Cluster 0: Low-Traffic Steady State**: Normal operations, average ~16,000 RPM, CPU < 45%, latency < 40ms, ample capacity headroom (> 15,000 RPM buffer).
2. **Cluster 1: Elevated Sale Ramp-Up**: Promotional events, ~28,000–35,000 RPM, CPU ~65%, latency ~75ms, headroom narrowing.
3. **Cluster 2: Critical Flash Surge Zone**: Flash sale spikes > 50,000 RPM, CPU > 85%, latency crossing 200ms SLA, negative/zero headroom.

### DBSCAN & Hierarchical Insights:
- Hierarchical dendrogram with Ward linkage cleanly reveals the 3 distinct operational strata.
- DBSCAN identified 2 core density clusters and isolated 32 extreme surge snapshots (1.06%) as noise points (`-1`). In production SRE, these noise points correspond to rare, high-severity flash events.

### PCA Dimensionality Reduction:
- 2 Principal Components capture **73.57% of total variance** (PC1: 65.16%, PC2: 8.41%), providing clear 2D separability between operational health regimes.

---

## 8. SRE Proactive Capacity Runbook (Task 8 & Bonus)

```
========================================================================================
                      NIMBUSCART SRE 2 AM OPERATIONAL RUNBOOK
========================================================================================
[1] MONITORING CADENCE:
    • Telemetry Ingestion: 1-minute resolution.
    • ML Model Inference: Rolling evaluation every 5 minutes.

[2] AUTOMATED SCALE-UP TRIGGER RULE:
    IF (Random_Forest_Prob(scale_up_required == 1) >= 0.65)
       OR (Forecasted_Demand_RPM >= 0.80 * Current_Capacity_RPM)
       OR (Capacity_Headroom <= 0.20 * Current_Capacity_RPM):
          --> TRIGGER IMMEDIATE PROVISIONING OF ADDITIONAL POD/NODE TIER.

[3] PRE-WARMING TIMELINE FOR PROMOTIONS:
    • T-60 min: Audit database replica lag and clear stale connection pools.
    • T-30 min: Validate baseline latency and warm cache layers.
    • T-15 min: Scale checkout and API pods to 1.5x expected capacity (5-7 min container boot buffer).
    • T-0  min: Lock deployment pipelines; switch to high-frequency paging.

[4] ADVANCED PRODUCTION TELEMETRY:
    • Ingest P99 Latency and Database Connection Pool Saturation to detect tail backpressure before average latency degrades.
========================================================================================
```

---

## 9. Deliverables & Repository Structure

```
d:\Ass 1\
├── data\
│   └── ecommerce_infrastructure_scaling.csv     # Telemetry dataset (3,023 rows)
├── outputs\
│   ├── 01_correlation_heatmap.png               # Correlation matrix
│   ├── 02_demand_forecast_timeseries.png        # Time-series overlay
│   ├── 03_demand_regression_scatter.png         # Regression actual vs predicted
│   ├── 04_classification_confusion_matrices.png # DT vs RF confusion matrices
│   ├── 05_rf_feature_importance.png             # RF Gini feature importances
│   ├── 06_kmeans_elbow_silhouette.png           # Elbow & Silhouette curves
│   ├── 07_hierarchical_dendrogram.png           # Ward linkage dendrogram
│   ├── 08_pca_2d_clusters.png                   # PCA 2D by infrastructure state
│   ├── 09_pca_2d_sale_event.png                 # PCA 2D by sale event context
│   └── 10_demand_vs_response_time.png           # Throughput vs latency & headroom
├── ecommerce_infrastructure_scaling_analysis.ipynb # Fully executed Capstone Notebook
├── generate_plots.py                            # Plot generator script
├── generate_notebook.py                         # Notebook generator script
├── run_analysis.py                              # Standalone validation script
├── ecomemrce.pdf                                # Skillfyme Assignment Guide
└── README.md                                    # Executive report & SRE runbook
```

---
*Authored for NimbusCart Global SRE & Infrastructure Analytics Leadership.*

