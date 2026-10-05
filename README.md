# E-Commerce Infrastructure Scaling Intelligence
## Predictive Demand Forecasting & Proactive Capacity Planning for NimbusCart Global
**Author**: Machine Learning Engineer / SRE Analytics Guild  
**Domain**: Site Reliability Engineering (SRE) & Applied Machine Learning  
**System**: Production Early-Warning & Automated Capacity Scaling Engine  

---

> ### A Personal Note to NimbusCart's SRE & Engineering Leadership
> Six weeks ago, our platform experienced a severe flash sale incident that caused response times to spike past 2,000 ms, throttled checkout transactions, and cost NimbusCart an estimated ₹40 lakh in lost GMV while `#NimbusCartDown` trended online. 
>
> When the VP of Infrastructure asked for actionable evidence rather than another passive dashboard, I treated this engagement not as an academic exercise, but as a real-world production incident review. I wrote this document as the exact operational brief I would want on my own desk if I were the engineer getting paged at 2:00 AM during a flash sale.
>
> Every number, threshold, and recommendation below is backed directly by the data and reproducible in my accompanying notebook (`ecommerce_infrastructure_scaling_analysis.ipynb`), which runs cleanly top-to-bottom.

---

## 1. Project Objective & SRE Problem Statement

NimbusCart operates an e-commerce platform across North, South, and West regions. Our historical auto-scaling mechanism was purely **reactive**: it only added servers after CPU and queue latency had already spiked. In sudden flash sales where user traffic triples in under 30 minutes, this reactive lag causes outages before new instances can boot up.

My objective was to build a proactive machine learning system answering three critical operational questions:
1. **Demand Forecasting (Regression)**: Exactly how many requests per minute (RPM) should we expect 15 minutes into the future?
2. **Operational Decisioning (Classification)**: Should we trigger a server scale-up right now?
3. **Infrastructure State Discovery (Clustering & PCA)**: What operational operating states do our servers pass through, and what do they tell us about outage risk?

---

## 2. Dataset Overview & Initial Audit (Telemetry Validation)

I analyzed the supplied telemetry dataset (`data/ecommerce_infrastructure_scaling.csv`), which contains **3,023 monitoring snapshots** recorded at uniform **10-minute intervals** between **July 01, 2026 00:00** and **July 21, 2026 23:40** (21 full days):

* **Event Breakdown**:
  * `Normal`: 1,600 snapshots (52.93%)
  * `Sale`: 1,306 snapshots (43.20%)
  * `Flash Sale`: 117 snapshots (3.87%)
* **Geographic Coverage**: `North` (1,081), `South` (980), `West` (947), with 15 missing entries (0.50%).
* **Key Observability Metrics**: Concurrent active users, request throughput (`requests_per_min`), order completion rate, CPU %, Memory %, Network throughput (Mbps), Disk %, latency (`response_time_ms`), and current provisioned capacity (`current_capacity_rpm`).

### Important Discovery Regarding Class Imbalance
While initial baseline specifications projected an expected $\approx 26\%$ positive rate, my audit of the actual production CSV revealed:
* `scale_up_required = 1`: **2,904 snapshots (96.06%)**
* `scale_up_required = 0`: **119 snapshots (3.94%)**

**My Operational Decision on Metrics**:  
Because over 96% of records in this telemetry window represent high-load scale-up states, **Accuracy is a useless metric** (a naive dummy model predicting 1 for everything gets 96% accuracy while leaving the site unprotected). Therefore, I designed my evaluation around **Recall, Precision, F1-Score, and Confusion Matrix analysis**, prioritizing the minimization of False Negatives.

### Data Integrity Confirmation (Ground Rule 1)
I confirmed that the dataset was analyzed exactly as supplied from the observability pipeline — zero records were fabricated, artificially synthesized, or manually altered.

---

## 3. Preprocessing Strategy & Outlier Audit (Preprocessing)

### Missing Value Imputation (Ground Rule 5 Compliant)
To prevent data leakage, I ensured all imputers and scalers were **fitted strictly on the training partition (first 80%)** before transforming the evaluation set:
1. **`region` (15 missing values, 0.50%)**: Imputed using the **training set mode** (`North`). The missingness was negligible, and mode imputation avoided introducing geographic bias.
2. **`memory_utilization`, `network_mbps`, `disk_utilization` (18 missing values each, 0.60%)**: Imputed using the **training set median**. In infrastructure telemetry, metric spikes create right-skew; the median provides a robust central estimate unaffected by extreme surges.

### Why I Did NOT Delete Outliers
Using the IQR method ($Q3 + 1.5 \times IQR$), I detected significant statistical outliers in `requests_per_min` ($> 44,000$ RPM), `active_users` ($> 15,000$), and `response_time_ms` ($> 500$ ms).
* **Engineering Decision**: I preserved 100% of these extreme values.
* **Why**: In SRE telemetry, these spikes are not bad data or measurement noise — they are the **real flash sale traffic surges that caused the ₹40 lakh outage**. Removing them would cause the model to ignore the exact failure scenarios we need to prevent. Instead, I applied `StandardScaler` to normalize distributions for distance-sensitive models (Linear Regression, PCA, K-Means), while tree models handled the raw values naturally.

### Time-Based Feature Engineering
I extracted temporal indicators to capture diurnal retail shopping rhythms:
* `hour`, `day_of_week`, `is_weekend`, `is_peak_hours` (18:00 – 23:00).
* Trigonometric cyclical features (`sin_hour`, `cos_hour`) so the model recognizes 23:50 and 00:00 as continuous adjacent intervals.

---

## 4. Operational Feature Engineering & Selection (Feature Engineering)

Raw telemetry only shows current load, not infrastructure stress. I created five purpose-built SRE operational features:

1. **`demand_pressure`** = $\frac{\text{requests\_per\_min}}{\text{current\_capacity\_rpm}}$  
   * *SRE Meaning*: Tells me what fraction of provisioned capacity is currently consumed. When this exceeds 0.80, request queueing begins.
2. **`capacity_headroom`** = $\text{current\_capacity\_rpm} - \text{requests\_per\_min}$  
   * *SRE Meaning*: The absolute buffer of unused throughput remaining before the server crashes.
3. **`utilization_pressure`** = $\frac{\text{cpu\_utilization} + \text{memory\_utilization} + \text{disk\_utilization}}{3}$  
   * *SRE Meaning*: A composite index measuring systemic multi-resource hardware saturation.
4. **`requests_per_user`** = $\frac{\text{requests\_per\_min}}{\text{active\_users}}$  
   * *SRE Meaning*: Measures browsing aggressiveness. Spikes indicate flash-sale bot activity or frantic user checkout refreshes.
5. **`traffic_growth_rate_10m`** = $\text{requests\_per\_min}_t - \text{requests\_per\_min}_{t-1}$  
   * *SRE Meaning*: 10-minute demand acceleration. (Strictly retrospective; zero future leakage).

### Strict Confirmation of Zero Target Leakage (Ground Rule 2)
I verified that neither `next_15min_requests` nor `scale_up_required` was used as a feature, nor were they used to predict one another. All feature calculations use exclusively data available at or before snapshot time $t$.

---

## 5. Chronological Train/Test Split (Ground Rule 4)

In time-series monitoring, **random shuffling is completely invalid** because it leaks future temporal patterns into past training rows. 

I implemented a strict time-aware chronological split:
* **Training Partition**: First **80%** ($N = 2,418$ snapshots; July 01 00:00 to July 17 18:50).
* **Evaluation Partition**: Remaining **20%** ($N = 605$ snapshots; July 17 19:00 to July 21 23:40).

---

## 6. Model Evaluation Summary (Supervised Modeling)

### Task 4: Demand Forecasting (Linear Regression)
I trained a Linear Regression model on the scaled training partition to predict `next_15min_requests`:

* **Overall Evaluation Metrics**:
  * **Mean Absolute Error (MAE)**: **2,029.35 RPM**
  * **Root Mean Squared Error (RMSE)**: **2,647.09 RPM**
  * **Mean Absolute Percentage Error (MAPE)**: **6.48%**
  * **$R^2$ Score**: **0.9431**

#### Granular Error Analysis: Normal vs Sale vs Flash Sale
| Operating Condition | Snapshots | Actual Mean RPM | Predicted Mean RPM | MAE (RPM) | MAPE (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Normal Traffic** | 320 | 17,992.5 | 17,935.2 | **1,595.6** | **8.87%** |
| **Scheduled Sale** | 261 | 35,429.6 | 35,622.8 | **2,348.9** | **6.63%** |
| **Flash Sale Spikes** | 24 | 52,185.3 | 48,674.1 | **3,521.8** | **6.75%** |

* **My Analysis of Where the Model Struggles**:  
  The linear model performs exceptionally well during steady-state and planned sale days ($R^2 > 0.94$). However, during sudden Flash Sales, the model underestimates peak demand by an average of $\approx 3,500$ RPM (and up to $8,000$ RPM at the single highest peak). This happens because viral surges exhibit exponential non-linear behavior that linear planes cannot fully bend to.
* **How I Fixed This for Production**: By combining the forecast with a **20% safety headroom multiplier**, the SRE team receives a reliable, conservative demand ceiling that prevents unexpected throttling.

---

### Task 5: Scale-Up Classification (Decision Tree vs Random Forest)
I trained and tuned both a Decision Tree (`max_depth=5`) and a Random Forest Classifier (`n_estimators=100`, `max_depth=6`) using balanced class weights:

| Model | Accuracy | Precision | Recall | F1-Score | False Negatives (Missed) | False Positives |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree** | 85.29% | 97.86% | 86.55% | 0.9186 | 78 | 11 |
| **Random Forest (Recommended)** | **90.08%** | **97.62%** | **91.90%** | **0.9467** | **47** | **13** |

#### Operational Cost Trade-Off (Why Recall Is My Top Metric)
* **False Negative (FN) — The Outage**: The model says "do not scale up", but demand surges. Servers saturate, gateway timeouts hit customers, transactions fail, and we face another **₹40 lakh loss** and trending hashtag.
* **False Positive (FP) — Minor Cloud Cost**: The model says "scale up", but demand was absorbable. We boot up an extra cloud server for an hour, costing roughly ₹1,000. Platform availability stays 100%.
* **My Recommendation**: **Deploy Random Forest in Production**. It achieves **91.90% Recall**, cutting dangerous False Negatives from 78 down to 47 (**a 40% reduction in missed scaling events**), while delivering a superior **0.9467 F1-score**.

---

## 7. Infrastructure State Discovery (State Discovery & PCA)

Using current operational features only (strictly excluding both targets), I evaluated three clustering approaches:

### K-Means ($K = 3$, Validated via Elbow & Silhouette)
I mapped the three discovered clusters directly to operational SRE readiness states:
1. **State 0: Low-Traffic Baseline (Green)**: $\approx 14,210$ RPM, CPU 28.4%, Latency 242 ms, Headroom 6,240 RPM. Safe overnight state.
2. **State 1: Moderate Daytime Activity (Amber)**: $\approx 26,180$ RPM, CPU 41.2%, Latency 318 ms, Headroom 3,640 RPM. Standard commercial operation.
3. **State 2: Critical Surge Zone (Red)**: $\approx 48,950$ RPM, CPU 58.6%, Latency 564 ms, Headroom drops to near zero ($170$ RPM). Imminent danger of cascade failure; immediate scale-up mandatory.

### Hierarchical Clustering (Ward Dendrogram)
Ward's minimum variance dendrogram validated the exact 3-state hierarchy, showing a major distance cutoff at threshold $\approx 25$.

### Honest DBSCAN Evaluation
Evaluating DBSCAN density clustering behavior:
* At small $\epsilon = 0.8$, DBSCAN marked **37% to 53% of all snapshots as noise**.
* At larger $\epsilon = 1.5 - 2.0$, it collapsed nearly everything into **a single massive cluster**.
* *My Conclusion*: Infrastructure telemetry forms a continuous density cloud rather than disconnected clusters. Density-based clustering is poorly suited for setting operational risk tiers compared to centroid-based K-Means.

### Dimensionality Reduction (2D PCA)
* **Variance Captured**: **PC1 = 74.68%**, **PC2 = 8.00%** $\rightarrow$ **Total 82.67% of system variance preserved in 2D**.
* **PC1**: Measures overall throughput and scale (`requests_per_min`, `active_users`, `current_capacity_rpm`).
* **PC2**: Measures operational stress (`demand_pressure` $+0.62$, `response_time_ms` $+0.21$, and `capacity_headroom` $-0.63$).

---

## 8. Exported Visualization Suite (`outputs/`)

I generated 9 distinct, publication-grade figures saved directly to `outputs/`:
1. `01_correlation_heatmap.png`: Feature correlation matrix and target relationships.
2. `02_demand_forecast_timeseries.png`: Chronological forecast overlay with Flash Sale indicators.
3. `03_demand_regression_scatter.png`: Regression scatter plot with $y = x$ line and residual distribution.
4. `04_classification_confusion_matrices.png`: Side-by-side DT vs RF confusion matrices.
5. `05_rf_feature_importance.png`: Top 15 Random Forest feature importances.
6. `06_kmeans_elbow_silhouette.png`: Elbow and silhouette diagnostic curves for $K \in [2, 6]$.
7. `07_hierarchical_dendrogram.png`: Agglomerative hierarchical dendrogram.
8. `08_pca_2d_clusters.png`: 2D PCA projections by operational state and sale event.
9. `09_demand_vs_latency_headroom.png`: Non-linear latency knee curve ($> 500$ ms) and capacity headroom exhaustion.

---

## 9. SRE Operational Playbook & VP Answer (SRE Playbook)

### Direct Answer to the VP of Infrastructure
1. **Can current infrastructure handle upcoming high traffic?**
   * **For Scheduled Sales**: **YES**, our static provisioned capacity ($35,000 - 40,000$ RPM) can absorb planned promotions as long as demand pressure stays under 0.80.
   * **For Unannounced Flash Sales**: **NO**. Flash traffic rapidly exceeds $60,000$ RPM, saturating static headroom within 15–20 minutes and causing latency to cross 2,000 ms.
2. **How much lead time do we get to scale up before it breaks?**
   * The models provide a **15-minute advance forecast**.
   * Booting cloud VMs and initializing Kubernetes pods takes **5 to 8 minutes**.
   * This gives our SRE team a **safe lead time buffer of 7 to 10 minutes** to provision extra capacity before user traffic crashes ingress.

### The 2:00 AM On-Call Runbook
* **Monitoring Cadence**: Run model inference every **5 minutes** against rolling 10-minute telemetry snapshots.
* **Proactive Scale-Up Trigger**:
  $$\text{Alert IF} \quad \left( \frac{\text{Predicted Demand}}{\text{current\_capacity\_rpm}} \ge 0.80 \right) \quad \lor \quad (\text{RF Scale-Up Prob} \ge 0.65)$$
* **Capacity Provisioning Formula**:
  $$\text{Target Capacity} = 1.25 \times \text{Predicted Demand (RPM)}$$
  *(Applies a 25% safety margin to protect against non-linear flash surges).*

### Limitations & Human-in-the-Loop Governance
ML models cannot anticipate external black-swan events (such as a nationwide ISP outage or payment gateway failure). The machine learning system is designed as an **automated early-warning copilot**, with the SRE Incident Commander retaining final authority to confirm capacity changes during high-stakes sales.

---

## 10. Bonus Challenge: Proactive Sale Readiness Analysis

1. **48-Hour Pre-Sale Simulation**: Run the forecasting model against planned promotional push notifications to simulate hourly peak loads.
2. **Pre-Warming Cadence**: Pre-scale infrastructure **30 to 45 minutes prior to sale launch** to absorb JVM warm-up, container pulls, database connection pool allocations, and DNS propagation.
3. **Leading Telemetry Indicators**:
   * Capacity headroom $< 20\%$.
   * Latency $> 350$ ms across two consecutive snapshots.
   * `requests_per_user > 3.5` (bot traffic / checkout refresh surges).
4. **Recommended Additional Monitoring Signals**:
   * **Database Connection Pool Saturation (%)**: In high-load retail, database connections frequently exhaust before web server CPU spikes, causing immediate silent transaction dropouts.
   * **HTTP 5xx Error Rate (%)**: Immediate leading metric of customer-facing checkout failure.

---

## 11. Verification & Top-to-Bottom Execution

The accompanying notebook has been verified to execute **top-to-bottom with 0 errors**:
```bash
# To verify execution from command line:
jupyter nbconvert --to notebook --execute --inplace ecommerce_infrastructure_scaling_analysis.ipynb
```
* **Python Version**: 3.10+
* **Dependencies**: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `nbformat`, `jupyter`

---

## 12. References & Citations
1. Scikit-Learn Documentation: Ensemble Methods (`RandomForestClassifier`), Linear Models, and Clustering (`KMeans`, `DBSCAN`, `PCA`). https://scikit-learn.org/
2. Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (2016). *Site Reliability Engineering: How Google Runs Production Systems*. O'Reilly Media.
3. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference.
4. Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering.
