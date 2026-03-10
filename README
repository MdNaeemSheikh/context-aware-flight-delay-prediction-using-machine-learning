<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=220&section=header&text=Aviation%20Delay%20Prediction&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Context-Aware%20ML%20Framework%20%E2%9C%88%EF%B8%8F&descAlignY=60&descSize=18&animation=fadeIn" width="100%"/>

<br/>

# 🛫 Context-Aware Predictive Framework for Aviation Delays

<br/>

![Python](https://img.shields.io/badge/Python-3.12.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Cloud%20Runtime-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)

<br/>

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Framework-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Extreme%20Gradient%20Boosting-189D3D?style=for-the-badge&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Fast%20GBDT-02569B?style=for-the-badge&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2?style=for-the-badge&logoColor=white)
![DiCE](https://img.shields.io/badge/DiCE-Counterfactual%20XAI-E91E63?style=for-the-badge&logoColor=white)

<br/>

![Status](https://img.shields.io/badge/Research%20Status-Complete-00C853?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Models](https://img.shields.io/badge/Models%20Benchmarked-10%2B-FF6F00?style=for-the-badge)
![Imbalance](https://img.shields.io/badge/SMOTE%20%7C%20ADASYN-Imbalance%20Handled-9C27B0?style=for-the-badge)

<br/><br/>

**👤 Author:** &nbsp;[Md Naeem Sheikh](https://www.linkedin.com/in/md-naeem-sheikh) &nbsp;|&nbsp; **📊 Data:** &nbsp;[U.S. Bureau of Transportation Statistics](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)

<br/>

---

</div>

## 📌 Abstract

> *"The goal is not merely to predict delays — it is to understand them well enough to prevent them."*

Airline delays impose systemic costs exceeding **$33 billion annually** on the U.S. aviation industry, affecting millions of passengers and creating cascading operational failures. Conventional predictive models treat all airports uniformly — a fundamental methodological flaw that ignores the structurally different delay dynamics between regional airports and high-throughput hub operations.

This research introduces a **Context-Aware Segmented Modeling Framework** that challenges that assumption directly. By fusing unsupervised K-Means clustering with cluster-specific ensemble learners, the pipeline builds operationally intelligent models that understand the environment they predict within. The framework is extended from **prediction → explanation → prescription** through SHAP interaction analysis, DiCE counterfactual generation, algorithmic robustness stress-testing, and a cost-sensitive Economic Impact Assessment — transforming a classification output into a deployable decision-support instrument.

<br/>

---

## 🏆 Key Research Contributions

<div align="center">

| # | Contribution | Technique Used |
|:---:|:---|:---:|
| 🔵 | **Context-Aware Segmented Modeling** — Airports split into 3 operational clusters; dedicated XGBoost expert per cluster | `K-Means + XGBoost` |
| 🟢 | **Hybrid Stacking Ensemble** — RF + XGBoost + LightGBM fused via Logistic Regression meta-learner | `StackingClassifier` |
| 🟣 | **SHAP Interaction Analysis** — Non-linear interaction between traffic volume & weather lag scientifically revealed | `TreeExplainer` |
| 🔴 | **DiCE Counterfactual Prescriptions** — Minimum operational intervention scenarios generated per high-risk flight | `dice_ml` |
| 🟠 | **Algorithmic Robustness Stress-Test** — Gaussian noise injected at 6 levels; model degradation curve benchmarked | `Noise Injection` |
| 🟡 | **Economic Impact Assessment** — Asymmetric cost matrix (FN = $4,000 / FP = $500) quantifies real-world savings | `Cost-Sensitive Eval` |
| ⚪ | **OSRI Stability Analysis** — Global model's "perfect stability" exposed as degenerate prediction blindness | `Perturbation Testing` |

</div>

<br/>

---

## 🔬 Full Research Pipeline

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         FULL RESEARCH PIPELINE                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   📥 RAW DATA  ──  BTS Airline Delay Cause Dataset                          ║
║         │                                                                    ║
║         ▼                                                                    ║
║   🧹 PHASE 1 — DATA CLEANING                                                ║
║         ├─ Drop rows with missing critical targets                          ║
║         └─ Zero-impute delay cause columns                                  ║
║         │                                                                    ║
║         ▼                                                                    ║
║   ⚙️  PHASE 2 — FEATURE ENGINEERING                                         ║
║         ├─ Binary target: delay_rate > 0.20                                 ║
║         ├─ Lag features: lag_delay_rate · lag_weather_rate                  ║
║         ├─ Cyclical encoding: month_sin · month_cos                         ║
║         ├─ Frequency encoding: carrier_enc · airport_enc                    ║
║         └─ Temporal split: Train < 2022  |  Test ≥ 2022                    ║
║         │                                                                    ║
║         ▼                                                                    ║
║   ⚖️  PHASE 3 — CLASS IMBALANCE CORRECTION                                  ║
║         ├─ SMOTE  (Synthetic Minority Oversampling)                         ║
║         └─ ADASYN (Adaptive Synthetic Sampling)  ← comparative study       ║
║         │                                                                    ║
║         ▼                                                                    ║
║   🤖 PHASE 4 — MODEL DEVELOPMENT & BENCHMARKING                             ║
║         ├─ Baseline:  Logistic Regression · Decision Tree · Random Forest   ║
║         ├─ Ensemble:  GBM · XGBoost · LightGBM  (ROC-AUC comparison)       ║
║         ├─ Advanced:  Hybrid Stacking Classifier                            ║
║         └─ Threshold Tuning:  [0.30 → 0.70],  optimal @ t = 0.46          ║
║         │                                                                    ║
║         ▼                                                                    ║
║   🗂️  PHASE 5 — CONTEXT-AWARE SEGMENTATION                                  ║
║         ├─ K-Means (k=3) on flight volume                                   ║
║         ├─ Cluster 0: Regional airports                                     ║
║         ├─ Cluster 1: Mid-tier airports                                     ║
║         └─ Cluster 2: Elite high-throughput hubs  ← PRIMARY FOCUS          ║
║         │                                                                    ║
║         ▼                                                                    ║
║   🧠 PHASE 6 — EXPLAINABLE AI (XAI)                                         ║
║         ├─ SHAP TreeExplainer:  Feature importance per cluster              ║
║         ├─ SHAP Interaction Values:  arr_flights × lag_weather_rate         ║
║         └─ DiCE:  Diverse counterfactual explanations for high-risk flights ║
║         │                                                                    ║
║         ▼                                                                    ║
║   🛡️  PHASE 7 — ROBUSTNESS & STABILITY TESTING                              ║
║         ├─ Gaussian noise injection (σ = 0.0 to 0.20)                      ║
║         └─ OSRI: Global vs. Expert model perturbation stability             ║
║         │                                                                    ║
║         ▼                                                                    ║
║   💰 PHASE 8 — ECONOMIC IMPACT ASSESSMENT                                   ║
║         ├─ Cost matrix: FN = $4,000  |  FP = $500                          ║
║         └─ Cluster 2 operational savings vs. reactive baseline              ║
║         │                                                                    ║
║         ▼                                                                    ║
║   📋 PHASE 9 — PRESCRIPTIVE ANALYTICS                                       ║
║         └─ "What-If" simulation: Min. traffic reduction to avert each delay ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

<br/>

---

## 🛠️ Technical Stack

<div align="center">

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Language** | Python 3.12.12 | Core development |
| **Platform** | Google Colab | Cloud GPU runtime |
| **Data** | `pandas`, `numpy` | Processing & feature engineering |
| **ML Core** | `scikit-learn` | Models, pipelines, evaluation |
| **Boosting** | `XGBoost`, `LightGBM`, `GBM` | High-performance ensemble learners |
| **Imbalance** | `SMOTE`, `ADASYN` via `imbalanced-learn` | Synthetic minority oversampling |
| **Stacking** | `StackingClassifier` + LR meta-learner | Hybrid ensemble architecture |
| **XAI** | `shap` (TreeExplainer + Interaction Values) | Model interpretation |
| **Counterfactuals** | `dice_ml` | Prescriptive intervention generation |
| **Validation** | `TimeSeriesSplit` 5-fold | Temporally-correct cross-validation |
| **Metrics** | Accuracy · Precision · Recall · F1 · **MCC** · AUC | Imbalance-robust evaluation suite |
| **Visualization** | `matplotlib`, `seaborn` | Learning curves, ROC, confusion matrices |

</div>

<br/>

---

## 🧠 Explainability: From Black Box → Glass Box

### 🔷 SHAP — *What drives each prediction?*
`shap.TreeExplainer` was deployed per cluster to produce global feature importance rankings and **SHAP interaction value plots**, revealing the non-linear coupling between `arr_flights` (traffic volume) and `lag_weather_rate`. This moves the analysis beyond a passive feature ranking into genuine scientific discovery: understanding *how* congestion and weather jointly amplify delay probability in a non-additive manner.

### 🔶 DiCE — *What would change this prediction?*
`dice_ml` generates **Diverse Counterfactual Explanations**: for every flight predicted as delayed, the system computes the minimum operational change required to flip the prediction to no-delay. Results are saved to CSV and visualized as feature delta charts — giving operations teams a concrete, quantified intervention target rather than a passive alert.

<br/>

---

## 📊 Model Benchmarking Overview

<div align="center">

| Model | Strategy | Validation | Metrics Reported |
|:---|:---:|:---:|:---:|
| Logistic Regression | Baseline | 5-Fold TSS | Acc · P · R · F1 · MCC |
| Decision Tree | Baseline | 5-Fold TSS | Acc · P · R · F1 · MCC |
| Random Forest | Baseline | 5-Fold TSS | Acc · P · R · F1 · MCC |
| Logistic Regression | SMOTE | 5-Fold TSS | Acc · P · R · F1 · MCC |
| Decision Tree | SMOTE | 5-Fold TSS | Acc · P · R · F1 · MCC |
| Random Forest | SMOTE | 5-Fold TSS | Acc · P · R · F1 · MCC |
| Gradient Boosting | SMOTE | 5-Fold TSS | + ROC-AUC |
| **XGBoost** | **SMOTE** | **5-Fold TSS** | **+ ROC-AUC** |
| **LightGBM** | **SMOTE** | **5-Fold TSS** | **+ ROC-AUC** |
| **Hybrid Stacking** | **SMOTE** | **5-Fold TSS** | **Full suite** |
| **Expert Models ×3** | **Per-Cluster** | **Held-out 2022+** | **Full suite** |

> ✅ Strict temporal train/test split enforced throughout. Zero data leakage by design.

</div>

<br/>

---

## ⚙️ How to Run

### ▶️ Option A — Google Colab *(Recommended — zero setup)*

```
1. Open https://colab.research.google.com
2. File → Upload notebook → select the .ipynb file
3. Upload Airline_Delay_Cause.csv to Google Drive at:
   /content/drive/MyDrive/Colab Notebooks/Airline_Delay_Cause.csv
4. Runtime → Run all
```

### 💻 Option B — Local Environment

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/aviation-delay-prediction.git
cd aviation-delay-prediction

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Launch notebook
jupyter notebook Context_Aware_Flight_Delay_Prediction_Using_ML.ipynb
```

> **For local use:** Remove `from google.colab import drive` and `drive.mount(...)` from Cell 1, and update the CSV file path to your local directory.

**`requirements.txt`**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
dice-ml
jupyter
```

<br/>

---

## 📁 Repository Structure

```
📦 aviation-delay-prediction
 ┣ 📓 Context_Aware_Flight_Delay_Prediction_Using_ML.ipynb   ← Full research notebook
 ┣ 📄 README.md                                               ← This file
 ┣ 📄 requirements.txt                                        ← Python dependencies
 ┗ 📂 data/
    ┗ 📄 README_data.md                                       ← Dataset download instructions
```

<br/>

---

## 🔭 Future Work

- [ ] Integrate real-time NWS weather API as a live inference feature
- [ ] Extend clustering to multi-dimensional airport profiling (volume + cancellation rate + network centrality)
- [ ] Investigate Temporal Fusion Transformer (TFT) for long-range sequence modeling
- [ ] Deploy expert models as a FastAPI microservice for airline operations dashboards
- [ ] Apply causal inference to separate correlation from true operational causality

<br/>

---

## 👤 Author

<div align="center">

### Md Naeem Sheikh

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/md-naeem-sheikh)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YOUR_USERNAME)

*Researcher in applied machine learning, transportation analytics, and data-driven decision systems.*
*Actively seeking MSc / PhD research opportunities.*

</div>

<br/>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=120&section=footer" width="100%"/>

*Dataset courtesy of the U.S. Bureau of Transportation Statistics (BTS). For academic and research purposes.*

⭐ **If this project helped you, please consider starring the repository** ⭐

</div>
