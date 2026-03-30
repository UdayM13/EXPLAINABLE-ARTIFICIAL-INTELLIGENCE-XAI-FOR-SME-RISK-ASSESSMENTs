# EXPLAINABLE-ARTIFICIAL-INTELLIGENCE-XAI-FOR-SME-RISK-ASSESSMENTs
# 🔍 Explainable AI (XAI) for SME Risk Assessment

> **Master's Thesis Project**
> *A Comparative Study Using SHAP-Based Interpretability*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44+-red.svg)](https://shap.readthedocs.io/)

---

## 📌 Research Question

> *"How can Explainable AI (XAI) techniques improve the transparency and trustworthiness of risk assessment models for Small and Medium Enterprises (SMEs)?"*

---

## 🎯 Overview

Small and Medium Enterprises (SMEs) account for over **99.8% of all EU businesses** and employ nearly **90 million people**, yet they remain disproportionately underserved by traditional credit risk frameworks. This project develops a full **XAI-enabled machine learning pipeline** for SME financial distress prediction, combining the predictive power of gradient boosting models with the theoretical rigour of **SHAP (SHapley Additive exPlanations)** to produce transparent, auditable, and regulatory-compliant risk assessments.

The pipeline is designed to satisfy the transparency requirements of the **EU AI Act (2024)**, **GDPR Article 22**, and **EBA guidelines on ML in credit risk**.

---

## 📊 Datasets

| Dataset | Source | Instances | Features | Target |
|---|---|---|---|---|
| Polish Companies Bankruptcy | [UCI ML Repository](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data) | 10,503 | 64 financial ratios | Bankruptcy (0/1) |
| Credit Risk Dataset | [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) | 32,581 | 11 features | Loan default (0/1) |

The Polish dataset is downloaded **automatically** via `ucimlrepo`. No manual download needed.

---

## 🤖 Models Compared

| Model | Type | Explainability |
|---|---|---|
| Logistic Regression | Interpretable baseline | Coefficients |
| Random Forest | Ensemble / Black-box | Feature importance |
| **XGBoost** ⭐ | Gradient Boosting | **SHAP (primary)** |
| LightGBM | Gradient Boosting | SHAP |

---

## 🧠 XAI Methods

This project applies **SHAP (SHapley Additive exPlanations)** — grounded in cooperative game theory (Shapley, 1953) — to provide:

- **Global explainability** — Beeswarm plots and bar charts ranking the most important financial risk drivers across all SMEs
- **Local explainability** — Waterfall plots showing exactly why an individual SME was classified as high or low risk
- **Dependence plots** — How the top risk driver affects predicted default probability across the full population

---

## 📁 Project Structure

```
xai-sme-risk/
│
├── xai_sme_fixed.py               ← Main Python pipeline (run this)
├── XAI_SME_Step_by_Step.ipynb     ← Step-by-step Jupyter Notebook
├── requirements.txt               ← All required libraries
├── README.md                      ← This file
│
└── thesis_outputs/                ← Auto-generated when you run the code
    ├── eda_01_class_distribution.png
    ├── eda_02_missing_values.png
    ├── eda_03_distributions.png
    ├── eda_04_correlation.png
    ├── plot_01_roc_curves.png
    ├── plot_02_model_comparison.png
    ├── plot_03_heatmap.png
    ├── plot_04_confusion_matrices.png
    ├── polish_shap_beeswarm.png
    ├── polish_shap_bar.png
    ├── polish_shap_waterfall_high.png
    ├── polish_shap_waterfall_low.png
    ├── polish_shap_dependence.png
    ├── credit_shap_beeswarm.png
    ├── credit_shap_bar.png
    ├── credit_shap_waterfall_high.png
    ├── credit_shap_waterfall_low.png
    ├── results_summary.csv
    ├── polish_shap_top_features.csv
    └── credit_shap_top_features.csv
```

---

## ⚙️ Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/xai-sme-risk.git
cd xai-sme-risk
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Download the Kaggle dataset
Download `credit_risk_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) and place it in the project root folder.

> If the file is not found, the pipeline automatically generates a synthetic fallback dataset and continues without interruption.

### Step 4 — Run the pipeline
```bash
python xai_sme_fixed.py
```

Or open the notebook:
```bash
jupyter notebook XAI_SME_Step_by_Step.ipynb
```

---

## 📦 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
xgboost>=2.0.0
lightgbm>=4.0.0
shap>=0.44.0
ucimlrepo>=0.0.6
```

Install all at once:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm shap ucimlrepo
```

---

## 📈 Key Results

### Model Performance

| Model | Polish AUC | Polish F1 | Credit AUC | Credit F1 |
|---|---|---|---|---|
| Logistic Regression | 0.930 | 0.368 | 0.952 | 0.744 |
| Random Forest | 0.906 | 0.020 | 0.943 | 0.748 |
| **XGBoost** | **0.923** | **0.417** | 0.937 | 0.734 |
| LightGBM | 0.919 | 0.438 | **0.940** | **0.746** |

### Top SHAP Risk Drivers

**Polish Bankruptcy Dataset:**
| Rank | Feature | Interpretation | Mean \|SHAP\| |
|---|---|---|---|
| 1 | Attr2 | Total liabilities / Total assets (leverage) | 0.362 |
| 2 | Attr1 | Net profit / Total assets (profitability) | 0.163 |
| 3 | Attr3 | Working capital / Total assets (liquidity) | 0.085 |
| 4 | Attr34 | Operating expenses ratio | 0.084 |
| 5 | Attr6 | Retained earnings / Total assets | 0.077 |

**Credit Risk Dataset:**
| Rank | Feature | Interpretation | Mean \|SHAP\| |
|---|---|---|---|
| 1 | int_rate | Loan interest rate | 0.319 |
| 2 | dti | Debt-to-income ratio | 0.306 |
| 3 | annual_inc | Annual income | 0.268 |

> Leverage, profitability, and liquidity consistently dominate as the primary SME risk drivers — directly validating classical financial distress theory (Altman, 1968; Beaver, 1966).

---

## 🖼️ Sample Outputs

### SHAP Beeswarm — Polish Bankruptcy
Each dot is one SME. Red = high feature value increases bankruptcy risk. Blue = low value reduces it.

### SHAP Waterfall — Individual SME Explanation
Shows exactly which financial ratios pushed a specific SME's predicted bankruptcy probability above or below the population average. Directly satisfies **GDPR Article 22** right to explanation.

---

## ⚖️ Regulatory Compliance

This pipeline directly addresses:

| Regulation | Requirement | How this project satisfies it |
|---|---|---|
| **EU AI Act (2024)** | Transparency for high-risk AI credit scoring | SHAP global bar charts + beeswarm plots |
| **GDPR Article 22** | Individual explanation of automated decisions | SHAP waterfall plots per SME |
| **EBA ML Guidelines** | Model documentation + feature importance | Results CSV + SHAP rankings |

---

## 📚 Key References

- Lundberg, S.M. and Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
- Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
- Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16.
- Altman, E.I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4).
- Beaver, W.H. (1966). Financial ratios as predictors of failure. *Journal of Accounting Research*, 4.
- Zieba, M. et al. (2016). Ensemble Boosted Trees for Bankruptcy Prediction. *Expert Systems with Applications*, 58.
- European Commission (2024). Regulation (EU) 2024/1689 — AI Act.

---

## 👤 Author

**[Your Name]**
[Your University] — Master's in [Your Programme]
Year: 2025
Supervisor: [Supervisor Name]

---

## 📄 License

This project is licensed under the **MIT License** — free to use for academic and research purposes with attribution.

---

## ⭐ Citation

If you use this pipeline in your research, please cite:

```
@mastersthesis{yourname2025xai,
  title  = {Explainable AI (XAI) for SME Risk Assessment:
             A Comparative Study Using SHAP-Based Interpretability},
  author = {[Your Name]},
  school = {[Your University]},
  year   = {2025}
}
```
