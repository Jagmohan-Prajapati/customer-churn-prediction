# Customer Churn Prediction & Business Insight Engine

An end-to-end Data Science project predicting customer churn for a telecom business —
covering the complete pipeline from raw data exploration to a deployed interactive
dashboard with SHAP-based explainability and actionable business recommendations.

---

## Problem Statement

Customer churn is one of the most critical business problems in the telecom industry.
Retaining an existing customer is **5x cheaper** than acquiring a new one.

This project builds a production-grade ML system that:
- Identifies customers likely to churn with **86.4% ROC-AUC**
- Explains **why** each customer is at risk using SHAP values
- Quantifies **revenue at risk** per customer segment
- Provides a **5-page interactive Streamlit dashboard** for business teams

---

## Results

| Model | Accuracy | ROC-AUC | F1-Score | Precision | Recall |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.808 | **0.847** | 0.597 | 0.672 | 0.537 |
| XGBoost (baseline) | 0.787 | 0.822 | 0.573 | 0.613 | 0.537 |
| Random Forest | 0.778 | 0.819 | 0.540 | 0.599 | 0.492 |
| KNN | 0.760 | 0.776 | 0.536 | 0.551 | 0.521 |
| Decision Tree | 0.742 | 0.682 | 0.533 | 0.514 | 0.553 |
| **XGBoost (tuned)** | **0.806** | **0.849** | **0.584** | **0.676** | **0.513** |

> **Best model:** XGBoost tuned via GridSearchCV (5-fold CV, 24 combinations)
> **Best params:** `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`, `subsample=0.8`
> **No overfitting:** CV AUC = 0.8488 vs Test AUC = 0.8490 (gap = 0.0002)

---

## Key Findings

### Top Churn Drivers (SHAP Analysis)

| Rank | Feature | Mean \|SHAP\| | Business Meaning |
|---|---|---|---|
| 1 | `Contract_Month-to-month` | 0.692 | No long-term commitment = highest churn risk |
| 2 | `charge_ratio` | 0.344 | Engineered feature — price shock = strong churn trigger |
| 3 | `OnlineSecurity_No` | 0.240 | No add-ons = lower engagement, higher churn |
| 4 | `MonthlyCharges` | 0.229 | Higher bill = cost-sensitive customers leave more |
| 5 | `InternetService_Fiber optic` | 0.190 | Competitive segment — higher churn rate |
| 6 | `TechSupport_No` | 0.168 | No support = frustration = churn |
| 7 | `PaymentMethod_Electronic check` | 0.157 | Least committed payment method |
| 8 | `PaperlessBilling` | 0.113 | Paperless customers churn more |
| 9 | `tenure` | 0.111 | New customers (< 12 months) are highest risk |
| 10 | `Contract_Two year` | 0.091 | Two-year contracts strongly protect against churn |

> `charge_ratio` is an **engineered feature** (MonthlyCharges / TotalCharges)
> that ranked **#2 out of 47 features** — validating the feature engineering phase.

### Business Impact
- **1,432 customers** predicted to churn (20.3% of base)
- **~$2.47M** estimated 24-month LTV at risk
- **Top 10 highest-risk customers** all correctly identified as actual churners
- At **20% retention rate** with $50/customer incentive → **~$494K revenue saved**

---

## Project Structure

customer-churn-prediction/  
│  
├── data/  
│ ├── telco_churn.csv # Raw dataset (from Kaggle)  
│ ├── processed_churn.csv # Cleaned & engineered (auto-generated)  
│ ├── batch_predictions.csv # Full scored dataset (auto-generated)  
│ ├── model_benchmark_results.csv # Model comparison table (auto-generated)  
│ ├── shap_bar_importance.png # Global SHAP bar chart  
│ ├── shap_beeswarm.png # SHAP beeswarm plot  
│ ├── shap_waterfall_high_risk.png # Waterfall — highest risk customer  
│ ├── shap_waterfall_low_risk.png # Waterfall — lowest risk customer  
│ ├── shap_feature_importance.csv # Mean |SHAP| values table  
│ ├── risk_tier_analysis.png # Risk tier bar charts  
│ ├── confusion_matrix_train.png # Confusion matrix  
│ ├── model_comparison_train.png # Model comparison bar chart  
│ ├── full_churn_risk_report.csv # All customers scored with risk tier  
│ └── top10_highest_risk_customers.csv # Top 10 highest-risk customers  
│  
├── notebooks/  
│ ├── 01_EDA.ipynb # Exploratory Data Analysis  
│ ├── 02_feature_engineering.ipynb # Feature creation & preprocessing  
│ ├── 03_modeling.ipynb # Model training, benchmarking & tuning  
│ ├── 04_shap_analysis.ipynb # SHAP explainability  
│ └── 05_business_report.ipynb # Business insights & retention strategy  
│  
├── src/  
│ ├── preprocess.py # ChurnPreprocessor — full pipeline class  
│ ├── train.py # ChurnTrainer — training pipeline class  
│ └── predict.py # ChurnPredictor — prediction utilities  
│  
├── models/  
│ ├── xgboost_best_model.pkl # Best trained XGBoost model (auto-generated)  
│ ├── scaler.pkl # Fitted StandardScaler (auto-generated)  
│ ├── feature_names.pkl # Feature name list (auto-generated)  
│ └── best_params.json # Best hyperparameters (auto-generated)  
│  
├── app.py # Streamlit dashboard (5 pages)  
├── requirements.txt # All dependencies  
├── .gitattributes # Suppresses Jupyter language detection  
└── README.md  


> Files marked *auto-generated* are produced by running the pipeline in order.
> All `models/` and generated `data/` files are excluded from version control via `.gitignore`.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Jagmohan-Prajapati/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create & Activate Virtual Environment
```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Preprocessing + Training Pipeline
```bash
# Preprocesses raw data AND trains all models in one command
python src/train.py
```

### 5. Run Predictions
```bash
# Single + batch predictions with evaluation
python src/predict.py
```

### 6. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## Run Notebooks (Optional — Deep Dive)

```bash
python -m ipykernel install --user --name=churn-env --display-name "Python (churn-env)"
jupyter notebook
```

> Select kernel **"Python (churn-env)"** inside each notebook.

| Order | Notebook | Output |
|---|---|---|
| 1 | `01_EDA.ipynb` | Churn pattern visualisations |
| 2 | `02_feature_engineering.ipynb` | `data/processed_churn.csv` |
| 3 | `03_modeling.ipynb` | `models/` artifacts + benchmark results |
| 4 | `04_shap_analysis.ipynb` | SHAP charts + feature importance CSV |
| 5 | `05_business_report.ipynb` | Risk report + retention strategy |

---

## Streamlit Dashboard — Pages

| Page | Description |
|---|---|
| **Overview** | Dataset KPIs, risk tier distribution, churn by contract & tenure |
| **Single Prediction** | Real-time churn prediction for one customer via form input |
| **Batch Prediction** | Upload CSV → score all customers → download results |
| **Model Performance** | Benchmark table, confusion matrix, overfitting check |
| **Business Insights** | SHAP charts, revenue at risk, retention strategy by tier |

---

## Approach

| Phase | Notebook | Key Work |
|---|---|---|
| **EDA** | `01_EDA.ipynb` | Class imbalance (26.5%), contract/tenure/charge patterns |
| **Feature Engineering** | `02_feature_engineering.ipynb` | `charge_ratio`, `service_score`, `tenure_group` → 47 features |
| **Modelling** | `03_modeling.ipynb` | 5 models benchmarked, XGBoost tuned via GridSearchCV |
| **Explainability** | `04_shap_analysis.ipynb` | SHAP TreeExplainer — global + per-customer explanations |
| **Business Report** | `05_business_report.ipynb` | Revenue at risk, risk tiers, retention recommendations |

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.12 |
| **Data** | Pandas, NumPy |
| **ML** | Scikit-Learn, XGBoost |
| **Explainability** | SHAP (TreeExplainer) |
| **Tuning** | GridSearchCV (5-fold CV) |
| **Visualisation** | Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Serialisation** | Joblib |
| **Environment** | Jupyter Notebook, venv |

---

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Raw features:** 21
- **Engineered features:** 47
- **Target:** `Churn` (Yes/No) — 26.5% positive class (imbalanced)

---

## Roadmap

- [x] Exploratory Data Analysis
- [x] Feature Engineering (47 features from 21 raw)
- [x] Model Benchmarking (5 models)
- [x] Hyperparameter Tuning (GridSearchCV)
- [x] SHAP Explainability
- [x] Business Report & Risk Segmentation
- [x] Production `src/` module (preprocess, train, predict)
- [x] Streamlit Dashboard (5 pages)
- [ ] Docker containerisation
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] REST API with FastAPI

---

## Author

**Jagmohan Prajapati**

[LinkedIn](https://www.linkedin.com/in/jagmohan-prajapati-aaa117200/) •
[GitHub](https://github.com/Jagmohan-Prajapati) •
[Email](mailto:Jagmohanprajapat003@gmail.com)