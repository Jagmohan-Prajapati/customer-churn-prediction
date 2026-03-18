# Customer Churn Prediction & Business Insight Engine

An end-to-end Data Science project that predicts customer churn for a telecom business — covering the complete pipeline from raw data exploration to a deployed interactive dashboard with SHAP-based business recommendations.

---

## Problem Statement

Customer churn is one of the most critical business problems in the telecom industry. Retaining an existing customer is 5x cheaper than acquiring a new one. This project builds a machine learning system that:
- Identifies customers likely to churn
- Explains **why** they are likely to churn (SHAP)
- Provides actionable business recommendations to reduce churn

---

## Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── telco_churn.csv              # Raw dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature creation & preprocessing
│   ├── 03_modeling.ipynb            # Model training & benchmarking
│   └── 04_shap_analysis.ipynb       # Feature importance & explainability
│
├── src/
│   ├── preprocess.py                # Data preprocessing functions
│   ├── train.py                     # Model training pipeline
│   └── predict.py                   # Prediction utilities
│
├── models/
│   └── xgboost_best_model.pkl       # Saved best model (generated after training)
│
├── app.py                           # Streamlit dashboard
├── requirements.txt                 # All dependencies
└── README.md
```

---

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Features:** 21 (demographics, services subscribed, contract type, charges)
- **Target:** `Churn` (Yes/No)

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Jagmohan-Prajapati/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create & Activate Virtual Environment
```bash
# Create venv
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

### 4. Register Jupyter Kernel
```bash
python -m ipykernel install --user --name=churn-env --display-name "Python (churn-env)"
```

### 5. Launch Jupyter & Run Notebooks in Order
```bash
jupyter notebook
```
> Select kernel **"Python (churn-env)"** inside each notebook before running.

| Order | Notebook | Purpose |
|---|---|---|
| 1 | `01_EDA.ipynb` | Exploratory Data Analysis |
| 2 | `02_feature_engineering.ipynb` | Feature engineering & preprocessing |
| 3 | `03_modeling.ipynb` | Model training & benchmarking |
| 4 | `04_shap_analysis.ipynb` | SHAP explainability |

### 6. Run Streamlit Dashboard
```bash
streamlit run app.py
```

---

## Approach

| Phase | Description |
|---|---|
| EDA | Distribution analysis, churn patterns, correlation heatmap |
| Feature Engineering | Tenure buckets, service scores, charge ratios, encoding |
| Modeling | Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost |
| Evaluation | ROC-AUC, F1-Score, Precision, Recall, Confusion Matrix |
| Explainability | SHAP values for global + local feature importance |
| Deployment | Streamlit app for real-time churn prediction |

---

## Results

| Model | ROC-AUC | F1-Score |
|---|---|---|
| Logistic Regression | 0.83 | 0.76 |
| Decision Tree | 0.73 | 0.70 |
| Random Forest | 0.85 | 0.78 |
| KNN | 0.78 | 0.72 |
| **XGBoost** | **0.87** | **0.81** |

---

## Key Business Insights

- **Contract type** is the strongest churn predictor — month-to-month customers churn 3x more
- Customers with **tenure < 12 months** are at highest risk
- Lack of **Tech Support** and **Online Security** strongly correlates with churn

---

## Tech Stack

- **Python**, Pandas, NumPy
- **Scikit-Learn**, XGBoost, SHAP
- **Matplotlib**, Seaborn
- **Streamlit**
- **SQL** (SQLAlchemy)

---

## Author

**Jagmohan Prajapat** — [LinkedIn](https://www.linkedin.com/in/jagmohan-prajapati-aaa117200/) | [GitHub](https://github.com/Jagmohan-Prajapati)
