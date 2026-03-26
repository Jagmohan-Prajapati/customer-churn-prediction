"""
app.py
──────
Streamlit dashboard for the Customer Churn Prediction project.

Provides:
  - Overview: dataset-level churn stats and risk distribution
  - Single Predictor: predict churn for one customer via sidebar inputs
  - Batch Predictor: upload a CSV and score all customers at once
  - Model Performance: benchmark results and confusion matrix
  - Business Insights: SHAP-based recommendations

Run:
    streamlit run app.py
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# Path Setup (import from src/)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import ChurnPredictor

# Page Config
st.set_page_config(
    page_title = "Churn Predictor",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# Constants
PROCESSED_PATH   = '../data/processed_churn.csv'
RAW_PATH         = '../data/telco_churn.csv'
BENCHMARK_PATH   = 'src/data/model_benchmark_results.csv'
SHAP_BAR_PATH    = '../data/shap_bar_importance.png'
SHAP_BEESWARM    = '../data/shap_beeswarm.png'
CM_PATH          = 'src/data/confusion_matrix_train.png'

RISK_COLORS = {
    'High':   '#e74c3c',
    'Medium': '#f39c12',
    'Low':    '#2ecc71'
}

TIER_ICONS = {
    'High':   '🔴',
    'Medium': '🟡',
    'Low':    '🟢'
}


# Caching
@st.cache_resource
def load_predictor() -> ChurnPredictor:
    """Load and cache ChurnPredictor (loaded once per session)."""
    return ChurnPredictor(threshold=0.5)


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    """Load and cache processed dataset."""
    return pd.read_csv(PROCESSED_PATH)


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """Load and cache raw dataset."""
    return pd.read_csv(RAW_PATH)


@st.cache_data
def load_benchmark() -> pd.DataFrame:
    """Load and cache model benchmark results."""
    return pd.read_csv(BENCHMARK_PATH, index_col=0)


@st.cache_data
def run_batch_on_processed() -> pd.DataFrame:
    """Run batch prediction on full processed dataset (cached)."""
    predictor = load_predictor()
    df        = load_processed_data()
    return predictor.predict_batch(df, save_path=None)


# Sidebar
def render_sidebar() -> str:
    """Render sidebar navigation and return selected page."""
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/combo-chart.png",
        width=60
    )
    st.sidebar.title("Churn Predictor")
    st.sidebar.markdown("*XGBoost + SHAP Explainability*")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        options=[
            "Overview",
            "Single Prediction",
            "Batch Prediction",
            "Model Performance",
            "Business Insights"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Threshold**\n\n"
        "Controls the decision boundary.\n"
        "Lower = more churners flagged."
    )
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Author:** Jagmohan Prajapat\n\n"
        "[GitHub](https://github.com/Jagmohan-Prajapati) | "
        "[LinkedIn](https://www.linkedin.com/in/jagmohan-prajapati-aaa117200/)"
    )

    return page, threshold


# Page 1: Overview
def page_overview(batch_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    st.title("Customer Churn Overview")
    st.markdown(
        "Dataset-level churn statistics and risk distribution "
        "across all **7,043 customers**."
    )

    # KPI Cards
    total      = len(batch_df)
    n_churn    = int(batch_df['predicted_label'].sum())
    n_high     = int((batch_df['risk_tier'] == 'High').sum())
    n_medium   = int((batch_df['risk_tier'] == 'Medium').sum())
    n_low      = int((batch_df['risk_tier'] == 'Low').sum())
    avg_prob   = batch_df['churn_probability'].mean()
    rev_risk   = batch_df.loc[
        batch_df['predicted_label'] == 1, 'churn_probability'
    ].count()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",    f"{total:,}")
    c2.metric("Predicted Churners", f"{n_churn:,}",
              f"{n_churn/total*100:.1f}%")
    c3.metric("High Risk",       f"{n_high:,}",
              f"{n_high/total*100:.1f}%")
    c4.metric("Medium Risk",     f"{n_medium:,}",
              f"{n_medium/total*100:.1f}%")
    c5.metric("Low Risk",        f"{n_low:,}",
              f"{n_low/total*100:.1f}%")

    st.markdown("---")

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Tier Distribution")
        tier_counts = batch_df['risk_tier'].value_counts()
        tier_counts = tier_counts.reindex(['High', 'Medium', 'Low'])

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [RISK_COLORS[t] for t in tier_counts.index]
        bars = ax.bar(tier_counts.index, tier_counts.values, color=colors, width=0.5)
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customers by Risk Tier")
        for bar, val in zip(bars, tier_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{val:,}", ha='center', fontweight='bold'
            )
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            batch_df['churn_probability'],
            bins=40, color='steelblue',
            edgecolor='white', alpha=0.85
        )
        ax.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Distribution of Predicted Churn Probabilities")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # Charts Row 2
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn Rate by Contract Type")
        contract_map = {
            'Contract_Month-to-month': 'Month-to-month',
            'Contract_One year':       'One year',
            'Contract_Two year':       'Two year'
        }

        target_col = None
        if 'Churn' in batch_df.columns:
            target_col = 'Churn'
        elif 'actual_churn' in batch_df.columns:
            target_col = 'actual_churn'

        if target_col:
            contract_churn = {}
            for col, label in contract_map.items():
                if col in batch_df.columns:
                    # handle both bool (True/False) and int (1/0)
                    mask = batch_df[col].astype(bool)
                    rate = batch_df.loc[mask, target_col].mean() * 100
                    contract_churn[label] = round(rate, 1)

            if contract_churn:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors  = ['#e74c3c', '#f39c12', '#2ecc71']
                bars    = ax.bar(
                    list(contract_churn.keys()),
                    list(contract_churn.values()),
                    color=colors, width=0.5
                )
                ax.set_ylabel("Churn Rate (%)")
                ax.set_title("Actual Churn Rate by Contract Type")
                ax.set_ylim(0, 60)
                for bar, val in zip(bars, contract_churn.values()):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val}%", ha='center', fontweight='bold'
                    )
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Contract columns not found in dataset.")
        else:
            st.info("Actual churn labels not available for this chart.")

    with col4:
        st.subheader("Avg Churn Probability by Tenure Group")
        tenure_cols = {
            'tenure_group_0-1yr': '0–1 yr',
            'tenure_group_1-2yr': '1–2 yr',
            'tenure_group_2-4yr': '2–4 yr',
            'tenure_group_4-5yr': '4–5 yr',
            'tenure_group_5-6yr': '5–6 yr',
        }
        tenure_probs = {}
        for col, label in tenure_cols.items():
            if col in batch_df.columns:
                mask = batch_df[col] == 1
                avg  = batch_df.loc[mask, 'churn_probability'].mean()
                tenure_probs[label] = round(avg, 3)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            tenure_probs.keys(),
            tenure_probs.values(),
            color='steelblue', width=0.5
        )
        ax.set_ylabel("Avg Churn Probability")
        ax.set_title("Avg Churn Probability by Tenure Group")
        ax.set_ylim(0, 0.6)
        st.pyplot(fig)
        plt.close()

    # Top 10 Highest Risk
    st.markdown("---")
    st.subheader("Top 10 Highest-Risk Customers")
    cols_show = ['churn_probability', 'risk_tier', 'predicted_label']
    if 'Churn' in batch_df.columns:
        cols_show = ['Churn'] + cols_show
    top10 = (
        batch_df[cols_show]
        .sort_values('churn_probability', ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={
            'index':             'customer_id',
            'Churn':             'actual_churn',
            'churn_probability': 'churn_prob',
            'predicted_label':   'predicted',
            'risk_tier':         'risk_tier'
        })
    )
    top10['churn_prob'] = top10['churn_prob'].map('{:.1%}'.format)
    st.dataframe(top10, use_container_width=True)


# Page 2: Single Prediction
def page_single_prediction(predictor: ChurnPredictor, threshold: float) -> None:
    predictor.set_threshold(threshold)

    st.title("🔮 Single Customer Churn Prediction")
    st.markdown(
        "Fill in customer details below and click **Predict** to get "
        "an instant churn probability with risk assessment."
    )

    with st.form("prediction_form"):
        st.subheader("Customer Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender     = st.selectbox("Gender", ["Male", "Female"])
            senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner    = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with col2:
            st.markdown("**Account Info**")
            tenure     = st.slider("Tenure (months)", 0, 72, 12)
            contract   = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"]
            )
            payment    = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            paperless  = st.selectbox("Paperless Billing", ["Yes", "No"])

        with col3:
            st.markdown("**Services & Charges**")
            internet       = st.selectbox(
                "Internet Service",
                ["Fiber optic", "DSL", "No"]
            )
            monthly        = st.slider("Monthly Charges ($)", 10, 120, 65)
            total          = st.slider("Total Charges ($)", 10, 9000, 1500)
            online_sec     = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            tech_support   = st.selectbox("Tech Support",    ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        # Build feature vector
        # Tenure group
        if   tenure <= 12: tg = '0-1yr'
        elif tenure <= 24: tg = '1-2yr'
        elif tenure <= 48: tg = '2-4yr'
        elif tenure <= 60: tg = '4-5yr'
        else:              tg = '5-6yr'

        charge_ratio  = monthly / (total + 1)
        service_score = sum([
            online_sec    == 'Yes',
            tech_support  == 'Yes',
        ])

        customer = {
            # Numeric
            'tenure':           tenure,
            'MonthlyCharges':   monthly,
            'TotalCharges':     total,
            'SeniorCitizen':    1 if senior     == 'Yes' else 0,
            'Partner':          1 if partner    == 'Yes' else 0,
            'Dependents':       1 if dependents == 'Yes' else 0,
            'PhoneService':     1,
            'PaperlessBilling': 1 if paperless  == 'Yes' else 0,
            'gender':           1 if gender     == 'Male' else 0,
            'charge_ratio':     round(charge_ratio, 4),
            'service_score':    service_score,

            # Contract
            'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
            'Contract_One year':       1 if contract == 'One year'        else 0,
            'Contract_Two year':       1 if contract == 'Two year'        else 0,

            # Internet
            'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
            'InternetService_DSL':         1 if internet == 'DSL'         else 0,
            'InternetService_No':          1 if internet == 'No'          else 0,

            # Online Security
            'OnlineSecurity_No':                  1 if online_sec == 'No'                   else 0,
            'OnlineSecurity_Yes':                 1 if online_sec == 'Yes'                  else 0,
            'OnlineSecurity_No internet service': 1 if online_sec == 'No internet service'  else 0,

            # Tech Support
            'TechSupport_No':                     1 if tech_support == 'No'                  else 0,
            'TechSupport_Yes':                    1 if tech_support == 'Yes'                 else 0,
            'TechSupport_No internet service':    1 if tech_support == 'No internet service' else 0,

            # Payment
            'PaymentMethod_Electronic check':        1 if payment == 'Electronic check'          else 0,
            'PaymentMethod_Mailed check':            1 if payment == 'Mailed check'               else 0,
            'PaymentMethod_Bank transfer (automatic)':1 if payment == 'Bank transfer (automatic)' else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)'   else 0,

            # Tenure group
            'tenure_group_0-1yr': 1 if tg == '0-1yr' else 0,
            'tenure_group_1-2yr': 1 if tg == '1-2yr' else 0,
            'tenure_group_2-4yr': 1 if tg == '2-4yr' else 0,
            'tenure_group_4-5yr': 1 if tg == '4-5yr' else 0,
            'tenure_group_5-6yr': 1 if tg == '5-6yr' else 0,
        }

        result = predictor.predict_single(customer, verbose=False)
        prob   = result['churn_probability']
        tier   = result['risk_tier']
        dec    = result['decision']

        # Result Display
        st.markdown("---")
        st.subheader("Prediction Result")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Decision",          dec)
        r2.metric("Churn Probability", f"{prob:.1%}")
        r3.metric("Risk Tier",         f"{TIER_ICONS[tier]}  {tier}")
        r4.metric("Threshold Used",    f"{threshold:.2f}")

        # Colour-coded result banner
        color = RISK_COLORS[tier]
        st.markdown(
            f"""
            <div style="
                background-color:{color}22;
                border-left: 5px solid {color};
                padding: 16px 20px;
                border-radius: 6px;
                margin-top: 12px;
            ">
                <h4 style="color:{color}; margin:0;">
                    {TIER_ICONS[tier]} {dec} — {prob:.1%} churn probability
                </h4>
                <p style="margin:6px 0 0 0; color:#555;">
                    This customer is in the <strong>{tier} Risk</strong> tier.
                    {"Immediate retention action recommended." if tier == "High"
                     else "Monitor and engage proactively." if tier == "Medium"
                     else "Customer is stable — focus on upsell opportunities."}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Input Summary
        st.markdown("---")
        st.subheader("Input Summary")
        summary = {
            'Tenure':           f"{tenure} months",
            'Contract':         contract,
            'Monthly Charges':  f"${monthly}",
            'Total Charges':    f"${total}",
            'Internet Service': internet,
            'Online Security':  online_sec,
            'Tech Support':     tech_support,
            'Payment Method':   payment,
            'Charge Ratio':     f"{charge_ratio:.4f}",
            'Service Score':    service_score,
            'Tenure Group':     tg
        }
        summary_df = pd.DataFrame(
            summary.items(),
            columns=['Feature', 'Value']
        )
        st.dataframe(summary_df, use_container_width=True)



# Page 3: Batch Prediction
def page_batch_prediction(predictor: ChurnPredictor, threshold: float) -> None:
    predictor.set_threshold(threshold)

    st.title("Batch Customer Churn Prediction")
    st.markdown(
        "Upload a **processed CSV file** to score all customers at once. "
        "Download the results with churn probabilities and risk tiers attached."
    )

    # File Upload
    st.subheader("Upload Customer Data")
    uploaded = st.file_uploader(
        label       = "Upload processed CSV (same format as processed_churn.csv)",
        type        = ["csv"],
        help        = "File must contain the same 47 feature columns used during training."
    )

    # Use full dataset as default if no upload
    if uploaded is None:
        st.info(
            "No file uploaded — showing results on the **full processed dataset** "
            "(7,043 customers). Upload your own CSV above to score new customers."
        )
        batch_df = run_batch_on_processed()
        source   = "Full Processed Dataset (7,043 customers)"
    else:
        try:
            df       = pd.read_csv(uploaded)
            batch_df = predictor.predict_batch(df, save_path=None)
            source   = uploaded.name
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return

    st.success(f"Predictions complete — source: **{source}**")
    st.markdown("---")

    # KPI Summary
    total    = len(batch_df)
    n_churn  = int(batch_df['predicted_label'].sum())
    n_high   = int((batch_df['risk_tier'] == 'High').sum())
    n_medium = int((batch_df['risk_tier'] == 'Medium').sum())
    n_low    = int((batch_df['risk_tier'] == 'Low').sum())

    st.subheader("Batch Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",    f"{total:,}")
    c2.metric("Predicted Churners", f"{n_churn:,}",
              f"{n_churn/total*100:.1f}%")
    c3.metric("High Risk",       f"{n_high:,}",
              f"{n_high/total*100:.1f}%")
    c4.metric("Medium Risk",     f"{n_medium:,}",
              f"{n_medium/total*100:.1f}%")
    c5.metric("Low Risk",        f"{n_low:,}",
              f"{n_low/total*100:.1f}%")

    # Evaluation metrics (only if actual labels present)
    if 'Churn' in batch_df.columns or 'actual_churn' in batch_df.columns:
        target_col = 'Churn' if 'Churn' in batch_df.columns else 'actual_churn'
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        y_true = batch_df[target_col]
        y_pred = batch_df['predicted_label']
        y_prob = batch_df['churn_probability']

        st.markdown("---")
        st.subheader("Model Evaluation (vs Actual Labels)")
        e1, e2, e3 = st.columns(3)
        e1.metric("Accuracy",  f"{accuracy_score(y_true, y_pred):.4f}")
        e2.metric("ROC-AUC",   f"{roc_auc_score(y_true, y_prob):.4f}")
        e3.metric("F1-Score",  f"{f1_score(y_true, y_pred):.4f}")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Tier Breakdown")
        tier_counts = (
            batch_df['risk_tier']
            .value_counts()
            .reindex(['High', 'Medium', 'Low'])
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        colors  = [RISK_COLORS[t] for t in tier_counts.index]
        bars    = ax.bar(
            tier_counts.index,
            tier_counts.values,
            color=colors, width=0.5
        )
        ax.set_ylabel("Customers")
        ax.set_title("Customers by Risk Tier")
        for bar, val in zip(bars, tier_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{val:,}", ha='center', fontweight='bold', fontsize=10
            )
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(
            batch_df['churn_probability'],
            bins=40, color='steelblue',
            edgecolor='white', alpha=0.85
        )
        ax.axvline(
            threshold, color='red',
            linestyle='--',
            label=f'Threshold ({threshold})'
        )
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Customers")
        ax.set_title("Predicted Churn Probability Distribution")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Filter by Risk Tier
    st.subheader("Filter & Explore Results")
    selected_tiers = st.multiselect(
        "Filter by Risk Tier",
        options  = ['High', 'Medium', 'Low'],
        default  = ['High'],
        help     = "Select one or more risk tiers to filter the results table."
    )

    display_cols = ['churn_probability', 'risk_tier', 'predicted_label']
    if 'Churn' in batch_df.columns:
        display_cols = ['Churn'] + display_cols
    if 'actual_churn' in batch_df.columns:
        display_cols = ['actual_churn'] + display_cols

    filtered = (
        batch_df[batch_df['risk_tier'].isin(selected_tiers)][display_cols]
        .sort_values('churn_probability', ascending=False)
        .reset_index()
        .rename(columns={'index': 'customer_id'})
    )
    filtered['churn_probability'] = filtered['churn_probability'].map('{:.1%}'.format)

    st.markdown(
        f"Showing **{len(filtered):,}** customers "
        f"in selected tier(s): {', '.join(selected_tiers)}"
    )
    st.dataframe(filtered, use_container_width=True)

    st.markdown("---")

    # Download Results
    st.subheader("Download Results")

    download_df = (
        batch_df[display_cols]
        .reset_index()
        .rename(columns={'index': 'customer_id'})
    )

    csv_bytes = download_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label     = "Download Full Results as CSV",
        data      = csv_bytes,
        file_name = "churn_predictions.csv",
        mime      = "text/csv",
        use_container_width = True
    )


# Page 4: Model Performance
def page_model_performance() -> None:
    st.title("Model Performance")
    st.markdown(
        "Benchmark results across 5 models, confusion matrix, "
        "and ROC-AUC analysis for the tuned XGBoost classifier."
    )

    # Benchmark Table
    st.subheader("Model Benchmark Results")
    st.markdown(
        "All 5 models trained on the same **stratified 80/20 split** "
        "and evaluated on the held-out test set (1,409 customers)."
    )

    if os.path.exists(BENCHMARK_PATH):
        bench_df = load_benchmark()
        bench_df = bench_df.sort_values('ROC-AUC', ascending=False)

        # Highlight best row
        def highlight_best(row):
            is_best = row.name == bench_df['ROC-AUC'].idxmax()
            return ['background-color: #d4edda; font-weight: bold'
                    if is_best else '' for _ in row]

        st.dataframe(
            bench_df.style.apply(highlight_best, axis=1).format('{:.3f}'),
            use_container_width=True
        )
    else:
        st.warning(
            "Benchmark results not found. "
            "Run `python src/train.py` to generate them."
        )
        return

    st.markdown("---")

    # Benchmark Bar Chart
    st.subheader("Visual Comparison")
    col1, col2 = st.columns(2)

    with col1:
        metrics_to_plot = st.multiselect(
            "Select metrics to compare",
            options  = ['Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall'],
            default  = ['Accuracy', 'ROC-AUC', 'F1-Score']
        )

    with col2:
        chart_type = st.radio(
            "Chart type",
            options   = ["Grouped Bar", "Heatmap"],
            horizontal= True
        )

    if metrics_to_plot:
        if chart_type == "Grouped Bar":
            plot_df = (
                bench_df[metrics_to_plot]
                .reset_index()
                .melt(id_vars='Model', var_name='Metric', value_name='Score')
            )
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.barplot(
                data    = plot_df,
                x       = 'Model',
                y       = 'Score',
                hue     = 'Metric',
                palette = 'Set2',
                ax      = ax
            )
            ax.set_title(
                'Model Comparison',
                fontsize=13, fontweight='bold'
            )
            ax.set_ylabel('Score')
            ax.set_xlabel('')
            ax.set_ylim(0.4, 1.0)
            ax.tick_params(axis='x', rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:  # Heatmap
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(
                bench_df[metrics_to_plot],
                annot   = True,
                fmt     = '.3f',
                cmap    = 'YlGn',
                ax      = ax,
                linewidths = 0.5
            )
            ax.set_title(
                'Model Performance Heatmap',
                fontsize=13, fontweight='bold'
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # Confusion Matrix
    st.subheader("Confusion Matrix — XGBoost (Tuned)")
    st.markdown(
        "Evaluated on **1,409 held-out test customers** "
        "using decision threshold = **0.50**."
    )

    col3, col4 = st.columns([1, 1])

    with col3:
        if os.path.exists(CM_PATH):
            st.image(CM_PATH, use_column_width=True)
        else:
            st.warning(
                "Confusion matrix image not found. "
                "Run `python src/train.py` to generate it."
            )

    with col4:
        st.markdown("#### How to Read the Confusion Matrix")
        st.markdown(
            """
| Cell | Meaning | Business Impact |
|---|---|---|
| **Top-Left** | True Negatives — correctly predicted No Churn | No action needed |
| **Bottom-Right** | True Positives — correctly predicted Churn | Retention triggered |
| **Top-Right** | False Positives — predicted Churn, actually stayed | Wasted retention spend |
| **Bottom-Left** | False Negatives — predicted No Churn, actually churned | **Most costly — lost customers** |
            """
        )
        st.markdown("---")
        st.markdown("#### Why ROC-AUC Over Accuracy?")
        st.markdown(
            """
- Dataset is **imbalanced** — 73.5% No Churn, 26.5% Churn
- A naive model always predicting "No Churn" scores **73.5% accuracy**
  with zero business value
- **ROC-AUC** measures the model's ability to rank churners above
  non-churners regardless of threshold — far more meaningful
- **F1-Score** balances precision and recall — critical when
  false negatives (missed churners) carry high business cost
            """
        )

    st.markdown("---")

    # Tuned Model Summary Card
    st.subheader("Best Model — XGBoost (Tuned)")

    params_path = 'models/best_params.json'
    if os.path.exists(params_path):
        import json
        with open(params_path) as f:
            best_params = json.load(f)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("n_estimators",  best_params.get('n_estimators',  '—'))
        p2.metric("max_depth",     best_params.get('max_depth',     '—'))
        p3.metric("learning_rate", best_params.get('learning_rate', '—'))
        p4.metric("subsample",     best_params.get('subsample',     '—'))

        st.markdown(
            """
            > Hyperparameters tuned using **GridSearchCV** with **5-fold
            > cross-validation** across 24 candidate combinations (120 total fits).
            > Optimisation metric: **ROC-AUC**.
            """
        )
    else:
        st.warning(
            "Best params file not found at models/best_params.json. "
            "Run `python src/train.py` to generate it."
        )

    st.markdown("---")

    # Train vs Test ROC-AUC Comparison
    st.subheader("Train vs Test Performance — Overfitting Check")
    st.markdown(
        "Comparing CV ROC-AUC during tuning vs held-out test ROC-AUC "
        "to verify the model generalises well."
    )

    cv_auc   = 0.8488   # from train.py output
    test_auc = 0.8490   # from train.py output
    gap      = abs(test_auc - cv_auc)

    g1, g2, g3 = st.columns(3)
    g1.metric("CV ROC-AUC (5-fold)",  f"{cv_auc:.4f}")
    g2.metric("Test ROC-AUC",         f"{test_auc:.4f}")
    g3.metric("Gap",                  f"{gap:.4f}",
              "No overfitting" if gap < 0.02 else "⚠️ Possible overfit")

    st.success(
        "CV AUC and Test AUC are virtually identical (gap = 0.0002) — "
        "the model generalises well with no signs of overfitting."
    )


#  Page 5: Business Insights
def page_business_insights(batch_df: pd.DataFrame) -> None:
    st.title("💡 Business Insights & Retention Strategy")
    st.markdown(
        "SHAP-based feature importance analysis translated into "
        "actionable retention recommendations for business teams."
    )

    # SHAP Feature Importance 
    st.subheader("Top Churn Drivers (SHAP Analysis)")
    st.markdown(
        "Features ranked by **Mean |SHAP Value|** — higher value means "
        "stronger influence on churn prediction."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if os.path.exists(SHAP_BAR_PATH):
            st.image(SHAP_BAR_PATH, use_column_width=True)
        else:
            st.warning(
                "⚠️ SHAP bar chart not found. "
                "Run `notebooks/04_shap_analysis.ipynb` to generate it."
            )

    with col2:
        shap_data = {
            'Rank': list(range(1, 11)),
            'Feature': [
                'Contract_Month-to-month',
                'charge_ratio',
                'OnlineSecurity_No',
                'MonthlyCharges',
                'InternetService_Fiber optic',
                'TechSupport_No',
                'PaymentMethod_Electronic check',
                'PaperlessBilling',
                'tenure',
                'Contract_Two year'
            ],
            'Mean |SHAP|': [
                0.6919, 0.3442, 0.2404,
                0.2293, 0.1902, 0.1678,
                0.1569, 0.1132, 0.1108, 0.0912
            ],
            'Direction': [
                'Increases churn',
                'Increases churn',
                'Increases churn',
                'Increases churn',
                'Increases churn',
                'Increases churn',
                'Increases churn',
                'Increases churn',
                'Decreases churn',
                'Decreases churn',
            ]
        }
        shap_df = pd.DataFrame(shap_data).set_index('Rank')
        st.dataframe(shap_df, use_container_width=True)

    st.markdown("---")

    # SHAP Beeswarm
    st.subheader("SHAP Beeswarm Plot")
    st.markdown(
        "Each dot = one customer. "
        "**Red dots** = high feature value. "
        "**Blue dots** = low feature value. "
        "**X-axis position** = SHAP value (positive = pushes toward churn)."
    )
    if os.path.exists(SHAP_BEESWARM):
        st.image(SHAP_BEESWARM, use_column_width=True)
    else:
        st.warning(
            "SHAP beeswarm not found. "
            "Run `notebooks/04_shap_analysis.ipynb` to generate it."
        )

    st.markdown("---")

    # Revenue at Risk
    st.subheader("Revenue at Risk")

    if 'churn_probability' in batch_df.columns:
        monthly_col = None
        for c in batch_df.columns:
            if 'MonthlyCharge' in c or 'monthly' in c.lower():
                monthly_col = c
                break

        if monthly_col:
            predicted_churners  = batch_df[batch_df['predicted_label'] == 1]
            monthly_rev_at_risk = predicted_churners[monthly_col].sum()
            ltv_at_risk         = monthly_rev_at_risk * 24
            retained_20         = ltv_at_risk * 0.20
            n_churners          = len(predicted_churners)

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Predicted Churners",       f"{n_churners:,}")
            r2.metric("Monthly Revenue at Risk",  f"${monthly_rev_at_risk:,.0f}")
            r3.metric("24-Month LTV at Risk",     f"${ltv_at_risk:,.0f}")
            r4.metric("Revenue Saved (20% Ret.)", f"${retained_20:,.0f}",
                      "at $50/customer incentive")
        else:
            # Fallback static numbers from business report
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Predicted Churners",       "1,432")
            r2.metric("Monthly Revenue at Risk",  "~$103,000")
            r3.metric("24-Month LTV at Risk",     "~$2,470,000")
            r4.metric("Revenue Saved (20% Ret.)", "~$494,000")

    st.markdown("---")

    # Risk Tier Breakdown
    st.subheader("Customer Risk Segmentation")

    col3, col4 = st.columns(2)

    with col3:
        tier_counts = (
            batch_df['risk_tier']
            .value_counts()
            .reindex(['High', 'Medium', 'Low'])
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        wedge_colors = [RISK_COLORS['High'],
                        RISK_COLORS['Medium'],
                        RISK_COLORS['Low']]
        ax.pie(
            tier_counts.values,
            labels     = tier_counts.index,
            colors     = wedge_colors,
            autopct    = '%1.1f%%',
            startangle = 90,
            wedgeprops = dict(edgecolor='white', linewidth=2)
        )
        ax.set_title('Risk Tier Distribution', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("#### Risk Tier Definitions")
        st.markdown(
            """
| Tier | Probability | Action |
|---|---|---|
|  **High** | 60–100% | Immediate intervention — personalised offer, contract upgrade |
|  **Medium** | 30–60% | Proactive engagement — outreach call, add-on trial |
|  **Low** | 0–30% | Maintain satisfaction — upsell, loyalty programme |
            """
        )
        st.markdown("---")
        st.markdown("#### Key Finding")
        st.info(
            " `charge_ratio` — an **engineered feature** created during "
            "preprocessing — ranked as the **#2 churn predictor** (SHAP = 0.344), "
            "outranking raw features like tenure and TotalCharges. "
            "This validates the feature engineering phase."
        )

    st.markdown("---")

    # Retention Recommendations
    st.subheader("Retention Strategy Recommendations")

    tab1, tab2, tab3 = st.tabs([
        "High Risk (Priority 1)",
        "Medium Risk (Priority 2)",
        "Low Risk (Priority 3)"
    ])

    with tab1:
        st.markdown(
            """
### High Risk — Immediate Action Required
*Churn probability > 60% — highest immediate revenue risk*

| Intervention | Target Segment | Expected Impact |
|---|---|---|
| **Contract upgrade offer** | Month-to-month customers > 6 months tenure | Highest ROI — directly addresses #1 SHAP driver |
| **Personalised loyalty discount** | High `charge_ratio` customers (recent price spike) | Addresses #2 SHAP driver directly |
| **Free security add-on trial (3 months)** | No OnlineSecurity + No TechSupport | Reduces 2 top-5 risk factors simultaneously |
| **Dedicated account manager outreach** | High-charge fiber optic customers | Addresses competitive vulnerability |

> **Budget recommendation:** Allocate **$50/customer retention incentive**
> to top 20% highest-risk customers (~917 customers).
> At 20% retention rate → **$494,000+ revenue saved over 24 months**.
            """
        )

    with tab2:
        st.markdown(
            """
### Medium Risk — Proactive Engagement
*Churn probability 30–60% — prevent escalation to high risk*

| Intervention | Target Segment | Expected Impact |
|---|---|---|
| **Proactive outreach call** | Fiber optic customers with no add-ons | Addresses competitive vulnerability before it worsens |
| **Electronic check → auto-pay migration** | PaymentMethod = Electronic check | Reduces payment-linked churn signal |
| **Paperless billing + loyalty points** | Tenure < 12 months | Year-1 retention is the most critical window |
| **Service bundle trial offer** | `service_score` of 0–1 | Increase engagement and switching cost |

> **Key insight:** Customers in their **first 12 months** are the highest
> escalation risk. Early engagement programmes in this window have the
> highest long-term retention ROI.
            """
        )

    with tab3:
        st.markdown(
            """
### Low Risk — Maintain & Grow
*Churn probability < 30% — stable customers, focus on LTV growth*

| Intervention | Target Segment | Expected Impact |
|---|---|---|
| **Annual contract renewal reminder** | Two-year contract customers nearing expiry | Lock in for another term before risk escalates |
| **Service bundle upgrade offer** | Long tenure, low MonthlyCharges | Increase ARPU without churn risk |
| **Referral programme invitation** | Highest satisfaction, longest tenure | Turn loyal customers into brand advocates |
| **Early access to new features** | Low-charge, long-tenure customers | Deepen engagement and product dependency |
            """
        )

    st.markdown("---")

    # Individual Customer Insights
    st.subheader("Highest vs Lowest Risk — Individual Profiles")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown(
            """
            <div style="
                background-color: #e74c3c22;
                border-left: 5px solid #e74c3c;
                padding: 16px;
                border-radius: 6px;
            ">
            <h4 style="color:#e74c3c;">Highest-Risk Customer</h4>
            <p><strong>Customer ID:</strong> 2208</p>
            <p><strong>Churn Probability:</strong> 91.3%</p>
            <p><strong>Actual Label:</strong> Churned (model correct)</p>
            <hr>
            <p><strong>Key Risk Drivers:</strong></p>
            <ul>
                >Month-to-month contract (+highest SHAP)</li>
                >High charge_ratio — recent price spike</li>
                >No OnlineSecurity</li>
                >No TechSupport</li>
                >Tenure &lt; 12 months</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col6:
        st.markdown(
            """
            <div style="
                background-color: #2ecc7122;
                border-left: 5px solid #2ecc71;
                padding: 16px;
                border-radius: 6px;
            ">
            <h4 style="color:#2ecc71;">Lowest-Risk Customer</h4>
            <p><strong>Customer ID:</strong> 1843</p>
            <p><strong>Churn Probability:</strong> 0.9%</p>
            <p><strong>Actual Label:</strong> No Churn (model correct)</p>
            <hr>
            <p><strong>Key Retention Factors:</strong></p>
            <ul>
                >Two-year contract (strongest protection)</li>
                >Long tenure (60+ months)</li>
                >Low monthly charges</li>
                >Has multiple add-on services</li>
                >Low charge_ratio — stable pricing history</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


# Main App Entry Point 
def main() -> None:
    """
    Main entry point — renders sidebar navigation and routes to selected page.
    """

    # Load shared resources 
    predictor  = load_predictor()
    batch_df   = run_batch_on_processed()

    # Sidebar + page selection
    page, threshold = render_sidebar()

    # Route to selected page
    if page == "Overview":
        raw_df = load_raw_data()
        page_overview(batch_df, raw_df)

    elif page == "Single Prediction":
        page_single_prediction(predictor, threshold)

    elif page == "Batch Prediction":
        page_batch_prediction(predictor, threshold)

    elif page == "Model Performance":
        page_model_performance()

    elif page == "Business Insights":
        page_business_insights(batch_df)



if __name__ == '__main__':
    main()