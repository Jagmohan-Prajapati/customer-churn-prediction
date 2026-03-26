"""
predict.py
──────────
Reusable prediction utilities for the Customer Churn Prediction project.

Loads saved model artifacts and provides:
  - Single customer churn prediction
  - Batch prediction on a CSV file
  - Risk tier classification
  - Human-readable prediction report

Usage:
    # From project root (with venv activated):
    python src/predict.py

    # Or import in app.py / notebooks:
    from src.predict import ChurnPredictor
    predictor = ChurnPredictor()
    result = predictor.predict_single(customer_data)
    batch_df = predictor.predict_batch('data/processed_churn.csv')
"""

import os
import json
import logging
import joblib
import warnings
import numpy as np
import pandas as pd

from typing import Union

warnings.filterwarnings('ignore')

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


#  Constants
MODELS_DIR = 'models'

MODEL_PATH    = os.path.join(MODELS_DIR, 'xgboost_best_model.pkl')
SCALER_PATH   = os.path.join(MODELS_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_names.pkl')
PARAMS_PATH   = os.path.join(MODELS_DIR, 'best_params.json')

# Risk tier thresholds
RISK_THRESHOLDS = {
    'Low':    (0.00, 0.30),
    'Medium': (0.30, 0.60),
    'High':   (0.60, 1.00)
}

# Default decision threshold
DEFAULT_THRESHOLD = 0.50


#  Predictor Class
class ChurnPredictor:
    """
    Loads trained artifacts and provides churn prediction utilities.

    Supports:
        - Single customer prediction (dict input)
        - Batch prediction (processed CSV or DataFrame)
        - Risk tier classification (Low / Medium / High)
        - Threshold adjustment for recall-precision tradeoff
        - Human-readable prediction summary

    Attributes:
        model         : Loaded XGBoost estimator
        scaler        : Loaded StandardScaler
        feature_names : List of expected feature column names
        best_params   : Best hyperparameters used during training
        threshold     : Decision threshold for binary classification
        is_loaded     : Whether artifacts have been loaded successfully
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.model         = None
        self.scaler        = None
        self.feature_names = []
        self.best_params   = {}
        self.threshold     = threshold
        self.is_loaded     = False
        self._load_artifacts()

    #  Public API 

    def predict_single(
        self,
        customer: dict,
        verbose: bool = True
    ) -> dict:
        """
        Predict churn probability for a single customer.

        Args:
            customer (dict): Feature values keyed by feature name.
                             Missing features are filled with 0.
            verbose  (bool): If True, print a human-readable report.

        Returns:
            dict: {
                'churn_probability' : float,
                'predicted_label'   : int   (0 or 1),
                'risk_tier'         : str   ('Low' | 'Medium' | 'High'),
                'decision'          : str   ('CHURN' | 'NO CHURN'),
                'threshold_used'    : float
            }
        """
        self._check_loaded()

        # Build feature vector — fill missing features with 0
        row = {feat: customer.get(feat, 0) for feat in self.feature_names}
        df  = pd.DataFrame([row])

        prob  = self._get_probabilities(df)[0]
        label = int(prob >= self.threshold)
        tier  = self._get_risk_tier(prob)

        result = {
            'churn_probability': round(float(prob), 4),
            'predicted_label':   label,
            'risk_tier':         tier,
            'decision':          'CHURN' if label == 1 else 'NO CHURN',
            'threshold_used':    self.threshold
        }

        if verbose:
            self._print_single_report(result)

        return result

    def predict_batch(
        self,
        source: Union[str, pd.DataFrame],
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Predict churn probability for a batch of customers.

        Args:
            source    (str | pd.DataFrame): Path to processed CSV or DataFrame.
            save_path (str, optional)     : If provided, saves results to this path.

        Returns:
            pd.DataFrame: Original data with 3 new columns added:
                          churn_probability | predicted_label | risk_tier
        """
        self._check_loaded()

        df = self._load_source(source)
        logger.info(f"Batch prediction — {len(df):,} customers")

        # Drop target if present (inference mode)
        target = None
        if 'Churn' in df.columns:
            target = df['Churn'].copy()
            df     = df.drop(columns=['Churn'])
            logger.info("  'Churn' column detected and held out for evaluation")

        # Align features
        df_aligned = self._align_features(df)

        # Predict
        probs  = self._get_probabilities(df_aligned)
        labels = (probs >= self.threshold).astype(int)
        tiers  = [self._get_risk_tier(p) for p in probs]

        # Attach results
        results_df = df_aligned.copy()
        if target is not None:
            results_df['actual_churn'] = target.values
        results_df['churn_probability'] = probs.round(4)
        results_df['predicted_label']   = labels
        results_df['risk_tier']         = tiers

        # Summary
        self._print_batch_summary(results_df, target)

        if save_path:
            results_df.to_csv(save_path, index=False)
            logger.info(f"  Results saved → {save_path}")

        return results_df

    def set_threshold(self, threshold: float) -> None:
        """
        Update the decision threshold.

        Args:
            threshold (float): Value between 0.0 and 1.0.
                               Lower = more churners flagged (higher recall).
                               Higher = fewer false positives (higher precision).
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0 (exclusive)")
        old = self.threshold
        self.threshold = threshold
        logger.info(f"Threshold updated: {old} → {threshold}")

    def get_top_risk_customers(
        self,
        batch_df: pd.DataFrame,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Return the top N highest-risk customers from a batch result.

        Args:
            batch_df (pd.DataFrame): Output of predict_batch()
            n        (int)          : Number of top customers to return

        Returns:
            pd.DataFrame: Top N rows sorted by churn_probability descending
        """
        if 'churn_probability' not in batch_df.columns:
            raise ValueError(
                "Input must be the output of predict_batch(). "
                "'churn_probability' column not found."
            )
        cols = ['churn_probability', 'risk_tier', 'predicted_label']
        if 'actual_churn' in batch_df.columns:
            cols.insert(0, 'actual_churn')

        return (
            batch_df[cols]
            .sort_values('churn_probability', ascending=False)
            .head(n)
            .reset_index()
            .rename(columns={'index': 'customer_id'})
        )

    def model_info(self) -> None:
        """Print a summary of the loaded model and its configuration."""
        self._check_loaded()
        print("\n── Model Info ───────────────────────────────────")
        print(f"  Model type      : {type(self.model).__name__}")
        print(f"  Features        : {len(self.feature_names)}")
        print(f"  Threshold       : {self.threshold}")
        print(f"  Best params     :")
        for k, v in self.best_params.items():
            print(f"    {k:<20} {v}")
        print("─────────────────────────────────────────────────\n")

    #  Private Helpers

    def _load_artifacts(self) -> None:
        """Load model, scaler, feature names, and best params from models/."""
        required = {
            'model':    MODEL_PATH,
            'scaler':   SCALER_PATH,
            'features': FEATURES_PATH
        }

        missing = [k for k, v in required.items() if not os.path.exists(v)]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts: {missing}\n"
                f"Run 'python src/train.py' first to generate them."
            )

        self.model         = joblib.load(MODEL_PATH)
        self.scaler        = joblib.load(SCALER_PATH)
        self.feature_names = joblib.load(FEATURES_PATH)

        if os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH) as f:
                self.best_params = json.load(f)

        self.is_loaded = True
        logger.info(
            f"Artifacts loaded — "
            f"model: {type(self.model).__name__} | "
            f"features: {len(self.feature_names)}"
        )

    def _get_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """Scale features and return churn probabilities."""
        X_sc = self.scaler.transform(df)
        return self.model.predict_proba(X_sc)[:, 1]

    def _get_risk_tier(self, prob: float) -> str:
        """Map a churn probability to a risk tier label."""
        for tier, (low, high) in RISK_THRESHOLDS.items():
            if low <= prob < high:
                return tier
        return 'High'   # catches prob == 1.0

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align input DataFrame columns to match training feature names.
        Missing columns are filled with 0, extra columns are dropped.
        """
        missing_cols = set(self.feature_names) - set(df.columns)
        extra_cols   = set(df.columns) - set(self.feature_names)

        if missing_cols:
            logger.warning(
                f"  {len(missing_cols)} missing features filled with 0: "
                f"{list(missing_cols)[:5]}{'...' if len(missing_cols) > 5 else ''}"
            )
        if extra_cols:
            logger.warning(
                f"  {len(extra_cols)} extra columns dropped: "
                f"{list(extra_cols)[:5]}{'...' if len(extra_cols) > 5 else ''}"
            )

        return df.reindex(columns=self.feature_names, fill_value=0)

    def _load_source(
        self,
        source: Union[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Load DataFrame from path or pass through if already a DataFrame."""
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        return pd.read_csv(source)

    def _print_single_report(self, result: dict) -> None:
        """Print a clean single-customer prediction report."""
        tier_icons = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
        icon = tier_icons.get(result['risk_tier'], '⚪')

        print("\n── Churn Prediction Report ──────────────────────")
        print(f"  Decision          : {result['decision']}")
        print(f"  Churn Probability : {result['churn_probability']:.1%}")
        print(f"  Risk Tier         : {icon}  {result['risk_tier']}")
        print(f"  Threshold Used    : {result['threshold_used']}")
        print("─────────────────────────────────────────────────\n")

    def _print_batch_summary(
        self,
        results_df: pd.DataFrame,
        target: pd.Series = None
    ) -> None:
        """Print a summary of batch prediction results."""
        total    = len(results_df)
        n_high   = (results_df['risk_tier'] == 'High').sum()
        n_medium = (results_df['risk_tier'] == 'Medium').sum()
        n_low    = (results_df['risk_tier'] == 'Low').sum()
        n_churn  = results_df['predicted_label'].sum()

        print("\n── Batch Prediction Summary ─────────────────────")
        print(f"  Total customers   : {total:,}")
        print(f"  Predicted churn   : {n_churn:,} ({n_churn/total*100:.1f}%)")
        print(f"  High risk      : {n_high:,} ({n_high/total*100:.1f}%)")
        print(f"  Medium risk    : {n_medium:,} ({n_medium/total*100:.1f}%)")
        print(f"  Low risk       : {n_low:,} ({n_low/total*100:.1f}%)")
        print(f"  Threshold used    : {self.threshold}")

        if target is not None:
            from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
            probs  = results_df['churn_probability'].values
            labels = results_df['predicted_label'].values
            print(f"\n  ── Evaluation (vs actual labels) ──")
            print(f"  Accuracy          : {accuracy_score(target, labels):.4f}")
            print(f"  ROC-AUC           : {roc_auc_score(target, probs):.4f}")
            print(f"  F1-Score          : {f1_score(target, labels):.4f}")

        print("─────────────────────────────────────────────────\n")

    def _check_loaded(self) -> None:
        """Raise RuntimeError if artifacts have not been loaded."""
        if not self.is_loaded:
            raise RuntimeError(
                "Model artifacts not loaded. "
                "Ensure models/ directory exists and contains required files."
            )


#  Standalone Runner 
if __name__ == '__main__':
    """
    Run prediction pipeline directly from command line:
        python src/predict.py
    """

    #  1. Load predictor 
    predictor = ChurnPredictor(threshold=DEFAULT_THRESHOLD)
    predictor.model_info()

    #  2. Single customer prediction (high-risk profile)
    logger.info("Testing single customer prediction (high-risk profile)...")

    high_risk_customer = {
        'tenure':                          2,
        'MonthlyCharges':                  85.0,
        'TotalCharges':                    170.0,
        'SeniorCitizen':                   0,
        'Partner':                         0,
        'Dependents':                      0,
        'PhoneService':                    1,
        'PaperlessBilling':                1,
        'gender':                          1,
        'charge_ratio':                    0.49,
        'service_score':                   0,
        'Contract_Month-to-month':         1,
        'Contract_One year':               0,
        'Contract_Two year':               0,
        'InternetService_Fiber optic':     1,
        'InternetService_DSL':             0,
        'InternetService_No':              0,
        'OnlineSecurity_No':               1,
        'OnlineSecurity_Yes':              0,
        'OnlineSecurity_No internet service': 0,
        'TechSupport_No':                  1,
        'TechSupport_Yes':                 0,
        'TechSupport_No internet service': 0,
        'PaymentMethod_Electronic check':  1,
        'PaymentMethod_Mailed check':      0,
        'PaymentMethod_Bank transfer (automatic)': 0,
        'PaymentMethod_Credit card (automatic)':   0,
        'tenure_group_0-1yr':              1,
        'tenure_group_1-2yr':              0,
        'tenure_group_2-4yr':              0,
        'tenure_group_4-5yr':              0,
        'tenure_group_5-6yr':              0,
    }

    result_high = predictor.predict_single(high_risk_customer, verbose=True)

    # 3. Single customer prediction (low-risk profile)
    logger.info("Testing single customer prediction (low-risk profile)...")

    low_risk_customer = {
        'tenure':                          60,
        'MonthlyCharges':                  45.0,
        'TotalCharges':                    2700.0,
        'SeniorCitizen':                   0,
        'Partner':                         1,
        'Dependents':                      1,
        'PhoneService':                    1,
        'PaperlessBilling':                0,
        'gender':                          0,
        'charge_ratio':                    0.016,
        'service_score':                   5,
        'Contract_Month-to-month':         0,
        'Contract_One year':               0,
        'Contract_Two year':               1,
        'InternetService_Fiber optic':     0,
        'InternetService_DSL':             1,
        'InternetService_No':              0,
        'OnlineSecurity_No':               0,
        'OnlineSecurity_Yes':              1,
        'OnlineSecurity_No internet service': 0,
        'TechSupport_No':                  0,
        'TechSupport_Yes':                 1,
        'TechSupport_No internet service': 0,
        'PaymentMethod_Electronic check':  0,
        'PaymentMethod_Mailed check':      0,
        'PaymentMethod_Bank transfer (automatic)': 1,
        'PaymentMethod_Credit card (automatic)':   0,
        'tenure_group_0-1yr':              0,
        'tenure_group_1-2yr':              0,
        'tenure_group_2-4yr':              0,
        'tenure_group_4-5yr':              0,
        'tenure_group_5-6yr':              1,
    }

    result_low = predictor.predict_single(low_risk_customer, verbose=True)

    #  4. Threshold adjustment demo 
    logger.info("Testing threshold adjustment (0.35 — higher recall)...")
    predictor.set_threshold(0.35)
    result_adjusted = predictor.predict_single(high_risk_customer, verbose=True)
    predictor.set_threshold(DEFAULT_THRESHOLD)   # reset back

    #  5. Batch prediction on full processed dataset
    PROCESSED_PATH = 'data/processed_churn.csv'

    if os.path.exists(PROCESSED_PATH):
        logger.info(f"Running batch prediction on: {PROCESSED_PATH}")
        batch_results = predictor.predict_batch(
            source    = PROCESSED_PATH,
            save_path = 'data/batch_predictions.csv'
        )

        print("\n── Top 10 Highest-Risk Customers ────────────────")
        top10 = predictor.get_top_risk_customers(batch_results, n=10)
        print(top10.to_string(index=False))
        print("─────────────────────────────────────────────────\n")
    else:
        logger.warning(
            f"Processed data not found at {PROCESSED_PATH}. "
            f"Skipping batch prediction. "
            f"Run 'python src/train.py' first."
        )

    logger.info("predict.py — all tests complete ")