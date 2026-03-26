"""
preprocess.py
─────────────
Reusable preprocessing pipeline for the Customer Churn Prediction project.

Usage:
    from src.preprocess import ChurnPreprocessor

    preprocessor = ChurnPreprocessor()
    df_processed = preprocessor.fit_transform('../data/telco_churn.csv')
    preprocessor.save('../data/processed_churn.csv')
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional

# ── Logging Setup 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ── Constants 
BINARY_COLUMNS = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'PaperlessBilling', 'Churn'
]

BINARY_MAP = {
    'Yes': 1, 'No': 0,
    'Male': 1, 'Female': 0
}

MULTI_CAT_COLUMNS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaymentMethod', 'tenure_group'
]

SERVICE_COLUMNS = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

TENURE_BINS   = [0, 12, 24, 48, 60, 72]
TENURE_LABELS = ['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5-6yr']

DROP_COLUMNS = ['customerID']


# ── Preprocessor Class
class ChurnPreprocessor:
    """
    End-to-end preprocessing pipeline for the Telco Customer Churn dataset.

    Stages:
        1. Load raw CSV
        2. Fix data types
        3. Handle missing values
        4. Drop irrelevant columns
        5. Engineer new features
        6. Encode categorical variables
        7. Validate output

    Attributes:
        df_raw (pd.DataFrame):       Raw loaded dataframe
        df_processed (pd.DataFrame): Fully processed dataframe
        feature_names (list):        Final list of feature column names
        is_fitted (bool):            Whether fit_transform has been called
    """

    def __init__(self) -> None:
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.feature_names: list = []
        self.is_fitted: bool = False

    # ── Public API 

    def fit_transform(self, filepath: str) -> pd.DataFrame:
        """
        Load raw CSV and run full preprocessing pipeline.

        Args:
            filepath (str): Path to raw telco_churn.csv

        Returns:
            pd.DataFrame: Fully processed, model-ready dataframe

        Raises:
            FileNotFoundError: If filepath does not exist
            ValueError: If required columns are missing from the dataset
        """
        logger.info("=" * 55)
        logger.info("  ChurnPreprocessor — Starting pipeline")
        logger.info("=" * 55)

        self.df_raw = self._load(filepath)
        df = self.df_raw.copy()

        df = self._fix_dtypes(df)
        df = self._handle_missing(df)
        df = self._drop_columns(df)
        df = self._engineer_features(df)
        df = self._encode(df)
        df = self._validate(df)

        self.df_processed = df
        self.feature_names = [c for c in df.columns if c != 'Churn']
        self.is_fitted = True

        logger.info("=" * 55)
        logger.info(f"  Pipeline complete — shape: {df.shape}")
        logger.info(f"  Features: {len(self.feature_names)} | Target: Churn")
        logger.info("=" * 55)

        return df

    def save(self, output_path: str) -> None:
        """
        Save processed dataframe to CSV.

        Args:
            output_path (str): Destination path for processed CSV

        Raises:
            RuntimeError: If fit_transform has not been called yet
        """
        self._check_fitted()
        os.makedirs(os.path.dirname(output_path), exist_ok=True) \
            if os.path.dirname(output_path) else None
        self.df_processed.to_csv(output_path, index=False)
        logger.info(f"Processed data saved → {output_path}")

    def get_feature_names(self) -> list:
        """
        Return list of feature column names (excludes target 'Churn').

        Returns:
            list: Feature column names

        Raises:
            RuntimeError: If fit_transform has not been called yet
        """
        self._check_fitted()
        return self.feature_names

    def get_summary(self) -> dict:
        """
        Return a summary dict of the preprocessing results.

        Returns:
            dict: Summary statistics
        """
        self._check_fitted()
        return {
            'raw_shape':        self.df_raw.shape,
            'processed_shape':  self.df_processed.shape,
            'n_features':       len(self.feature_names),
            'n_missing_final':  int(self.df_processed.isnull().sum().sum()),
            'churn_rate':       round(self.df_processed['Churn'].mean() * 100, 2),
            'churn_count':      int(self.df_processed['Churn'].sum()),
            'no_churn_count':   int((self.df_processed['Churn'] == 0).sum()),
        }

    # ── Private Pipeline Steps 

    def _load(self, filepath: str) -> pd.DataFrame:
        """Load raw CSV and validate it exists and has required columns."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Loaded raw data — shape: {df.shape}")

        required = ['customerID', 'tenure', 'MonthlyCharges',
                    'TotalCharges', 'Churn']
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def _fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix known dtype issues in the raw dataset.
        - TotalCharges stored as object due to blank strings → convert to float
        """
        logger.info("Step 1 — Fixing data types")

        before = df['TotalCharges'].dtype
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        after = df['TotalCharges'].dtype

        n_nulls = df['TotalCharges'].isnull().sum()
        logger.info(
            f"  TotalCharges: {before} → {after} "
            f"| {n_nulls} blank strings became NaN"
        )
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values.
        - TotalCharges: fill with median (robust to skew)
        """
        logger.info("Step 2 — Handling missing values")

        total_missing = df.isnull().sum().sum()
        logger.info(f"  Total missing values found: {total_missing}")

        if df['TotalCharges'].isnull().sum() > 0:
            median_val = df['TotalCharges'].median()
            df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
            logger.info(
                f"  TotalCharges — filled {df['TotalCharges'].isnull().sum()} "
                f"NaNs with median: ${median_val:.2f}"
            )

        remaining = df.isnull().sum().sum()
        logger.info(f"  Missing values after imputation: {remaining}")
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that carry no predictive value (e.g., customerID)."""
        logger.info("Step 3 — Dropping irrelevant columns")

        cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        logger.info(f"  Dropped: {cols_to_drop} | Shape now: {df.shape}")
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 3 new features based on domain knowledge and EDA insights:

        1. tenure_group  — lifecycle stage buckets (0–1yr ... 5–6yr)
        2. charge_ratio  — MonthlyCharges / (TotalCharges + 1)
                           detects customers experiencing recent price shocks
        3. service_score — count of add-on services subscribed (0–6)
                           more services = more engaged = lower churn risk
        """
        logger.info("Step 4 — Engineering new features")

        # Feature 1: tenure_group
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=TENURE_BINS,
            labels=TENURE_LABELS
        )
        logger.info(
            f"  tenure_group — distribution:\n"
            f"{df['tenure_group'].value_counts().to_string()}"
        )

        # Feature 2: charge_ratio
        df['charge_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
        logger.info(
            f"  charge_ratio — mean: {df['charge_ratio'].mean():.4f} "
            f"| max: {df['charge_ratio'].max():.4f}"
        )

        # Feature 3: service_score
        available_service_cols = [c for c in SERVICE_COLUMNS if c in df.columns]
        df['service_score'] = df[available_service_cols].apply(
            lambda row: (row == 'Yes').sum(), axis=1
        )
        logger.info(
            f"  service_score — distribution:\n"
            f"{df['service_score'].value_counts().sort_index().to_string()}"
        )

        logger.info(f"  Shape after feature engineering: {df.shape}")
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables:
        - Binary columns (Yes/No, Male/Female) → label encode to 1/0
        - Multi-category columns → one-hot encode with pd.get_dummies
        """
        logger.info("Step 5 — Encoding categorical variables")

        # Binary encoding
        binary_cols_present = [c for c in BINARY_COLUMNS if c in df.columns]
        for col in binary_cols_present:
            df[col] = df[col].map(BINARY_MAP)
        logger.info(f"  Binary encoded: {binary_cols_present}")

        # One-hot encoding
        multi_cat_present = [c for c in MULTI_CAT_COLUMNS if c in df.columns]
        df = pd.get_dummies(df, columns=multi_cat_present, drop_first=False)
        logger.info(f"  One-hot encoded: {multi_cat_present}")
        logger.info(f"  Shape after encoding: {df.shape}")

        return df

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final validation checks before returning processed dataframe:
        - No remaining null values
        - Target column 'Churn' exists
        - All columns are numeric or boolean
        """
        logger.info("Step 6 — Validating processed dataset")

        # Check nulls
        nulls = df.isnull().sum().sum()
        if nulls > 0:
            logger.warning(f"{nulls} null values remain after preprocessing!")
        else:
            logger.info("  No null values")

        # Check target exists
        if 'Churn' not in df.columns:
            raise ValueError("Target column 'Churn' missing after preprocessing")
        logger.info(
            f"  Target 'Churn' present — "
            f"churn rate: {df['Churn'].mean()*100:.1f}%"
        )

        # Check all numeric OR boolean (get_dummies returns bool dtype)
        non_numeric = df.select_dtypes(
            exclude=[np.number, 'bool']      # ← added 'bool'
        ).columns.tolist()
        if non_numeric:
            logger.warning(f"  Non-numeric columns remain: {non_numeric}")
        else:
            logger.info("  All columns are numeric/boolean — model ready")

        return df

    def _check_fitted(self) -> None:
        """Raise RuntimeError if fit_transform has not been called."""
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted yet. "
                "Call fit_transform(filepath) first."
            )


# Standalone Runner
if __name__ == '__main__':
    """
    Run preprocessing pipeline directly from command line:
        python src/preprocess.py
    """
    RAW_PATH  = '../data/telco_churn.csv'
    SAVE_PATH = '../data/processed_churn.csv'

    preprocessor = ChurnPreprocessor()
    df = preprocessor.fit_transform(RAW_PATH)
    preprocessor.save(SAVE_PATH)

    print("\n── Summary ")
    summary = preprocessor.get_summary()
    for key, val in summary.items():
        print(f"  {key:<22} {val}")
    print("────────────")