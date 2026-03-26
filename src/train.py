"""
train.py
────────
Reusable model training pipeline for the Customer Churn Prediction project.

Trains and benchmarks 5 ML models, tunes the best one using GridSearchCV,
and saves all artifacts to the models/ directory.

Usage:
    # From project root (with venv activated):
    python src/train.py

    # Or import in notebooks / app:
    from src.train import ChurnTrainer
    trainer = ChurnTrainer()
    trainer.run('../data/processed_churn.csv')
"""

import os
import json
import logging
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Constants
MODELS_DIR   = 'models'
REPORTS_DIR  = 'data'
RANDOM_STATE = 42
TEST_SIZE    = 0.2

BASELINE_MODELS = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5
    ),
    'XGBoost': XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )
}

XGB_PARAM_GRID = {
    'n_estimators':  [100, 200],
    'max_depth':     [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample':     [0.8, 1.0]
}


# Trainer Class
class ChurnTrainer:
    """
    End-to-end model training pipeline for Customer Churn Prediction.

    Stages:
        1. Load processed data
        2. Split into train / test sets (stratified)
        3. Scale features with StandardScaler
        4. Benchmark 5 baseline models
        5. Tune best model (XGBoost) with GridSearchCV
        6. Evaluate tuned model on held-out test set
        7. Save model artifacts + training report

    Attributes:
        X_train, X_test   : Feature splits (unscaled)
        y_train, y_test   : Target splits
        X_train_sc, X_test_sc : Scaled feature splits
        scaler            : Fitted StandardScaler
        baseline_results  : DataFrame of all model benchmark scores
        best_model        : Tuned XGBoost estimator
        best_params       : Best hyperparameters found by GridSearchCV
        feature_names     : List of feature column names
        is_trained        : Whether run() has been called
    """

    def __init__(self) -> None:
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.X_train_sc = self.X_test_sc = None
        self.scaler: Optional[StandardScaler] = None
        self.baseline_results: Optional[pd.DataFrame] = None
        self.best_model = None
        self.best_params: dict = {}
        self.feature_names: list = []
        self.is_trained: bool = False

    # Public API

    def run(self, processed_filepath: str) -> None:
        """
        Execute the full training pipeline end-to-end.

        Args:
            processed_filepath (str): Path to processed_churn.csv

        Raises:
            FileNotFoundError: If processed_filepath does not exist
        """
        logger.info("=" * 55)
        logger.info("  ChurnTrainer — Starting training pipeline")
        logger.info("=" * 55)

        df = self._load(processed_filepath)
        self._split(df)
        self._scale()
        self._benchmark_models()
        self._tune_xgboost()
        self._evaluate_best_model()
        self._save_artifacts()
        self._save_report()
        self._plot_confusion_matrix()
        self._plot_model_comparison()

        self.is_trained = True

        logger.info("=" * 55)
        logger.info("  Training pipeline complete")
        logger.info(f"  Artifacts saved to: {MODELS_DIR}/")
        logger.info("=" * 55)

    def get_results(self) -> pd.DataFrame:
        """
        Return benchmark results DataFrame sorted by ROC-AUC.

        Returns:
            pd.DataFrame: Model comparison table

        Raises:
            RuntimeError: If run() has not been called yet
        """
        self._check_trained()
        return self.baseline_results.sort_values('ROC-AUC', ascending=False)

    # Private Pipeline Steps

    def _load(self, filepath: str) -> pd.DataFrame:
        """Load processed CSV and validate."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed data not found: {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Loaded processed data — shape: {df.shape}")

        if 'Churn' not in df.columns:
            raise ValueError("'Churn' column missing from processed data")

        return df

    def _split(self, df: pd.DataFrame) -> None:
        """Stratified train/test split — preserves class ratio."""
        logger.info("Step 1 — Splitting data")

        X = df.drop('Churn', axis=1)
        y = df['Churn']
        self.feature_names = list(X.columns)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        logger.info(f"  Train: {self.X_train.shape} | Test: {self.X_test.shape}")
        logger.info(
            f"  Train churn rate: {self.y_train.mean()*100:.1f}% | "
            f"Test churn rate: {self.y_test.mean()*100:.1f}%"
        )

    def _scale(self) -> None:
        """Fit StandardScaler on train set, transform both splits."""
        logger.info("Step 2 — Scaling features")

        self.scaler = StandardScaler()
        self.X_train_sc = self.scaler.fit_transform(self.X_train)
        self.X_test_sc  = self.scaler.transform(self.X_test)

        logger.info("  StandardScaler fitted on train set and applied to both splits")

    def _evaluate_model(self, model, name: str) -> dict:
        """Train a single model and return its evaluation metrics."""
        model.fit(self.X_train_sc, self.y_train)
        preds = model.predict(self.X_test_sc)
        probs = model.predict_proba(self.X_test_sc)[:, 1]

        return {
            'Model':     name,
            'Accuracy':  round(accuracy_score(self.y_test, preds), 3),
            'ROC-AUC':   round(roc_auc_score(self.y_test, probs), 3),
            'F1-Score':  round(f1_score(self.y_test, preds), 3),
            'Precision': round(precision_score(self.y_test, preds), 3),
            'Recall':    round(recall_score(self.y_test, preds), 3)
        }

    def _benchmark_models(self) -> None:
        """Train and evaluate all 5 baseline models."""
        logger.info("Step 3 — Benchmarking baseline models")

        results = []
        for name, model in BASELINE_MODELS.items():
            logger.info(f"  Training: {name}")
            result = self._evaluate_model(model, name)
            results.append(result)
            logger.info(
                f"    ROC-AUC: {result['ROC-AUC']} | "
                f"F1: {result['F1-Score']} | "
                f"Accuracy: {result['Accuracy']}"
            )

        self.baseline_results = (
            pd.DataFrame(results)
            .set_index('Model')
            .sort_values('ROC-AUC', ascending=False)
        )

        logger.info("\n" + self.baseline_results.to_string())

    def _tune_xgboost(self) -> None:
        """
        Tune XGBoost hyperparameters using GridSearchCV (5-fold CV).
        Uses n_jobs=1 for Windows compatibility.
        """
        logger.info("Step 4 — Tuning XGBoost with GridSearchCV")
        logger.info(
            f"  Grid: {len(XGB_PARAM_GRID['n_estimators']) * len(XGB_PARAM_GRID['max_depth']) * len(XGB_PARAM_GRID['learning_rate']) * len(XGB_PARAM_GRID['subsample'])} "
            f"candidates × 5 folds = "
            f"{len(XGB_PARAM_GRID['n_estimators']) * len(XGB_PARAM_GRID['max_depth']) * len(XGB_PARAM_GRID['learning_rate']) * len(XGB_PARAM_GRID['subsample']) * 5} fits"
        )
        logger.info("  This will take 4–6 minutes on Windows (n_jobs=1)...")

        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE
        )

        grid_search = GridSearchCV(
            xgb,
            XGB_PARAM_GRID,
            cv=5,
            scoring='roc_auc',
            n_jobs=1,       # Windows-safe
            verbose=1
        )
        grid_search.fit(self.X_train_sc, self.y_train)

        self.best_model  = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        logger.info(f"  Best params:   {self.best_params}")
        logger.info(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    def _evaluate_best_model(self) -> None:
        """Evaluate tuned XGBoost on held-out test set."""
        logger.info("Step 5 — Evaluating tuned model on test set")

        preds = self.best_model.predict(self.X_test_sc)
        probs = self.best_model.predict_proba(self.X_test_sc)[:, 1]

        logger.info(f"  Accuracy:  {accuracy_score(self.y_test, preds):.4f}")
        logger.info(f"  ROC-AUC:   {roc_auc_score(self.y_test, probs):.4f}")
        logger.info(f"  F1-Score:  {f1_score(self.y_test, preds):.4f}")
        logger.info(f"  Precision: {precision_score(self.y_test, preds):.4f}")
        logger.info(f"  Recall:    {recall_score(self.y_test, preds):.4f}")
        logger.info("\n" + classification_report(
            self.y_test, preds,
            target_names=['No Churn', 'Churn']
        ))

    def _save_artifacts(self) -> None:
        """Save model, scaler, and feature names to models/ directory."""
        logger.info("Step 6 — Saving artifacts")
        os.makedirs(MODELS_DIR, exist_ok=True)

        model_path    = os.path.join(MODELS_DIR, 'xgboost_best_model.pkl')
        scaler_path   = os.path.join(MODELS_DIR, 'scaler.pkl')
        features_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
        params_path   = os.path.join(MODELS_DIR, 'best_params.json')

        joblib.dump(self.best_model,    model_path)
        joblib.dump(self.scaler,        scaler_path)
        joblib.dump(self.feature_names, features_path)

        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)

        logger.info(f"  Model saved       → {model_path}")
        logger.info(f"  Scaler saved      → {scaler_path}")
        logger.info(f"  Features saved    → {features_path}")
        logger.info(f"  Best params saved → {params_path}")

    def _save_report(self) -> None:
        """Save benchmark results table to CSV."""
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report_path = os.path.join(REPORTS_DIR, 'model_benchmark_results.csv')
        self.baseline_results.to_csv(report_path)
        logger.info(f"  Benchmark report  → {report_path}")

    def _plot_confusion_matrix(self) -> None:
        """Plot and save confusion matrix for tuned XGBoost."""
        preds = self.best_model.predict(self.X_test_sc)
        cm    = confusion_matrix(self.y_test, preds)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['No Churn', 'Churn']
        ).plot(ax=ax, colorbar=False, cmap='Blues')

        ax.set_title(
            'XGBoost (Tuned) — Confusion Matrix',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        path = os.path.join(REPORTS_DIR, 'confusion_matrix_train.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  confusion matrix → {path}")

    def _plot_model_comparison(self) -> None:
        """Plot and save grouped bar chart comparing all 5 models."""
        df_plot = (
            self.baseline_results[['Accuracy', 'ROC-AUC', 'F1-Score']]
            .reset_index()
            .melt(id_vars='Model', var_name='Metric', value_name='Score')
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=df_plot,
            x='Model', y='Score',
            hue='Metric',
            palette='Set2',
            ax=ax
        )
        ax.set_title(
            'Model Comparison — Accuracy, ROC-AUC, F1-Score',
            fontsize=13, fontweight='bold'
        )
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.set_ylim(0.5, 1.0)
        ax.tick_params(axis='x', rotation=15)
        plt.tight_layout()

        path = os.path.join(REPORTS_DIR, 'model_comparison_train.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  model comparison → {path}")

    def _check_trained(self) -> None:
        """Raise RuntimeError if run() has not been called yet."""
        if not self.is_trained:
            raise RuntimeError(
                "Trainer has not been run yet. "
                "Call run(processed_filepath) first."
            )


# Standalone Runner 
if __name__ == '__main__':
    """
    Run training pipeline directly from command line:
        python src/train.py
    """
    PROCESSED_PATH = 'data/processed_churn.csv'

    # Verify processed data exists — if not, run preprocessor first
    if not os.path.exists(PROCESSED_PATH):
        logger.warning(
            f"Processed data not found at {PROCESSED_PATH}. "
            f"Running preprocessor first..."
        )
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from preprocess import ChurnPreprocessor

        preprocessor = ChurnPreprocessor()
        preprocessor.fit_transform('../data/telco_churn.csv')
        preprocessor.save(PROCESSED_PATH)

    trainer = ChurnTrainer()
    trainer.run(PROCESSED_PATH)

    print("\n── Final Benchmark Results ──────────────────────")
    print(trainer.get_results().to_string())
    print("─────────────────────────────────────────────────")