"""SACCO financial-distress prediction tool.

A single, self-contained pipeline that consolidates the original notebook
experiments (Logistic Regression, Random Forest, XGBoost and an LSTM) into one
reproducible tool.

It will:
  1. Load a SACCO panel dataset (CSV or Excel) or generate a realistic
     synthetic one when no dataset is supplied.
  2. Train and evaluate the tabular models (Logistic Regression, Random Forest,
     XGBoost) with feature scaling and SMOTE-based class balancing.
  3. Optionally train and evaluate a sequence (LSTM) model on each SACCO's
     multi-year history.
  4. Save a comparison table plus all plots (confusion matrices, ROC curves,
     feature importance, distress boxplots) to an output folder.

Usage examples:
    python sacco_predictor.py
    python sacco_predictor.py --data path/to/sacco_dataset.xlsx
    python sacco_predictor.py --no-lstm --output-dir results
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib

# Use a non-interactive backend so plots save reliably even without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

warnings.filterwarnings("ignore")

# Column roles shared across the whole pipeline.
ID_COL = "sacco_id"
YEAR_COL = "year"
TARGET_COL = "distress"

# Features that describe a SACCO's financial position in a given year.
FEATURE_COLS = [
    "total_assets",
    "total_loans",
    "total_deposits",
    "capital_adequacy",
    "liquidity_ratio",
    "npl_ratio",
    "roa",
    "operating_expense",
    "loan_asset_ratio",
    "deposit_growth",
    "asset_growth",
]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(
    n_saccos: int = 300,
    start_year: int = 2015,
    n_years: int = 10,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a realistic synthetic SACCO panel dataset.

    Each SACCO is observed for ``n_years`` consecutive years. Financial ratios
    are drawn around SASRA-style prudential ranges and evolve year-on-year with
    a SACCO-specific baseline so the LSTM has meaningful sequences to learn.
    A SACCO is flagged as distressed in a year when it breaches any of the
    capital, liquidity or asset-quality thresholds.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for sacco in range(1, n_saccos + 1):
        # SACCO-specific baselines (some SACCOs are structurally weaker).
        base_capital = rng.normal(0.16, 0.03)
        base_liquidity = rng.normal(0.20, 0.04)
        base_npl = rng.normal(0.07, 0.04)
        base_roa = rng.normal(0.05, 0.02)
        base_opex = rng.normal(0.30, 0.07)
        base_loan_ratio = rng.normal(0.75, 0.05)

        total_assets = rng.lognormal(mean=18.0, sigma=0.5)  # absolute KES level
        prev_assets = total_assets
        prev_deposits = total_assets * rng.uniform(0.7, 0.85)

        for offset in range(n_years):
            year = start_year + offset

            # Ratios drift around the SACCO baseline each year.
            capital_adequacy = max(0.0, base_capital + rng.normal(0, 0.015))
            liquidity_ratio = max(0.0, base_liquidity + rng.normal(0, 0.02))
            npl_ratio = max(0.0, base_npl + rng.normal(0, 0.02))
            roa = base_roa + rng.normal(0, 0.01)
            operating_expense = max(0.0, base_opex + rng.normal(0, 0.03))
            loan_asset_ratio = float(np.clip(base_loan_ratio + rng.normal(0, 0.03), 0.4, 0.95))

            # Balance-sheet growth.
            asset_growth = rng.normal(0.08, 0.05)
            total_assets = max(prev_assets * (1 + asset_growth), 1.0)
            total_deposits = total_assets * rng.uniform(0.7, 0.85)
            deposit_growth = (total_deposits - prev_deposits) / prev_deposits if prev_deposits else 0.0
            total_loans = total_assets * loan_asset_ratio

            distress = int(
                (npl_ratio > 0.10) or (liquidity_ratio < 0.15) or (capital_adequacy < 0.10)
            )

            rows.append(
                {
                    ID_COL: sacco,
                    YEAR_COL: year,
                    "total_assets": total_assets,
                    "total_loans": total_loans,
                    "total_deposits": total_deposits,
                    "capital_adequacy": capital_adequacy,
                    "liquidity_ratio": liquidity_ratio,
                    "npl_ratio": npl_ratio,
                    "roa": roa,
                    "operating_expense": operating_expense,
                    "loan_asset_ratio": loan_asset_ratio,
                    "deposit_growth": deposit_growth,
                    "asset_growth": asset_growth,
                    TARGET_COL: distress,
                }
            )

            prev_assets = total_assets
            prev_deposits = total_deposits

    return pd.DataFrame(rows)


def load_dataset(path: str) -> pd.DataFrame:
    """Load a SACCO dataset from a CSV or Excel file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {ext!r}. Use .csv, .xlsx or .xls.")


def resolve_dataset(data_path: Optional[str], output_dir: str) -> pd.DataFrame:
    """Return the working dataset, generating a synthetic one if needed."""
    if data_path:
        print(f"Loading dataset from: {data_path}")
        return load_dataset(data_path)

    print("No dataset supplied - generating a synthetic SACCO dataset.")
    df = generate_synthetic_dataset()
    synthetic_path = os.path.join(output_dir, "sacco_synthetic_dataset.csv")
    df.to_csv(synthetic_path, index=False)
    print(f"Synthetic dataset saved to: {synthetic_path}")
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    """Return the feature columns present in ``df``."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    if not available:
        # Fall back to every column that is not an identifier/target.
        available = [c for c in df.columns if c not in {ID_COL, YEAR_COL, TARGET_COL}]
    return available


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@dataclass
class Result:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def compute_metrics(name: str, y_true, y_pred, y_prob) -> Result:
    return Result(
        model=name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob),
    )


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def save_confusion_matrix(name: str, y_true, y_pred, output_dir: str) -> None:
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred))
    disp.plot(ax=plt.gca())
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{_slug(name)}_confusion_matrix.png"), dpi=300)
    plt.close()


def save_roc_curve(name: str, y_true, y_prob, output_dir: str) -> None:
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=plt.gca())
    plt.title(f"ROC Curve - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{_slug(name)}_roc_curve.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Tabular models (Logistic Regression, Random Forest, XGBoost)
# ---------------------------------------------------------------------------
@dataclass
class TabularData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    scaler: StandardScaler


def prepare_tabular_data(df: pd.DataFrame, feature_names: list[str]) -> TabularData:
    """Split, scale and SMOTE-balance the tabular training data."""
    X = df[feature_names]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    X_train_balanced, y_train_balanced = _balance_with_smote(X_train_scaled, y_train)

    print("\nClass distribution before SMOTE:")
    print(y_train.value_counts())
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())

    return TabularData(
        X_train=X_train_balanced,
        X_test=X_test_scaled,
        y_train=y_train_balanced,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )


def _balance_with_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """Apply SMOTE if available; otherwise return the data unchanged."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("imbalanced-learn not installed - skipping SMOTE balancing.")
        return X_train, y_train

    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)


def build_tabular_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    except ImportError:
        print("xgboost not installed - skipping the XGBoost model.")

    return models


def train_tabular_models(data: TabularData, output_dir: str) -> tuple[list[Result], dict]:
    results: list[Result] = []
    trained: dict = {}

    for name, model in build_tabular_models().items():
        print("\n" + "=" * 70)
        print(f"MODEL: {name}")
        print("=" * 70)

        model.fit(data.X_train, data.y_train)
        trained[name] = model

        y_pred = model.predict(data.X_test)
        y_prob = model.predict_proba(data.X_test)[:, 1]

        results.append(compute_metrics(name, data.y_test, y_pred, y_prob))

        print("\nClassification Report:")
        print(classification_report(data.y_test, y_pred, zero_division=0))

        save_confusion_matrix(name, data.y_test, y_pred, output_dir)
        save_roc_curve(name, data.y_test, y_prob, output_dir)

    return results, trained


def save_feature_importance(
    trained: dict, feature_names: list[str], output_dir: str
) -> None:
    """Save a feature-importance table and plot from the best available model."""
    model_name = next(
        (n for n in ("Random Forest", "XGBoost") if n in trained), None
    )
    if model_name is None:
        return

    model = trained[model_name]
    importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print(f"\nFeature importance ({model_name}):")
    print(importance)

    importance.to_excel(os.path.join(output_dir, "feature_importance_table.xlsx"), index=False)

    top = importance.head(10).sort_values("Importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top["Feature"], top["Importance"])
    plt.title(f"Feature Importance ({model_name})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_plot.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# LSTM sequence model
# ---------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame, feature_names: list[str]):
    """Turn the panel data into one fixed-length sequence per SACCO.

    The model sees every year except the last and predicts the distress label
    of the final year.
    """
    df_seq = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)

    scaler = StandardScaler()
    df_seq[feature_names] = scaler.fit_transform(df_seq[feature_names])

    seq_len = df_seq.groupby(ID_COL).size().max()

    X_seq, y_seq = [], []
    for _, group in df_seq.groupby(ID_COL):
        group = group.sort_values(YEAR_COL)
        if len(group) != seq_len:
            continue  # keep only SACCOs with a full history
        X_seq.append(group[feature_names].iloc[:-1].values)
        y_seq.append(group[TARGET_COL].iloc[-1])

    return np.array(X_seq), np.array(y_seq)


def train_lstm(df: pd.DataFrame, feature_names: list[str], output_dir: str) -> Optional[Result]:
    """Train and evaluate an LSTM; returns None if TensorFlow is unavailable."""
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.models import Sequential
    except ImportError:
        print("\nTensorFlow not installed - skipping the LSTM model.")
        return None

    print("\n" + "=" * 70)
    print("MODEL: LSTM")
    print("=" * 70)

    X_seq, y_seq = build_sequences(df, feature_names)
    if len(X_seq) < 10:
        print("Not enough complete SACCO sequences to train an LSTM - skipping.")
        return None

    print("LSTM sequence shape:", X_seq.shape)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    classes = np.unique(y_train)
    class_weights = dict(
        zip(
            classes,
            compute_class_weight(class_weight="balanced", classes=classes, y=y_train),
        )
    )

    model = Sequential(
        [
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=0,
    )

    _save_lstm_history(history, output_dir)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    save_confusion_matrix("LSTM", y_test, y_pred, output_dir)
    save_roc_curve("LSTM", y_test, y_prob, output_dir)

    return compute_metrics("LSTM", y_test, y_pred, y_prob)


def _save_lstm_history(history, output_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_training_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("LSTM Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_training_accuracy.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def save_distress_boxplots(df: pd.DataFrame, output_dir: str) -> None:
    """Save boxplots of key ratios split by distress status."""
    variables = [
        "npl_ratio",
        "liquidity_ratio",
        "capital_adequacy",
        "loan_asset_ratio",
        "roa",
        "operating_expense",
    ]
    for var in variables:
        if var not in df.columns:
            continue
        plt.figure(figsize=(7, 5))
        plt.boxplot(
            [df[df[TARGET_COL] == 0][var], df[df[TARGET_COL] == 1][var]],
            tick_labels=["Non-distressed", "Distressed"],
        )
        title = var.replace("_", " ").title()
        plt.title(f"{title} by Distress Status")
        plt.ylabel(title)
        plt.xlabel("Distress Status")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{var}_boxplot.png"), dpi=300)
        plt.close()


def build_comparison(results: list[Result], output_dir: str) -> pd.DataFrame:
    results_df = (
        pd.DataFrame([vars(r) for r in results])
        .rename(
            columns={
                "model": "Model",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1-score",
                "roc_auc": "ROC-AUC",
            }
        )
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_df.to_excel(os.path.join(output_dir, "final_model_comparison.xlsx"), index=False)

    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    results_df.set_index("Model")[metrics].plot(kind="bar", figsize=(12, 6))
    plt.title("Comparison of SACCO Financial-Distress Models")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=0)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_performance_comparison.png"), dpi=300)
    plt.close()

    return results_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(data_path: Optional[str], output_dir: str, use_lstm: bool) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    df = resolve_dataset(data_path, output_dir)
    print("\nDataset shape:", df.shape)
    print(df.head())

    feature_names = select_features(df)
    print("\nFeatures used:", feature_names)

    tabular = prepare_tabular_data(df, feature_names)
    results, trained = train_tabular_models(tabular, output_dir)
    save_feature_importance(trained, feature_names, output_dir)

    if use_lstm:
        lstm_result = train_lstm(df, feature_names, output_dir)
        if lstm_result is not None:
            results.append(lstm_result)

    save_distress_boxplots(df, output_dir)
    results_df = build_comparison(results, output_dir)

    print("\nAll results, tables and plots saved in:")
    print(os.path.abspath(output_dir))
    return results_df


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict SACCO financial distress using tabular and LSTM models."
    )
    parser.add_argument(
        "--data",
        dest="data_path",
        default=None,
        help="Path to a SACCO dataset (.csv/.xlsx). If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_outputs",
        help="Folder to write tables and plots into (default: model_outputs).",
    )
    parser.add_argument(
        "--no-lstm",
        dest="use_lstm",
        action="store_false",
        help="Skip the LSTM model (useful when TensorFlow is unavailable).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run(data_path=args.data_path, output_dir=args.output_dir, use_lstm=args.use_lstm)


if __name__ == "__main__":
    main()
