# SACCO Financial-Distress Predictor

A single, reproducible tool for predicting financial distress in SACCOs
(Savings and Credit Co-operative Organisations). It consolidates the original
notebook experiments into one clean pipeline that trains and compares four
models:

- Logistic Regression
- Random Forest
- XGBoost
- LSTM (sequence model over each SACCO's multi-year history)

## Installation

```bash
pip install -r requirements.txt
```

The core models (Logistic Regression, Random Forest) only need
`pandas`, `numpy`, `scikit-learn` and `matplotlib`. `xgboost`,
`imbalanced-learn` (SMOTE) and `tensorflow` (LSTM) are optional — the tool
automatically skips any model whose dependency is not installed.

## Web app

A Streamlit UI wraps the whole pipeline:

```bash
streamlit run app.py
```

It opens in your browser and lets you:

- generate a synthetic dataset or upload your own (`.csv`/`.xlsx`),
- train and compare the models with interactive charts and per-model diagnostics,
- **score a single SACCO** from its financial ratios (distress-probability gauge), and
- explore the data (class balance, distributions, correlations).

## Command-line usage

Run with a generated synthetic dataset (no data file needed):

```bash
python sacco_predictor.py
```

Run against your own dataset:

```bash
python sacco_predictor.py --data path/to/sacco_dataset.xlsx
```

Skip the LSTM (e.g. when TensorFlow is not installed):

```bash
python sacco_predictor.py --no-lstm --output-dir results
```

### Options

| Flag | Description |
| --- | --- |
| `--data` | Path to a `.csv`/`.xlsx` SACCO dataset. If omitted, a synthetic dataset is generated and saved. |
| `--output-dir` | Folder for tables and plots (default: `model_outputs`). |
| `--no-lstm` | Skip the LSTM model. |

## Expected data schema

A panel (long) dataset with one row per SACCO per year:

- `sacco_id`, `year` — identifiers
- `total_assets`, `total_loans`, `total_deposits`, `capital_adequacy`,
  `liquidity_ratio`, `npl_ratio`, `roa`, `operating_expense`,
  `loan_asset_ratio`, `deposit_growth`, `asset_growth` — features
- `distress` — binary target (1 = distressed)

Missing optional feature columns are tolerated; any column that is not an
identifier or the target is treated as a feature.

## Outputs

Everything is written to the output folder:

- `final_model_comparison.xlsx` — metrics table for all models
- `model_performance_comparison.png` — comparison bar chart
- `*_confusion_matrix.png`, `*_roc_curve.png` — per-model diagnostics
- `feature_importance_table.xlsx`, `feature_importance_plot.png`
- `*_boxplot.png` — key ratios split by distress status
- `lstm_training_loss.png`, `lstm_training_accuracy.png` (when the LSTM runs)
