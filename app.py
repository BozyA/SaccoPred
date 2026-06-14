"""SACCO Financial-Distress Predictor - Streamlit web app.

A modern UI on top of ``sacco_predictor.py``. It lets you:
  - load your own SACCO dataset or generate a synthetic one,
  - train and compare Logistic Regression / Random Forest / XGBoost (+ optional LSTM),
  - inspect per-model diagnostics, and
  - score a single SACCO from its financial ratios.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import RocCurveDisplay, confusion_matrix

import matplotlib.pyplot as plt

import sacco_predictor as sp

st.set_page_config(
    page_title="SACCO Distress Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }
      h1, h2, h3 { font-weight: 700; letter-spacing: -0.01em; }
      .hero {
        background: linear-gradient(120deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem 2.25rem; border-radius: 18px; color: #fff; margin-bottom: 1.5rem;
      }
      .hero h1 { color: #fff; margin: 0 0 .35rem 0; font-size: 2rem; }
      .hero p { color: #cfe3ee; margin: 0; font-size: 1.02rem; }
      .metric-card {
        background: #ffffff; border: 1px solid #eaecef; border-radius: 14px;
        padding: 1rem 1.15rem; box-shadow: 0 1px 3px rgba(16,24,40,.06);
      }
      .metric-card .label { color: #667085; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
      .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #101828; }
      .pill { display:inline-block; padding:.2rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600; }
      .pill-ok { background:#ecfdf3; color:#067647; }
      .pill-bad { background:#fef3f2; color:#b42318; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RISK_THRESHOLD = 0.5


def metric_card(label: str, value: str) -> str:
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>'


def train_models(df: pd.DataFrame, include_lstm: bool) -> dict:
    """Train the tabular models (and optionally the LSTM) and collect everything
    the UI needs for plots and single-SACCO scoring."""
    feature_names = sp.select_features(df)
    data = sp.prepare_tabular_data(df, feature_names)

    models: dict = {}
    preds: dict = {}
    results: list[sp.Result] = []

    for name, model in sp.build_tabular_models().items():
        model.fit(data.X_train, data.y_train)
        y_pred = model.predict(data.X_test)
        y_prob = model.predict_proba(data.X_test)[:, 1]
        models[name] = model
        preds[name] = (data.y_test.values, y_pred, y_prob)
        results.append(sp.compute_metrics(name, data.y_test, y_pred, y_prob))

    if include_lstm:
        lstm = _train_lstm_web(df, feature_names)
        if lstm is not None:
            result, y_true, y_pred, y_prob = lstm
            preds["LSTM"] = (y_true, y_pred, y_prob)
            results.append(result)

    comparison = (
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

    return {
        "feature_names": feature_names,
        "scaler": data.scaler,
        "models": models,
        "preds": preds,
        "comparison": comparison,
        "df": df,
    }


def _train_lstm_web(df: pd.DataFrame, feature_names: list[str]):
    """Train an LSTM for the comparison view; returns None if TF is unavailable."""
    try:
        from sklearn.utils.class_weight import compute_class_weight
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.models import Sequential
    except ImportError:
        return None

    X_seq, y_seq = sp.build_sequences(df, feature_names)
    if len(X_seq) < 10:
        return None

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    classes = np.unique(y_train)
    class_weights = dict(
        zip(classes, compute_class_weight("balanced", classes=classes, y=y_train))
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
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        class_weight=class_weights,
        verbose=0,
    )

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    result = sp.compute_metrics("LSTM", y_test, y_pred, y_prob)
    return result, y_test, y_pred, y_prob


# ---------------------------------------------------------------------------
# Sidebar - data + training controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    source = st.radio("Data source", ["Synthetic dataset", "Upload file"], index=0)

    if source == "Synthetic dataset":
        n_saccos = st.slider("Number of SACCOs", 50, 1000, 300, step=50)
        seed = st.number_input("Random seed", value=42, step=1)
        if st.button("Generate dataset", width="stretch"):
            st.session_state.df = sp.generate_synthetic_dataset(
                n_saccos=int(n_saccos), seed=int(seed)
            )
            st.session_state.pop("trained", None)
    else:
        uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded is not None and st.button("Load dataset", width="stretch"):
            if uploaded.name.lower().endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded)
            else:
                st.session_state.df = pd.read_excel(uploaded)
            st.session_state.pop("trained", None)

    # Default dataset on first run so the app is never empty.
    if "df" not in st.session_state:
        st.session_state.df = sp.generate_synthetic_dataset()

    st.divider()
    include_lstm = st.checkbox("Include LSTM (slower)", value=False)
    if st.button("🚀 Train models", type="primary", width="stretch"):
        with st.spinner("Training models..."):
            st.session_state.trained = train_models(st.session_state.df, include_lstm)
        st.success("Models trained.")

df: pd.DataFrame = st.session_state.df

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>🏦 SACCO Financial-Distress Predictor</h1>'
    "<p>Train and compare machine-learning models, then score individual SACCOs "
    "on their risk of financial distress.</p></div>",
    unsafe_allow_html=True,
)

if sp.TARGET_COL not in df.columns:
    st.error(
        f"The dataset has no '{sp.TARGET_COL}' column. "
        "Please provide a dataset that includes the binary distress target."
    )
    st.stop()

distress_rate = df[sp.TARGET_COL].mean()
cols = st.columns(4)
cols[0].markdown(metric_card("SACCO-year records", f"{len(df):,}"), unsafe_allow_html=True)
cols[1].markdown(
    metric_card("Unique SACCOs", f"{df[sp.ID_COL].nunique():,}" if sp.ID_COL in df else "—"),
    unsafe_allow_html=True,
)
cols[2].markdown(metric_card("Features", f"{len(sp.select_features(df))}"), unsafe_allow_html=True)
cols[3].markdown(metric_card("Distress rate", f"{distress_rate:.1%}"), unsafe_allow_html=True)

st.write("")
tab_compare, tab_predict, tab_data = st.tabs(
    ["📊 Train & Compare", "🎯 Predict a SACCO", "🔎 Data Explorer"]
)

# ---------------------------------------------------------------------------
# Tab: Train & compare
# ---------------------------------------------------------------------------
with tab_compare:
    if "trained" not in st.session_state:
        st.info("Use **Train models** in the sidebar to fit and compare the models.")
    else:
        out = st.session_state.trained
        comparison = out["comparison"]

        best = comparison.iloc[0]
        st.subheader(f"🏆 Best model: {best['Model']} (ROC-AUC {best['ROC-AUC']:.3f})")

        styled = comparison.style.format(
            {c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]}
        ).background_gradient(cmap="Greens", subset=["ROC-AUC"])
        st.dataframe(styled, width="stretch", hide_index=True)

        melted = comparison.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            melted, x="Model", y="Score", color="Metric", barmode="group",
            range_y=[0, 1.05], title="Model performance comparison",
        )
        fig.update_layout(legend_title_text="", height=440)
        st.plotly_chart(fig, width="stretch")

        st.divider()
        model_choice = st.selectbox("Inspect a model", list(out["preds"].keys()))
        y_true, y_pred, y_prob = out["preds"][model_choice]

        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            fig_cm, ax = plt.subplots(figsize=(4.5, 4))
            cm = confusion_matrix(y_true, y_pred)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["No distress", "Distress"])
            ax.set_yticklabels(["No distress", "Distress"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black", fontsize=13)
            fig_cm.colorbar(im, fraction=0.046, pad=0.04)
            fig_cm.tight_layout()
            st.pyplot(fig_cm)
        with right:
            st.markdown("**ROC curve**")
            fig_roc, ax = plt.subplots(figsize=(4.5, 4))
            RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
            ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
            fig_roc.tight_layout()
            st.pyplot(fig_roc)

# ---------------------------------------------------------------------------
# Tab: Single-SACCO prediction
# ---------------------------------------------------------------------------
with tab_predict:
    if "trained" not in st.session_state:
        st.info("Train the models first (sidebar) to enable scoring.")
    else:
        out = st.session_state.trained
        feature_names = out["feature_names"]
        tabular_models = [m for m in out["models"]]  # excludes LSTM
        scorer_name = st.selectbox("Scoring model", tabular_models, key="scorer")

        st.caption("Enter the SACCO's latest financial indicators:")
        defaults = df[feature_names].median()
        inputs: dict = {}
        ncols = 3
        grid = st.columns(ncols)
        for i, feat in enumerate(feature_names):
            col = grid[i % ncols]
            label = feat.replace("_", " ").title()
            inputs[feat] = col.number_input(
                label, value=float(defaults[feat]), format="%.4f", key=f"in_{feat}"
            )

        if st.button("Predict distress risk", type="primary"):
            model = out["models"][scorer_name]
            x = pd.DataFrame([inputs])[feature_names]
            x_scaled = out["scaler"].transform(x)
            prob = float(model.predict_proba(x_scaled)[0, 1])
            distressed = prob >= RISK_THRESHOLD

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%"},
                    title={"text": "Distress probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#b42318" if distressed else "#067647"},
                        "steps": [
                            {"range": [0, 35], "color": "#ecfdf3"},
                            {"range": [35, 65], "color": "#fff7e6"},
                            {"range": [65, 100], "color": "#fef3f2"},
                        ],
                    },
                )
            )
            gauge.update_layout(height=320, margin=dict(t=60, b=10))
            gleft, gright = st.columns([1, 1])
            gright.plotly_chart(gauge, width="stretch")
            with gleft:
                if distressed:
                    st.markdown(
                        '<span class="pill pill-bad">⚠ HIGH RISK — likely distressed</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="pill pill-ok">✓ LOW RISK — financially stable</span>',
                        unsafe_allow_html=True,
                    )
                st.metric("Distress probability", f"{prob:.1%}")
                st.caption(
                    f"Model: {scorer_name}. Classification threshold: {RISK_THRESHOLD:.0%}."
                )

            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame(
                    {"Feature": feature_names, "Importance": model.feature_importances_}
                ).sort_values("Importance", ascending=True).tail(10)
                fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                                 title=f"What drives the {scorer_name} model")
                fig_imp.update_layout(height=380)
                st.plotly_chart(fig_imp, width="stretch")

# ---------------------------------------------------------------------------
# Tab: Data explorer
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(df.head(50), width="stretch", hide_index=True)

    feature_names = sp.select_features(df)
    c1, c2 = st.columns(2)
    with c1:
        counts = df[sp.TARGET_COL].value_counts().rename({0: "No distress", 1: "Distress"})
        fig_bal = px.pie(values=counts.values, names=counts.index, title="Class balance", hole=0.5)
        st.plotly_chart(fig_bal, width="stretch")
    with c2:
        feat = st.selectbox("Feature distribution by distress status", feature_names)
        plot_df = df.copy()
        plot_df["Status"] = plot_df[sp.TARGET_COL].map({0: "No distress", 1: "Distress"})
        fig_box = px.box(plot_df, x="Status", y=feat, color="Status",
                         title=f"{feat.replace('_', ' ').title()} by status")
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, width="stretch")

    numeric = df[feature_names]
    if len(feature_names) > 1:
        st.markdown("**Feature correlation**")
        fig_corr = px.imshow(
            numeric.corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )
        fig_corr.update_layout(height=520)
        st.plotly_chart(fig_corr, width="stretch")
