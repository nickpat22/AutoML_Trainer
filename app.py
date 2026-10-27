"""AutoML Trainer: upload data, train, compare, predict and review history."""

import os
import re
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
import joblib

from trainer import (
    CLASSIFICATION_ALGORITHMS, REGRESSION_ALGORITHMS,
    default_parameters, train_model,
)
from predict import list_compatible_models, predict_with_model
from database import get_training_history, save_training


# ---------------- Page and small helpers ----------------
st.set_page_config(page_title="AutoML Trainer", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2.5rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)


def safe_filename_part(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_") or "Model"


def save_model_file(model, model_name, algorithm):
    os.makedirs("models", exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{safe_filename_part(model_name)}_{safe_filename_part(algorithm)}_{stamp}"
    path = os.path.join("models", base + ".joblib")
    if os.path.exists(path):
        path = os.path.join("models", f"{base}_{uuid.uuid4().hex[:6]}.joblib")
    joblib.dump(model, path)
    return path


def parameter_controls(algorithm, row_count, test_size, key_prefix="train"):
    """Show only parameters relevant to the selected algorithm."""
    params = default_parameters(algorithm)
    if "max_depth" in params:
        params["max_depth"] = st.slider("Maximum tree depth", 1, 30, 5, key=f"{key_prefix}_depth")
    if "n_estimators" in params:
        params["n_estimators"] = st.slider("Number of estimators", 10, 300, 100, key=f"{key_prefix}_trees")
    if "n_neighbors" in params:
        train_rows = max(1, int(row_count * (1 - test_size)))
        max_k = max(1, min(20, train_rows))
        params["n_neighbors"] = st.slider("Number of neighbors (K)", 1, max_k, min(5, max_k), key=f"{key_prefix}_knn")
    if "learning_rate" in params:
        params["learning_rate"] = st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.2], value=0.1, key=f"{key_prefix}_rate")
    if "max_iter" in params:
        params["max_iter"] = st.slider("Maximum iterations", 100, 2000, 1000, step=100, key=f"{key_prefix}_iterations")
    if "C" in params:
        params["C"] = st.select_slider("SVM regularization (C)", [0.1, 1.0, 10.0, 100.0], value=1.0, key=f"{key_prefix}_c")
        params["kernel"] = st.selectbox("SVM kernel", ["linear", "rbf", "poly"], key=f"{key_prefix}_kernel")
    return params


def show_report(task, report):
    st.subheader("Evaluation report")
    if task == "Classification":
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    else:
        st.dataframe(pd.DataFrame(list(report.items()), columns=["Metric", "Value"]), use_container_width=True, hide_index=True)


# ---------------- Sidebar: dataset and settings ----------------
st.title("🤖 AutoML Trainer")
st.caption("Upload a CSV, train machine-learning models, compare results, and make predictions.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("For local development, the app uses the MongoDB server configured in database.py.")

    df = None
    if uploaded_file is not None:
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("Please upload a CSV smaller than 50 MB.")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                if df.empty:
                    st.error("The CSV contains no data rows.")
                    df = None
                elif df.columns.duplicated().any():
                    st.error("Column names must be unique.")
                    df = None
            except Exception as exc:
                st.error(f"Could not read the CSV: {exc}")
                df = None

    target = task = test_size = None
    if df is not None:
        st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        target = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns) - 1)
        task = st.selectbox("ML task", ["Classification", "Regression"])
        test_size = st.slider("Test split (%)", 10, 40, 20) / 100
        with st.expander("Preview dataset"):
            st.dataframe(df.head(10), use_container_width=True)
            st.write("Missing values per column")
            st.dataframe(df.isna().sum().rename("Missing values"), use_container_width=True)

# ---------------- Main sections ----------------
train_tab, compare_tab, predict_tab, history_tab = st.tabs(
    ["Train model", "Compare algorithms", "Make predictions", "Training history"]
)

with train_tab:
    st.subheader("Train a model")
    if df is None:
        st.info("Upload a CSV from the sidebar to start training.")
    else:
        algorithms = CLASSIFICATION_ALGORITHMS if task == "Classification" else REGRESSION_ALGORITHMS
        with st.form("train_form"):
            algorithm = st.selectbox("Algorithm", algorithms)
            params = parameter_controls(algorithm, len(df), test_size)
            model_name = st.text_input("Model name", "My First Model")
            train_clicked = st.form_submit_button("Train model", type="primary")

        if train_clicked:
            try:
                with st.spinner(f"Training {algorithm}..."):
                    model, metrics, report = train_model(df, target, task, algorithm, test_size, params)
                    model_path = save_model_file(model, model_name, algorithm)
                    record = {
                        "model_name": model_name.strip() or "My Model",
                        "dataset_name": uploaded_file.name,
                        "task": task,
                        "algorithm": algorithm,
                        "target": target,
                        "test_size": test_size,
                        "parameters": params,
                        "metrics": metrics,
                        "model_path": model_path,
                    }
                    db_error = None
                    try:
                        save_training(record)
                    except Exception as exc:
                        db_error = str(exc)
                st.success("Model trained and saved successfully.")
                st.caption(f"Saved file: `{model_path}`")
                c1, c2, c3 = st.columns(3)
                c1.metric(metrics["metric_name"], f"{metrics['score']:.4f}")
                if task == "Classification":
                    c2.metric("Precision (weighted)", f"{metrics['Precision (weighted)']:.4f}")
                    c3.metric("F1 (weighted)", f"{metrics['F1 (weighted)']:.4f}")
                else:
                    c2.metric("MAE", f"{metrics['MAE']:.4f}")
                    c3.metric("RMSE", f"{metrics['RMSE']:.4f}")
                show_report(task, report)
                if db_error:
                    st.warning(f"The model was saved, but MongoDB history could not be saved: {db_error}")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

with compare_tab:
    st.subheader("Compare algorithms")
    if df is None:
        st.info("Upload a CSV from the sidebar to compare algorithms.")
    else:
        algorithms = CLASSIFICATION_ALGORITHMS if task == "Classification" else REGRESSION_ALGORITHMS
        chosen = st.multiselect("Choose at least two algorithms", algorithms, default=algorithms[:3], key="compare_algorithms")
        st.caption("Each algorithm uses the same dataset split and fixed random seed. Defaults are used for model parameters.")
        if st.button("Run comparison", type="primary"):
            if len(chosen) < 2:
                st.warning("Select at least two algorithms.")
            else:
                rows = []
                progress = st.progress(0)
                status = st.empty()
                for idx, name in enumerate(chosen):
                    status.write(f"Training {name} ({idx + 1}/{len(chosen)})")
                    try:
                        _, metrics, _ = train_model(df, target, task, name, test_size, default_parameters(name))
                        rows.append({"Algorithm": name, **metrics})
                    except Exception as exc:
                        st.warning(f"{name} skipped: {exc}")
                    progress.progress((idx + 1) / len(chosen))
                status.empty()
                progress.empty()
                if rows:
                    results = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
                    results.insert(0, "Rank", results["score"].rank(method="min", ascending=False).astype(int))
                    st.dataframe(results, use_container_width=True, hide_index=True)
                    st.bar_chart(results.set_index("Algorithm")["score"])
                    top_score = results.iloc[0]["score"]
                    tied = results[results["score"] == top_score]["Algorithm"].tolist()
                    metric_name = results.iloc[0]["metric_name"]
                    if len(tied) == 1:
                        st.success(f"Best by {metric_name}: {tied[0]} ({top_score:.4f})")
                    else:
                        st.info(f"Tie on {metric_name} ({top_score:.4f}): {', '.join(tied)}")
                    st.caption("Classification is ranked by accuracy; regression is ranked by R². A single split is only an initial comparison, not proof of general performance.")
                else:
                    st.error("No algorithm completed successfully.")

with predict_tab:
    st.subheader("Make predictions")
    compatible, skipped = list_compatible_models("models")
    if not compatible:
        st.info("No compatible saved models found. Train a new model first.")
        if skipped:
            st.caption(f"Skipped {skipped} older or incompatible model file(s).")
    else:
        options = [item["path"] for item in compatible]
        selected_path = st.selectbox("Trained model", options, format_func=os.path.basename)
        selected = next(item for item in compatible if item["path"] == selected_path)
        model, schema = selected["model"], selected["schema"]
        metadata = getattr(model, "automl_metadata_", {})
        st.caption(f"{metadata.get('task', 'ML')} · {metadata.get('algorithm', 'Saved pipeline')} · {os.path.basename(selected_path)}")
        with st.form("prediction_form"):
            values = {}
            left, right = st.columns(2)
            for i, feature in enumerate(schema):
                container = left if i % 2 == 0 else right
                with container:
                    name = feature["name"]
                    key = f"predict_{os.path.basename(selected_path)}_{name}"
                    if feature["type"] == "numeric":
                        values[name] = st.number_input(name, value=float(feature["default"]), key=key)
                    else:
                        values[name] = st.selectbox(name, feature["options"], key=key)
            predict_clicked = st.form_submit_button("Predict", type="primary")
        if predict_clicked:
            try:
                result = predict_with_model(model, values)
                st.success(f"Predicted result: {result}")
                estimator = model.named_steps.get("model")
                if hasattr(estimator, "predict_proba"):
                    try:
                        row = pd.DataFrame([values])
                        row = row[list(model.named_steps["preprocessing"].feature_names_in_)]
                        confidence = float(model.predict_proba(row).max())
                        st.metric("Highest class probability", f"{confidence:.1%}")
                    except Exception:
                        pass
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with history_tab:
    st.subheader("Training history")
    if st.button("Refresh history"):
        st.session_state["refresh_history"] = True
    try:
        history = get_training_history(limit=100)
        if history:
            history_df = pd.DataFrame(history).drop(columns=["_id"], errors="ignore")
            useful = [c for c in ["created_at", "model_name", "dataset_name", "task", "algorithm", "target", "metrics", "model_path"] if c in history_df.columns]
            st.dataframe(history_df[useful], use_container_width=True, hide_index=True)
        else:
            st.info("No training records found in MongoDB yet.")
    except Exception as exc:
        st.warning(f"Could not connect to MongoDB to load history: {exc}")

st.divider()
st.caption("AutoML Trainer · Models are saved locally in `models/`; training history is stored in MongoDB when available.")
