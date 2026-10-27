"""Load saved pipelines and build prediction inputs from their features."""

import glob
import os
import joblib
import numpy as np
import pandas as pd


def load_saved_model(model_path):
    # Only load model files created by this trusted local application.
    return joblib.load(model_path)


def get_input_schema(model):
    """Return feature names, types, defaults and categories for the UI."""
    if not hasattr(model, "named_steps") or "preprocessing" not in model.named_steps:
        raise ValueError("This older model has no supported preprocessing pipeline.")

    prep = model.named_steps["preprocessing"]
    if not hasattr(prep, "transformers_"):
        raise ValueError("The model's preprocessing step has not been fitted.")

    schema_by_name = {}
    for name, transformer, selected_columns in prep.transformers_:
        # Empty feature groups are commonly represented as 'drop'.
        columns = list(selected_columns) if not isinstance(selected_columns, str) else [selected_columns]
        if name not in {"numeric", "categorical"} or not columns:
            continue
        if isinstance(transformer, str) or not hasattr(transformer, "named_steps"):
            continue

        if name == "numeric":
            imputer = transformer.named_steps["imputer"]
            for col, value in zip(columns, imputer.statistics_):
                value = float(value) if np.isfinite(value) else 0.0
                schema_by_name[col] = {"name": str(col), "type": "numeric", "default": value}

        elif name == "categorical":
            encoder = transformer.named_steps["encoder"]
            if not hasattr(encoder, "categories_"):
                raise ValueError("The categorical encoder is not fitted. Train a new model.")
            for col, values in zip(columns, encoder.categories_):
                # Keep original values, so numeric/bool categories preserve their types.
                options = [v.item() if hasattr(v, "item") else v for v in values]
                if options:
                    schema_by_name[col] = {"name": str(col), "type": "categorical", "options": options}

    feature_names = list(getattr(prep, "feature_names_in_", schema_by_name.keys()))
    schema = [schema_by_name[c] for c in feature_names if c in schema_by_name]
    if not schema:
        raise ValueError("No supported input features found in this model.")
    return schema


def list_compatible_models(folder="models"):
    """Return compatible pipelines, newest first; skip stale/corrupt models."""
    paths = glob.glob(os.path.join(folder, "*.joblib"))
    paths.sort(key=os.path.getmtime, reverse=True)
    compatible = []
    skipped = 0
    for path in paths:
        try:
            model = load_saved_model(path)
            schema = get_input_schema(model)
            compatible.append({"path": path, "model": model, "schema": schema})
        except Exception:
            skipped += 1
    return compatible, skipped


def predict_with_model(model, input_values):
    # Use a DataFrame so feature names and order are preserved.
    input_df = pd.DataFrame([input_values])
    required = list(model.named_steps["preprocessing"].feature_names_in_)
    missing = [name for name in required if name not in input_df.columns]
    if missing:
        raise ValueError("Missing input fields: " + ", ".join(missing))
    input_df = input_df[required]
    return model.predict(input_df)[0]
