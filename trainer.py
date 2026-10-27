"""Training and evaluation for tabular classification and regression."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLASSIFICATION_ALGORITHMS = [
    "Logistic Regression", "K-Nearest Neighbors", "Decision Tree",
    "Random Forest", "Support Vector Machine", "Gaussian Naive Bayes",
    "Gradient Boosting",
]

REGRESSION_ALGORITHMS = [
    "Linear Regression", "K-Nearest Neighbors Regressor",
    "Decision Tree Regressor", "Random Forest Regressor",
    "Gradient Boosting Regressor",
]


def default_parameters(algorithm):
    """Return one sensible parameter set shared by training and comparison."""
    params = {}
    if "Tree" in algorithm or "Forest" in algorithm or "Boosting" in algorithm:
        params["max_depth"] = 5
    if "Forest" in algorithm or "Boosting" in algorithm:
        params["n_estimators"] = 100
    if "K-Nearest Neighbors" in algorithm:
        params["n_neighbors"] = 5
    if "Boosting" in algorithm:
        params["learning_rate"] = 0.1
    if algorithm == "Logistic Regression":
        params["max_iter"] = 1000
    if algorithm == "Support Vector Machine":
        params.update(C=1.0, kernel="rbf")
    return params


def _stringify_categories(values):
    """Normalize mixed categorical values to strings before encoding."""
    return np.asarray(values).astype(str)


def _one_hot_encoder(dense=False):
    """Support both new and older scikit-learn encoder argument names."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)


def _make_model(task, algorithm, params):
    depth = params.get("max_depth", 5)
    trees = params.get("n_estimators", 100)
    neighbors = params.get("n_neighbors", 5)
    rate = params.get("learning_rate", 0.1)

    models = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=params.get("max_iter", 1000)),
        "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=neighbors),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=depth, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=42, n_jobs=-1),
        "Support Vector Machine": lambda: SVC(C=params.get("C", 1.0), kernel=params.get("kernel", "rbf")),
        "Gaussian Naive Bayes": lambda: GaussianNB(),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=trees, learning_rate=rate, max_depth=depth, random_state=42),
        "Linear Regression": lambda: LinearRegression(),
        "K-Nearest Neighbors Regressor": lambda: KNeighborsRegressor(n_neighbors=neighbors),
        "Decision Tree Regressor": lambda: DecisionTreeRegressor(max_depth=depth, random_state=42),
        "Random Forest Regressor": lambda: RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": lambda: GradientBoostingRegressor(n_estimators=trees, learning_rate=rate, max_depth=depth, random_state=42),
    }
    allowed = CLASSIFICATION_ALGORITHMS if task == "Classification" else REGRESSION_ALGORITHMS
    if algorithm not in allowed or algorithm not in models:
        raise ValueError(f"'{algorithm}' is not a valid {task.lower()} algorithm.")
    return models[algorithm]()


def train_model(df, target, task, algorithm, test_size=0.2, params=None):
    """Fit a preprocessing + ML pipeline and evaluate it on held-out rows."""
    params = params or {}
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Upload a non-empty CSV dataset first.")
    if target not in df.columns:
        raise ValueError("The target column is not in this dataset.")
    if not 0.05 <= float(test_size) <= 0.5:
        raise ValueError("Test size must be between 5% and 50%.")
    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique.")

    data = df.dropna(subset=[target]).copy()
    X, y = data.drop(columns=[target]), data[target]

    if task == "Regression":
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
    elif task != "Classification":
        raise ValueError("Select Classification or Regression.")

    if X.empty or len(X.columns) == 0:
        raise ValueError("At least one input feature is required.")

    # Convert datetime columns to text so the encoder can handle them.
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype("string")

    # Drop unused features and rows with no feature values.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("All input feature columns are empty.")
    rows_with_features = X.notna().any(axis=1)
    X, y = X.loc[rows_with_features], y.loc[rows_with_features]
    if len(X) < 5:
        raise ValueError("At least 5 usable rows are required for training.")
    if task == "Classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Gradient Boosting and GaussianNB need dense transformed data.
    dense_algorithms = {
        "Gaussian Naive Bayes", "Gradient Boosting",
        "Gradient Boosting Regressor",
    }
    dense = algorithm in dense_algorithms
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("stringify", FunctionTransformer(_stringify_categories)),
            ("encoder", _one_hot_encoder(dense=dense)),
        ]), categorical_cols))
    preprocessing = ColumnTransformer(transformers, remainder="drop")

    # Select the algorithm and validate train/test split feasibility.
    model = _make_model(task, algorithm, params)
    test_count = max(1, int(np.ceil(len(y) * float(test_size))))
    train_count = len(y) - test_count
    stratify = None
    if task == "Classification":
        counts = y.value_counts()
        if counts.min() >= 2 and test_count >= y.nunique() and train_count >= y.nunique():
            stratify = y
    if train_count < 2 or test_count < 1:
        raise ValueError("The dataset is too small for this test split.")
    if algorithm in {"K-Nearest Neighbors", "K-Nearest Neighbors Regressor"}:
        if params.get("n_neighbors", 5) > train_count:
            raise ValueError(f"K must be no greater than {train_count} for this split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    if task == "Classification":
        score = float(accuracy_score(y_test, predictions))
        metrics = {
            "metric_name": "Accuracy", "score": score,
            "Accuracy": score,
            "Precision (weighted)": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "Recall (weighted)": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "F1 (weighted)": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        }
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    else:
        mse = mean_squared_error(y_test, predictions)
        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, predictions)) if len(y_test) > 1 else 0.0
        metrics = {"metric_name": "R² Score", "score": r2, "R2 Score": r2, "MAE": mae, "RMSE": rmse}
        report = {"MAE": mae, "RMSE": rmse, "R2 Score": r2}

    # Metadata makes later model selection and explanations easier.
    pipeline.automl_metadata_ = {
        "task": task, "target": str(target), "algorithm": algorithm,
        "feature_names": [str(c) for c in X.columns],
    }
    return pipeline, metrics, report
