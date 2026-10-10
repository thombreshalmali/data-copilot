from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
def train_churn_model(df, target_column, feature_columns):
    if not feature_columns:
        raise ValueError("Select at least one predictor column.")

    if target_column in feature_columns:
        raise ValueError("The target cannot also be a predictor.")

    data = df.dropna(subset=[target_column]).drop_duplicates().copy()

    labels = data[target_column].astype(str).str.strip()

    if set(labels.unique()) != {"Yes", "No"}:
        raise ValueError("This first model requires Yes/No target values.")

    y = labels.map({"No": 0, "Yes": 1})
    X = data[feature_columns].copy()

    if len(data) < 40 or y.value_counts().min() < 8:
        raise ValueError("Use at least 40 rows and 8 examples of each class.")

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    # Standardize categorical missing values for Scikit-learn.
    for column in categorical_columns:
        X[column] = X[column].astype(object)
        X[column] = X[column].where(X[column].notna(), float("nan"))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Missing",
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    comparison_rows = []
    candidate_pipelines = {}

    for name, classifier in candidates.items():
        candidate = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ])

        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
        )

        comparison_rows.append({
            "Model": name,
            "CV churn F1": float(scores.mean()),
            "CV standard deviation": float(scores.std()),
        })

        candidate_pipelines[name] = candidate

    best_row = max(
        comparison_rows,
        key=lambda row: row["CV churn F1"],
    )

    best_model_name = best_row["Model"]
    model = candidate_pipelines[best_model_name]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    feature_importance = [
        {
            "Feature": column,
            "Mean F1 decrease": float(mean),
            "Standard deviation": float(std),
        }
        for column, mean, std in zip(
            feature_columns,
            importance.importances_mean,
            importance.importances_std,
        )
    ]
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        zero_division=0,
    )

    return {
        "model": model,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
        "baseline_accuracy": baseline_accuracy,
        "baseline_f1": baseline_f1,
        "best_model_name": best_model_name,
        "cv_comparison": comparison_rows,
        "feature_importance": feature_importance,
        "feature_columns": list(feature_columns),
        "numeric_columns": list(numeric_columns),
        "categorical_columns": list(categorical_columns),
        "target_column": target_column,
    }
def predict_new_customers(new_df, training_results):
    import pandas as pd

    features = training_results["feature_columns"]
    missing_columns = [
        column for column in features
        if column not in new_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    if new_df.empty:
        raise ValueError("The uploaded CSV has no customer rows.")

    X = new_df[features].copy()
    X = X.replace(r"^\s*$", pd.NA, regex=True)

    for column in training_results["numeric_columns"]:
        converted = pd.to_numeric(X[column], errors="coerce")

        invalid = X[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(
                f"Column '{column}' contains "
                f"{int(invalid.sum())} invalid numeric values."
            )

        X[column] = converted.astype("float64")

    for column in training_results["categorical_columns"]:
        X[column] = X[column].astype(object)
        X[column] = X[column].where(
            X[column].notna(), float("nan")
        )

    model = training_results["model"]
    positive_index = list(model.classes_).index(1)

    output = new_df.copy()
    output["PredictedChurn"] = [
        "Yes" if prediction == 1 else "No"
        for prediction in model.predict(X)
    ]
    output["ChurnProbability"] = model.predict_proba(X)[
        :, positive_index
    ]

    return output