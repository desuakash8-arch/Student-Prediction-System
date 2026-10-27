import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

FEATURE_COLUMNS = [
    "Attendance",
    "StudyHours",
    "InternetUsage",
    "SleepHours",
    "PreviousMarks",
    "AssignmentScore",
    "ClassParticipation",
    "ParentalSupport",
    "ExtraActivities",
]

TARGET_COLUMN = "FinalPerformance"

FEATURE_RULES = {
    "Attendance": {"min": 0, "max": 100, "integer": False},
    "StudyHours": {"min": 0, "max": 24, "integer": False},
    "InternetUsage": {"min": 0, "max": 24, "integer": False},
    "SleepHours": {"min": 0, "max": 24, "integer": False},
    "PreviousMarks": {"min": 0, "max": 100, "integer": False},
    "AssignmentScore": {"min": 0, "max": 100, "integer": False},
    "ClassParticipation": {"min": 0, "max": 10, "integer": False},
    "ParentalSupport": {"min": 0, "max": 1, "allowed": {0, 1}},
    "ExtraActivities": {"min": 0, "max": 1, "allowed": {0, 1}},
}

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "student_data.csv"
DEFAULT_MODEL = BASE_DIR / "model.pkl"
DEFAULT_METRICS = BASE_DIR / "model_metrics.json"
DEFAULT_IMPORTANCE_PLOT = BASE_DIR / "static" / "feature_importance.png"


def generate_student_dataset(n_samples=1000, seed=42):
    """Create a larger, reproducible student-performance dataset."""
    rng = np.random.default_rng(seed)

    attendance = np.clip(rng.normal(78, 12, n_samples), 40, 100)
    study_hours = np.clip(rng.normal(3.8, 1.6, n_samples), 0.5, 10)
    internet_usage = np.clip(rng.normal(4.2, 1.8, n_samples), 0.5, 12)
    sleep_hours = np.clip(rng.normal(6.8, 1.1, n_samples), 4, 10)
    previous_marks = np.clip(rng.normal(72, 14, n_samples), 30, 100)
    assignment_score = np.clip(rng.normal(74, 13, n_samples), 30, 100)
    class_participation = np.clip(rng.normal(6.5, 1.8, n_samples), 1, 10)
    parental_support = rng.integers(0, 2, n_samples)
    extra_activities = rng.integers(0, 2, n_samples)

    performance = (
        0.16 * attendance
        + 3.1 * study_hours
        - 1.35 * internet_usage
        + 1.7 * sleep_hours
        + 0.32 * previous_marks
        + 0.24 * assignment_score
        + 1.15 * class_participation
        + 3.2 * parental_support
        + 2.1 * extra_activities
        + rng.normal(0, 3.5, n_samples)
    )
    performance = np.clip(performance, 0, 100)

    return pd.DataFrame(
        {
            "Attendance": np.round(attendance, 2),
            "StudyHours": np.round(study_hours, 2),
            "InternetUsage": np.round(internet_usage, 2),
            "SleepHours": np.round(sleep_hours, 2),
            "PreviousMarks": np.round(previous_marks, 2),
            "AssignmentScore": np.round(assignment_score, 2),
            "ClassParticipation": np.round(class_participation, 2),
            "ParentalSupport": parental_support,
            "ExtraActivities": extra_activities,
            "FinalPerformance": np.round(performance, 2),
        }
    )


def save_dataset(path=DEFAULT_CSV, n_samples=1000, seed=42):
    data = generate_student_dataset(n_samples=n_samples, seed=seed)
    data.to_csv(path, index=False)
    return data


def validate_student_features(raw_values):
    """Validate that all nine features are present and within expected ranges."""
    if not isinstance(raw_values, dict):
        return None, "All nine input features are required."

    missing = [name for name in FEATURE_COLUMNS if raw_values.get(name) in (None, "")]
    if missing:
        return None, f"Missing required features: {', '.join(missing)}"

    cleaned = []
    for name in FEATURE_COLUMNS:
        try:
            value = float(raw_values[name])
        except (TypeError, ValueError):
            return None, f"{name} must be a number."

        rules = FEATURE_RULES[name]
        if value < rules["min"] or value > rules["max"]:
            return None, (
                f"{name} must be between {rules['min']} and {rules['max']}."
            )

        allowed = rules.get("allowed")
        if allowed is not None and value not in allowed:
            return None, f"{name} must be 0 or 1."

        cleaned.append(value)

    return cleaned, None


def evaluate_model(model, X_test, y_test, X, y, cv_splits=5, random_state=42):
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "neg_mae": "neg_mean_absolute_error",
            "neg_mse": "neg_mean_squared_error",
        },
    )

    return {
        "n_samples": int(len(X)),
        "test_mae": round(mae, 4),
        "test_rmse": round(rmse, 4),
        "test_r2": round(r2, 4),
        "cv_splits": cv_splits,
        "cv_r2_mean": round(float(cv_results["test_r2"].mean()), 4),
        "cv_r2_std": round(float(cv_results["test_r2"].std()), 4),
        "cv_mae_mean": round(float((-cv_results["test_neg_mae"]).mean()), 4),
        "cv_rmse_mean": round(
            float(np.sqrt((-cv_results["test_neg_mse"]).mean())), 4
        ),
    }


def train_model(
    csv_path=DEFAULT_CSV,
    model_path=DEFAULT_MODEL,
    metrics_path=DEFAULT_METRICS,
    importance_plot=DEFAULT_IMPORTANCE_PLOT,
    n_estimators=150,
    random_state=42,
):
    data = pd.read_csv(csv_path)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test, X, y, random_state=random_state)
    metrics["model"] = "RandomForestRegressor"

    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importance_plot = Path(importance_plot)
    importance_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.barh(FEATURE_COLUMNS, model.feature_importances_)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(importance_plot)
    plt.close()

    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)

    return model, metrics


def load_trained_model(model_path=DEFAULT_MODEL):
    with open(model_path, "rb") as handle:
        return pickle.load(handle)


if __name__ == "__main__":
    dataset = save_dataset()
    _, metrics = train_model()
    print(f"Saved dataset with {len(dataset)} rows")
    print("Model trained and saved!")
    print(json.dumps(metrics, indent=2))
