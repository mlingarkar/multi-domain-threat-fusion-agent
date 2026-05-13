"""
Anomaly Detection Module

This module trains machine learning models to detect anomalous
multi-domain mission events.

It includes:
1. Isolation Forest for unsupervised anomaly detection.
2. Random Forest for supervised anomaly classification.
"""

from typing import Tuple, Dict, List
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.12,
    random_state: int = 42
) -> Tuple[IsolationForest, pd.Series]:
    """
    Train an Isolation Forest model for unsupervised anomaly detection.

    Args:
        X: Feature DataFrame.
        contamination: Expected anomaly fraction.
        random_state: Random seed.

    Returns:
        Trained Isolation Forest model and anomaly predictions.
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=150
    )

    predictions = model.fit_predict(X)

    # Isolation Forest returns -1 for anomalies and 1 for normal events.
    anomaly_predictions = pd.Series(
        [1 if prediction == -1 else 0 for prediction in predictions],
        index=X.index,
        name="isolation_forest_anomaly"
    )

    return model, anomaly_predictions


def train_random_forest_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, object], pd.DataFrame]:
    """
    Train a supervised Random Forest classifier to detect anomaly labels.

    Args:
        X: Feature DataFrame.
        y: Target labels.
        random_state: Random seed.

    Returns:
        Trained model, metrics dictionary, and feature importance DataFrame.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced",
        max_depth=10
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred)
    }

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    feature_importance["importance"] = feature_importance["importance"].round(4)

    return model, metrics, feature_importance


def add_model_outputs(
    df: pd.DataFrame,
    isolation_predictions: pd.Series,
    supervised_model: RandomForestClassifier,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Add model prediction outputs to the scored mission dataset.

    Args:
        df: Original mission-event DataFrame.
        isolation_predictions: Isolation Forest anomaly predictions.
        supervised_model: Trained Random Forest classifier.
        X: Feature DataFrame used by the model.

    Returns:
        DataFrame with ML model prediction outputs.
    """
    df = df.copy()

    df["isolation_forest_anomaly"] = isolation_predictions.values
    df["ml_anomaly_prediction"] = supervised_model.predict(X)

    if hasattr(supervised_model, "predict_proba"):
        df["ml_anomaly_probability"] = supervised_model.predict_proba(X)[:, 1].round(4)

    return df


def create_model_summary_text(
    metrics: Dict[str, object],
    feature_importance: pd.DataFrame,
    top_n: int = 10
) -> str:
    """
    Create a readable model summary.

    Args:
        metrics: Model metrics dictionary.
        feature_importance: Feature importance DataFrame.
        top_n: Number of top features to include.

    Returns:
        Model summary as a string.
    """
    top_features = feature_importance.head(top_n)

    lines = [
        "Machine Learning Model Summary",
        "================================",
        "",
        f"Accuracy: {metrics['accuracy']}",
        f"Training Rows: {metrics['training_rows']}",
        f"Testing Rows: {metrics['testing_rows']}",
        "",
        "Confusion Matrix:",
        str(metrics["confusion_matrix"]),
        "",
        "Top Feature Drivers:",
    ]

    for _, row in top_features.iterrows():
        lines.append(f"- {row['feature']}: {row['importance']}")

    lines.extend([
        "",
        "Classification Report:",
        metrics["classification_report"]
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    from preprocessing import preprocess_pipeline
    from feature_engineering import create_engineered_features
    from risk_scoring import calculate_mission_risk_score

    X, y, feature_names, cleaned_df = preprocess_pipeline()

    engineered_df = create_engineered_features(cleaned_df)
    scored_df = calculate_mission_risk_score(engineered_df)

    # Rebuild feature set after engineered features are added.
    encoded_df = pd.get_dummies(
        scored_df.drop(columns=["event_id", "timestamp"], errors="ignore"),
        columns=["mission_phase", "region_type", "anomaly_type", "risk_level"],
        drop_first=True
    )

    X_engineered = encoded_df.drop(columns=["anomaly_label"], errors="ignore")
    y_engineered = scored_df["anomaly_label"]

    isolation_model, isolation_predictions = train_isolation_forest(X_engineered)
    supervised_model, metrics, feature_importance = train_random_forest_classifier(
        X_engineered,
        y_engineered
    )

    final_df = add_model_outputs(
        scored_df,
        isolation_predictions,
        supervised_model,
        X_engineered
    )

    print("Anomaly detection completed successfully.")
    print(create_model_summary_text(metrics, feature_importance))
    print("\nSample model outputs:")
    print(
        final_df[
            [
                "event_id",
                "anomaly_label",
                "isolation_forest_anomaly",
                "ml_anomaly_prediction",
                "ml_anomaly_probability",
                "mission_risk_score",
                "risk_level"
            ]
        ].head()
    )