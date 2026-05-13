"""
Main Pipeline

This script runs the full Multi-Domain Threat Fusion Agent workflow:

1. Generate synthetic multi-domain mission data.
2. Clean and preprocess the dataset.
3. Engineer domain-level risk features.
4. Score mission risk.
5. Train anomaly detection models.
6. Add ML predictions to the final dataset.
7. Generate visualizations.
8. Generate an AI-assisted mission briefing.
9. Save all project outputs.
"""

from pathlib import Path
import pandas as pd

from data_generator import generate_synthetic_events, save_dataset
from preprocessing import clean_data, add_time_features
from feature_engineering import create_engineered_features
from risk_scoring import calculate_mission_risk_score, summarize_risk_distribution
from anomaly_detection import (
    train_isolation_forest,
    train_random_forest_classifier,
    add_model_outputs,
    create_model_summary_text,
)
from agent_briefing import generate_mission_briefing, save_briefing
from visualization import create_all_visualizations


DATA_PATH = "data/synthetic_multi_domain_events.csv"
ASSESSMENT_OUTPUT_PATH = "outputs/reports/mission_risk_assessment.csv"
MODEL_SUMMARY_OUTPUT_PATH = "outputs/reports/model_summary.txt"
RISK_DISTRIBUTION_OUTPUT_PATH = "outputs/reports/risk_distribution.csv"


def ensure_output_directories() -> None:
    """
    Create required project output directories.
    """
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)


def build_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build model-ready features from the scored mission-event dataset.

    Args:
        df: Scored and engineered mission-event DataFrame.

    Returns:
        X: Model feature DataFrame.
        y: Target anomaly labels.
    """
    encoded_df = pd.get_dummies(
        df.drop(columns=["event_id", "timestamp"], errors="ignore"),
        columns=["mission_phase", "region_type", "anomaly_type", "risk_level"],
        drop_first=True,
    )

    X = encoded_df.drop(columns=["anomaly_label"], errors="ignore")
    y = df["anomaly_label"]

    return X, y


def save_text_report(report_text: str, output_path: str) -> None:
    """
    Save a text report.

    Args:
        report_text: Report content.
        output_path: Destination file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(report_text)


def run_pipeline(
    num_records: int = 1000,
    random_state: int = 42,
    use_openai: bool = True,
) -> pd.DataFrame:
    """
    Run the full project pipeline.

    Args:
        num_records: Number of synthetic mission records to generate.
        random_state: Random seed for reproducibility.
        use_openai: Whether to use OpenAI for the mission briefing when an API key is available.

    Returns:
        Final mission assessment DataFrame.
    """
    ensure_output_directories()

    print("Generating synthetic multi-domain mission data...")
    raw_df = generate_synthetic_events(
        num_records=num_records,
        random_state=random_state,
    )
    save_dataset(raw_df, DATA_PATH)

    print("Cleaning data and adding time features...")
    cleaned_df = clean_data(raw_df)
    cleaned_df = add_time_features(cleaned_df)

    print("Creating engineered mission-risk features...")
    engineered_df = create_engineered_features(cleaned_df)

    print("Calculating mission risk scores...")
    scored_df = calculate_mission_risk_score(engineered_df)

    print("Training anomaly detection models...")
    X, y = build_model_features(scored_df)

    isolation_model, isolation_predictions = train_isolation_forest(X)
    supervised_model, metrics, feature_importance = train_random_forest_classifier(X, y)

    print("Adding ML model outputs...")
    final_df = add_model_outputs(
        scored_df,
        isolation_predictions,
        supervised_model,
        X,
    )

    print("Saving mission risk assessment...")
    final_df.to_csv(ASSESSMENT_OUTPUT_PATH, index=False)

    print("Saving risk distribution summary...")
    risk_distribution = summarize_risk_distribution(final_df)
    risk_distribution.to_csv(RISK_DISTRIBUTION_OUTPUT_PATH, index=False)

    print("Saving model summary...")
    model_summary = create_model_summary_text(metrics, feature_importance)
    save_text_report(model_summary, MODEL_SUMMARY_OUTPUT_PATH)

    print("Creating visualizations...")
    create_all_visualizations(final_df)

    print("Generating AI mission briefing...")
    briefing = generate_mission_briefing(final_df, use_openai=use_openai)
    save_briefing(briefing)

    print()
    print("Pipeline completed successfully.")
    print(f"Dataset saved to: {DATA_PATH}")
    print(f"Mission assessment saved to: {ASSESSMENT_OUTPUT_PATH}")
    print(f"Risk distribution saved to: {RISK_DISTRIBUTION_OUTPUT_PATH}")
    print(f"Model summary saved to: {MODEL_SUMMARY_OUTPUT_PATH}")
    print("Mission briefing saved to: outputs/reports/ai_mission_briefing.txt")
    print("Figures saved to: outputs/figures/")

    return final_df


if __name__ == "__main__":
    run_pipeline()