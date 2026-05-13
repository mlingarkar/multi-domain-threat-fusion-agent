"""
Mission Risk Scoring Module

This module converts engineered multi-domain signals into a mission risk
score and readable risk category.
"""

import pandas as pd


def calculate_mission_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mission risk score using engineered domain indicators.

    Args:
        df: DataFrame with engineered features.

    Returns:
        DataFrame with mission risk score and risk level.
    """
    df = df.copy()

    required_columns = [
        "radar_health_score",
        "satellite_stability_score",
        "uav_operational_score",
        "communication_reliability_score",
        "cyber_pressure_index",
        "environmental_risk_index",
        "sensor_degradation_index",
        "multi_domain_pressure_score",
        "simultaneous_domain_stress_count",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns for risk scoring: {missing_columns}")

    df["mission_risk_score"] = (
        (100 - df["radar_health_score"]) * 0.15
        + (100 - df["satellite_stability_score"]) * 0.12
        + (100 - df["uav_operational_score"]) * 0.10
        + (100 - df["communication_reliability_score"]) * 0.15
        + df["cyber_pressure_index"] * 0.18
        + df["environmental_risk_index"] * 0.08
        + df["sensor_degradation_index"] * 0.10
        + df["multi_domain_pressure_score"] * 0.08
        + df["simultaneous_domain_stress_count"] * 4
    ).clip(lower=0, upper=100)

    df["mission_risk_score"] = df["mission_risk_score"].round(2)

    df["risk_level"] = df["mission_risk_score"].apply(assign_risk_level)

    return df


def assign_risk_level(score: float) -> str:
    """
    Assign readable risk level based on mission risk score.

    Args:
        score: Mission risk score.

    Returns:
        Risk level string.
    """
    if score < 30:
        return "Low"
    if score < 55:
        return "Moderate"
    if score < 75:
        return "High"
    return "Critical"


def summarize_risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize mission risk levels.

    Args:
        df: DataFrame with risk levels.

    Returns:
        Risk distribution DataFrame.
    """
    if "risk_level" not in df.columns:
        raise ValueError("Column 'risk_level' not found. Run calculate_mission_risk_score first.")

    summary = (
        df["risk_level"]
        .value_counts()
        .rename_axis("risk_level")
        .reset_index(name="event_count")
    )

    summary["percentage"] = (summary["event_count"] / len(df) * 100).round(2)

    risk_order = ["Low", "Moderate", "High", "Critical"]
    summary["risk_level"] = pd.Categorical(
        summary["risk_level"],
        categories=risk_order,
        ordered=True
    )

    return summary.sort_values("risk_level").reset_index(drop=True)


if __name__ == "__main__":
    from preprocessing import load_data, clean_data, add_time_features
    from feature_engineering import create_engineered_features

    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    cleaned_df = add_time_features(cleaned_df)

    engineered_df = create_engineered_features(cleaned_df)
    scored_df = calculate_mission_risk_score(engineered_df)

    print("Mission risk scoring completed successfully.")
    print("\nRisk distribution:")
    print(summarize_risk_distribution(scored_df))

    print("\nSample scored events:")
    print(scored_df[["event_id", "anomaly_type", "mission_risk_score", "risk_level"]].head())