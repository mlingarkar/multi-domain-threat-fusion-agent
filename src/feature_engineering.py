"""
Feature Engineering Module

This module creates higher-level mission risk indicators from the
synthetic multi-domain dataset.

These engineered features help represent domain-specific health signals
across radar, satellite telemetry, UAV status, communications, cyber
activity, and environmental conditions.
"""

import pandas as pd


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-level engineered features.

    Args:
        df: Cleaned mission-event DataFrame.

    Returns:
        DataFrame with additional engineered features.
    """
    df = df.copy()

    df["radar_health_score"] = (
        df["radar_signal_strength"] - df["radar_noise_level"]
    ).clip(lower=0)

    df["satellite_stability_score"] = (
        df["satellite_power_level"] - (df["satellite_temperature"] * 0.5)
    ).clip(lower=0)

    df["uav_operational_score"] = (
        (df["uav_battery_level"] * 0.7) + ((df["uav_altitude"] / 25000) * 30)
    ).clip(lower=0, upper=100)

    df["communication_reliability_score"] = (
        100
        - (df["communication_latency_ms"] * 0.08)
        - (df["communication_packet_loss"] * 3.5)
    ).clip(lower=0, upper=100)

    df["cyber_pressure_index"] = (
        (df["cyber_alert_count"] * 2.5)
        + (df["failed_login_attempts"] * 1.8)
    )

    df["environmental_risk_index"] = df["weather_severity"] * 10

    df["sensor_degradation_index"] = (
        (100 - df["radar_health_score"]) * 0.35
        + (100 - df["satellite_stability_score"]) * 0.25
        + (100 - df["communication_reliability_score"]) * 0.25
        + df["environmental_risk_index"] * 0.15
    ).clip(lower=0, upper=100)

    df["multi_domain_pressure_score"] = (
        df["sensor_degradation_index"] * 0.45
        + df["cyber_pressure_index"] * 0.35
        + (100 - df["uav_operational_score"]) * 0.20
    ).clip(lower=0, upper=100)

    df["simultaneous_domain_stress_count"] = (
        (df["radar_health_score"] < 45).astype(int)
        + (df["satellite_stability_score"] < 45).astype(int)
        + (df["communication_reliability_score"] < 55).astype(int)
        + (df["cyber_pressure_index"] > 35).astype(int)
        + (df["uav_operational_score"] < 45).astype(int)
        + (df["weather_severity"] >= 4).astype(int)
    )

    return df


def get_engineered_feature_columns() -> list:
    """
    Return the list of engineered feature names.

    Returns:
        List of engineered feature column names.
    """
    return [
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


if __name__ == "__main__":
    from preprocessing import load_data, clean_data, add_time_features

    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    cleaned_df = add_time_features(cleaned_df)

    engineered_df = create_engineered_features(cleaned_df)

    print("Feature engineering completed successfully.")
    print("Engineered columns created:")
    for column in get_engineered_feature_columns():
        print(f"- {column}")

    print("\nSample engineered feature values:")
    print(engineered_df[get_engineered_feature_columns()].head())