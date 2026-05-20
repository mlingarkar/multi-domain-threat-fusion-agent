"""
Synthetic Multi-Domain Defense Event Data Generator
"""

from pathlib import Path
import numpy as np
import pandas as pd


MISSION_PHASES = ["Pre-Mission", "Launch", "Transit", "Surveillance", "Return"]
REGION_TYPES = ["Urban", "Coastal", "Mountain", "Desert", "Open Ocean"]


def generate_synthetic_events(num_records: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic multi-domain mission event data.

    Args:
        num_records: Number of synthetic records to create.
        random_state: Random seed for reproducibility.

    Returns:
        A pandas DataFrame containing synthetic mission event data.
    """
    rng = np.random.default_rng(random_state)

    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=num_records,
        freq="15min"
    )

    mission_phase = rng.choice(MISSION_PHASES, size=num_records)
    region_type = rng.choice(REGION_TYPES, size=num_records)

    radar_signal_strength = rng.normal(loc=78, scale=9, size=num_records).clip(25, 100)
    radar_noise_level = rng.normal(loc=18, scale=6, size=num_records).clip(1, 60)

    satellite_temperature = rng.normal(loc=45, scale=8, size=num_records).clip(15, 90)
    satellite_power_level = rng.normal(loc=86, scale=7, size=num_records).clip(35, 100)

    uav_battery_level = rng.normal(loc=72, scale=14, size=num_records).clip(5, 100)
    uav_altitude = rng.normal(loc=12000, scale=2500, size=num_records).clip(1000, 25000)

    communication_latency_ms = rng.normal(loc=120, scale=35, size=num_records).clip(20, 500)
    communication_packet_loss = rng.normal(loc=2.5, scale=1.6, size=num_records).clip(0, 20)

    cyber_alert_count = rng.poisson(lam=2, size=num_records)
    failed_login_attempts = rng.poisson(lam=3, size=num_records)

    weather_severity = rng.integers(1, 6, size=num_records)

    df = pd.DataFrame({
        "event_id": [f"EVT-{i:05d}" for i in range(1, num_records + 1)],
        "timestamp": timestamps,
        "mission_phase": mission_phase,
        "region_type": region_type,
        "radar_signal_strength": radar_signal_strength.round(2),
        "radar_noise_level": radar_noise_level.round(2),
        "satellite_temperature": satellite_temperature.round(2),
        "satellite_power_level": satellite_power_level.round(2),
        "uav_battery_level": uav_battery_level.round(2),
        "uav_altitude": uav_altitude.round(2),
        "communication_latency_ms": communication_latency_ms.round(2),
        "communication_packet_loss": communication_packet_loss.round(2),
        "cyber_alert_count": cyber_alert_count,
        "failed_login_attempts": failed_login_attempts,
        "weather_severity": weather_severity
    })

    df = inject_anomalies(df, anomaly_fraction=0.12, random_state=random_state)

    return df


def inject_anomalies(
    df: pd.DataFrame,
    anomaly_fraction: float = 0.12,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Inject synthetic anomaly patterns into the dataset.

    Args:
        df: Base mission-event DataFrame.
        anomaly_fraction: Fraction of rows to modify as anomalies.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with anomaly labels and modified feature values.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    num_anomalies = int(len(df) * anomaly_fraction)
    anomaly_indices = rng.choice(df.index, size=num_anomalies, replace=False)

    df["anomaly_label"] = 0
    df["anomaly_type"] = "Normal"

    anomaly_patterns = [
        "Radar Degradation",
        "Satellite Telemetry Spike",
        "Cyber Intrusion Pattern",
        "Communication Disruption",
        "Multi-Domain Escalation"
    ]

    for idx in anomaly_indices:
        pattern = rng.choice(anomaly_patterns)

        if pattern == "Radar Degradation":
            df.loc[idx, "radar_signal_strength"] *= rng.uniform(0.35, 0.65)
            df.loc[idx, "radar_noise_level"] *= rng.uniform(1.8, 3.2)

        elif pattern == "Satellite Telemetry Spike":
            df.loc[idx, "satellite_temperature"] *= rng.uniform(1.35, 1.75)
            df.loc[idx, "satellite_power_level"] *= rng.uniform(0.45, 0.75)

        elif pattern == "Cyber Intrusion Pattern":
            df.loc[idx, "cyber_alert_count"] += rng.integers(8, 25)
            df.loc[idx, "failed_login_attempts"] += rng.integers(10, 35)

        elif pattern == "Communication Disruption":
            df.loc[idx, "communication_latency_ms"] *= rng.uniform(2.0, 4.0)
            df.loc[idx, "communication_packet_loss"] *= rng.uniform(3.0, 6.0)

        elif pattern == "Multi-Domain Escalation":
            df.loc[idx, "radar_signal_strength"] *= rng.uniform(0.45, 0.70)
            df.loc[idx, "radar_noise_level"] *= rng.uniform(1.8, 3.0)
            df.loc[idx, "satellite_temperature"] *= rng.uniform(1.25, 1.60)
            df.loc[idx, "communication_latency_ms"] *= rng.uniform(1.8, 3.5)
            df.loc[idx, "cyber_alert_count"] += rng.integers(8, 20)

        df.loc[idx, "anomaly_label"] = 1
        df.loc[idx, "anomaly_type"] = pattern

    numeric_columns = df.select_dtypes(include=["number"]).columns
    df[numeric_columns] = df[numeric_columns].round(2)

    return df


def save_dataset(df: pd.DataFrame, output_path: str = "data/synthetic_multi_domain_events.csv") -> None:
    """
    Save generated dataset to CSV.

    Args:
        df: DataFrame to save.
        output_path: File path for the CSV output.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    dataset = generate_synthetic_events(num_records=1000)
    save_dataset(dataset)
    print("Synthetic multi-domain dataset created successfully.")
    print("Saved to: data/synthetic_multi_domain_events.csv")
