"""
Unit tests for the synthetic multi-domain data generator.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from data_generator import generate_synthetic_events  # noqa: E402


def test_generate_synthetic_events_row_count():
    df = generate_synthetic_events(num_records=100, random_state=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100


def test_generate_synthetic_events_required_columns():
    df = generate_synthetic_events(num_records=50, random_state=42)

    required_columns = [
        "event_id",
        "timestamp",
        "mission_phase",
        "region_type",
        "radar_signal_strength",
        "radar_noise_level",
        "satellite_temperature",
        "satellite_power_level",
        "uav_battery_level",
        "uav_altitude",
        "communication_latency_ms",
        "communication_packet_loss",
        "cyber_alert_count",
        "failed_login_attempts",
        "weather_severity",
        "anomaly_label",
        "anomaly_type",
    ]

    for column in required_columns:
        assert column in df.columns


def test_generate_synthetic_events_anomaly_labels_exist():
    df = generate_synthetic_events(num_records=200, random_state=42)

    assert "anomaly_label" in df.columns
    assert set(df["anomaly_label"].unique()).issubset({0, 1})
    assert df["anomaly_label"].sum() > 0