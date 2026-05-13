"""
Unit tests for mission risk scoring.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from data_generator import generate_synthetic_events  # noqa: E402
from preprocessing import clean_data, add_time_features  # noqa: E402
from feature_engineering import create_engineered_features  # noqa: E402
from risk_scoring import calculate_mission_risk_score, assign_risk_level  # noqa: E402


def test_assign_risk_level_outputs_expected_categories():
    assert assign_risk_level(10) == "Low"
    assert assign_risk_level(40) == "Moderate"
    assert assign_risk_level(60) == "High"
    assert assign_risk_level(90) == "Critical"


def test_calculate_mission_risk_score_creates_columns():
    raw_df = generate_synthetic_events(num_records=100, random_state=42)
    cleaned_df = clean_data(raw_df)
    cleaned_df = add_time_features(cleaned_df)
    engineered_df = create_engineered_features(cleaned_df)

    scored_df = calculate_mission_risk_score(engineered_df)

    assert "mission_risk_score" in scored_df.columns
    assert "risk_level" in scored_df.columns


def test_mission_risk_score_range():
    raw_df = generate_synthetic_events(num_records=100, random_state=42)
    cleaned_df = clean_data(raw_df)
    cleaned_df = add_time_features(cleaned_df)
    engineered_df = create_engineered_features(cleaned_df)

    scored_df = calculate_mission_risk_score(engineered_df)

    assert scored_df["mission_risk_score"].between(0, 100).all()