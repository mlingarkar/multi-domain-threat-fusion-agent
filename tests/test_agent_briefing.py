"""
Unit tests for the AI mission briefing workflow.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from data_generator import generate_synthetic_events  # noqa: E402
from preprocessing import clean_data, add_time_features  # noqa: E402
from feature_engineering import create_engineered_features  # noqa: E402
from risk_scoring import calculate_mission_risk_score  # noqa: E402
from agent_briefing import generate_local_briefing, identify_top_risk_events  # noqa: E402


def create_test_scored_dataframe():
    raw_df = generate_synthetic_events(num_records=100, random_state=42)
    cleaned_df = clean_data(raw_df)
    cleaned_df = add_time_features(cleaned_df)
    engineered_df = create_engineered_features(cleaned_df)
    scored_df = calculate_mission_risk_score(engineered_df)

    scored_df["ml_anomaly_probability"] = scored_df["mission_risk_score"] / 100

    return scored_df


def test_identify_top_risk_events_returns_expected_count():
    scored_df = create_test_scored_dataframe()

    top_events = identify_top_risk_events(scored_df, top_n=5)

    assert len(top_events) == 5
    assert top_events["mission_risk_score"].is_monotonic_decreasing


def test_generate_local_briefing_returns_text():
    scored_df = create_test_scored_dataframe()

    briefing = generate_local_briefing(scored_df)

    assert isinstance(briefing, str)
    assert "AI Mission Risk Briefing" in briefing
    assert "Executive Summary" in briefing
    assert "Recommended Analyst Actions" in briefing