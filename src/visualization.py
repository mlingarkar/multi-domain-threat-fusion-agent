"""
Visualization Module

This module creates charts for the multi-domain mission risk analysis.
Charts are saved to the outputs/figures directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_output_directory(output_dir: str = "outputs/figures") -> None:
    """
    Create the output directory if it does not already exist.

    Args:
        output_dir: Directory where figures will be saved.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def plot_risk_level_distribution(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/risk_level_distribution.png"
) -> None:
    """
    Plot the distribution of mission risk levels.

    Args:
        df: DataFrame containing risk_level.
        output_path: Figure output path.
    """
    ensure_output_directory()

    risk_order = ["Low", "Moderate", "High", "Critical"]
    counts = df["risk_level"].value_counts().reindex(risk_order, fill_value=0)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Mission Risk Level Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Event Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_anomaly_timeline(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/anomaly_timeline.png"
) -> None:
    """
    Plot mission risk score over time with anomaly events.

    Args:
        df: DataFrame containing timestamp, mission_risk_score, and anomaly_label.
        output_path: Figure output path.
    """
    ensure_output_directory()

    timeline_df = df.copy()
    timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"], errors="coerce")

    plt.figure(figsize=(12, 5))
    plt.plot(timeline_df["timestamp"], timeline_df["mission_risk_score"], linewidth=1)

    anomalies = timeline_df[timeline_df["anomaly_label"] == 1]
    plt.scatter(
        anomalies["timestamp"],
        anomalies["mission_risk_score"],
        s=18,
        label="Synthetic Anomaly"
    )

    plt.title("Mission Risk Timeline with Synthetic Anomalies")
    plt.xlabel("Timestamp")
    plt.ylabel("Mission Risk Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_domain_risk_contribution(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/domain_risk_contribution.png"
) -> None:
    """
    Plot average domain risk indicators.

    Args:
        df: DataFrame containing engineered domain risk features.
        output_path: Figure output path.
    """
    ensure_output_directory()

    domain_columns = {
        "Radar Degradation": 100 - df["radar_health_score"],
        "Satellite Instability": 100 - df["satellite_stability_score"],
        "UAV Degradation": 100 - df["uav_operational_score"],
        "Comms Disruption": 100 - df["communication_reliability_score"],
        "Cyber Pressure": df["cyber_pressure_index"],
        "Environmental Risk": df["environmental_risk_index"],
    }

    contribution_df = pd.DataFrame(domain_columns)
    averages = contribution_df.mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    averages.plot(kind="bar")
    plt.title("Average Domain Risk Contribution")
    plt.xlabel("Domain Indicator")
    plt.ylabel("Average Risk Contribution")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_mission_risk_heatmap(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/mission_risk_heatmap.png"
) -> None:
    """
    Plot average mission risk by mission phase and region type.

    Args:
        df: DataFrame containing mission_phase, region_type, and mission_risk_score.
        output_path: Figure output path.
    """
    ensure_output_directory()

    heatmap_data = df.pivot_table(
        values="mission_risk_score",
        index="mission_phase",
        columns="region_type",
        aggfunc="mean"
    )

    plt.figure(figsize=(9, 5))
    plt.imshow(heatmap_data, aspect="auto")

    plt.title("Average Mission Risk by Phase and Region")
    plt.xlabel("Region Type")
    plt.ylabel("Mission Phase")

    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=35, ha="right")
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)

    plt.colorbar(label="Average Mission Risk Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_all_visualizations(df: pd.DataFrame) -> None:
    """
    Create all project visualizations.

    Args:
        df: Final scored mission-event DataFrame.
    """
    plot_risk_level_distribution(df)
    plot_anomaly_timeline(df)
    plot_domain_risk_contribution(df)
    plot_mission_risk_heatmap(df)


if __name__ == "__main__":
    from preprocessing import load_data, clean_data, add_time_features
    from feature_engineering import create_engineered_features
    from risk_scoring import calculate_mission_risk_score

    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    cleaned_df = add_time_features(cleaned_df)

    engineered_df = create_engineered_features(cleaned_df)
    scored_df = calculate_mission_risk_score(engineered_df)

    create_all_visualizations(scored_df)

    print("Visualizations created successfully.")
    print("Saved to: outputs/figures/")