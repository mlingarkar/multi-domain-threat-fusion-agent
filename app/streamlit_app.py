"""
Streamlit Dashboard

Interactive dashboard for the Multi-Domain Threat Fusion Agent project.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from main import run_pipeline  # noqa: E402


ASSESSMENT_PATH = PROJECT_ROOT / "outputs" / "reports" / "mission_risk_assessment.csv"
BRIEFING_PATH = PROJECT_ROOT / "outputs" / "reports" / "ai_mission_briefing.txt"
MODEL_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "reports" / "model_summary.txt"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


st.set_page_config(
    page_title="Multi-Domain Threat Fusion Agent",
    layout="wide",
)


@st.cache_data
def load_assessment_data() -> pd.DataFrame:
    """
    Load the final mission risk assessment dataset.
    """
    return pd.read_csv(ASSESSMENT_PATH)


def load_text_file(path: Path) -> str:
    """
    Load a text file if it exists.
    """
    if path.exists():
        return path.read_text(encoding="utf-8")

    return "Report not found. Run the pipeline first."


def display_metric_cards(df: pd.DataFrame) -> None:
    """
    Display high-level project metrics.
    """
    total_events = len(df)
    anomaly_count = int(df["anomaly_label"].sum())
    average_risk = round(df["mission_risk_score"].mean(), 2)
    high_critical_count = len(df[df["risk_level"].isin(["High", "Critical"])])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Events", total_events)
    col2.metric("Synthetic Anomalies", anomaly_count)
    col3.metric("Average Risk Score", average_risk)
    col4.metric("High/Critical Events", high_critical_count)


def display_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display dashboard filters and return filtered DataFrame.
    """
    st.sidebar.header("Filters")

    risk_levels = sorted(df["risk_level"].unique())
    selected_risks = st.sidebar.multiselect(
        "Risk Level",
        options=risk_levels,
        default=risk_levels,
    )

    mission_phases = sorted(df["mission_phase"].unique())
    selected_phases = st.sidebar.multiselect(
        "Mission Phase",
        options=mission_phases,
        default=mission_phases,
    )

    anomaly_types = sorted(df["anomaly_type"].unique())
    selected_anomalies = st.sidebar.multiselect(
        "Anomaly Type",
        options=anomaly_types,
        default=anomaly_types,
    )

    minimum_risk = st.sidebar.slider(
        "Minimum Mission Risk Score",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
    )

    filtered_df = df[
        (df["risk_level"].isin(selected_risks))
        & (df["mission_phase"].isin(selected_phases))
        & (df["anomaly_type"].isin(selected_anomalies))
        & (df["mission_risk_score"] >= minimum_risk)
    ]

    return filtered_df


def display_figures() -> None:
    """
    Display saved project figures.
    """
    st.subheader("Generated Visualizations")

    figure_files = [
        "risk_level_distribution.png",
        "anomaly_timeline.png",
        "domain_risk_contribution.png",
        "mission_risk_heatmap.png",
    ]

    for figure_name in figure_files:
        figure_path = FIGURES_DIR / figure_name

        if figure_path.exists():
            st.image(
                str(figure_path),
                caption=figure_name.replace("_", " ").replace(".png", "").title(),
            )
        else:
            st.warning(f"Missing figure: {figure_name}")


def main() -> None:
    """
    Run the Streamlit dashboard.
    """
    st.title("Multi-Domain Threat Fusion Agent")

    st.caption(
        "Synthetic AI/ML defense-style risk analysis dashboard for radar, satellite, cyber, "
        "communications, UAV, and environmental mission signals."
    )

    st.info(
        "This project uses fully synthetic data for portfolio demonstration purposes. "
        "It is designed for safe AI-assisted anomaly detection and analyst-support workflows."
    )

    st.sidebar.header("Pipeline Controls")

    use_openai = st.sidebar.checkbox(
        "Use OpenAI briefing if API key is available",
        value=True,
    )

    if use_openai:
        st.sidebar.caption(
            "When checked, the pipeline will try to use your OPENAI_API_KEY from the .env file. "
            "If no key is found, it will automatically use the local fallback briefing."
        )
    else:
        st.sidebar.caption(
            "When unchecked, the project will use the local rule-based briefing generator."
        )

    if st.sidebar.button("Run Full Pipeline"):
        with st.spinner("Running full pipeline..."):
            run_pipeline(use_openai=use_openai)

        st.success("Pipeline completed. Refreshing dashboard data.")
        st.cache_data.clear()

    if not ASSESSMENT_PATH.exists():
        st.warning("Mission risk assessment file not found. Click 'Run Full Pipeline' in the sidebar first.")
        return

    df = load_assessment_data()
    filtered_df = display_filters(df)

    display_metric_cards(filtered_df)

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Mission Assessment",
            "High-Risk Events",
            "AI Briefing",
            "Model Summary",
            "Visualizations",
        ]
    )

    with tab1:
        st.subheader("Filtered Mission Risk Assessment")
        st.dataframe(filtered_df, use_container_width=True)

    with tab2:
        st.subheader("Highest-Risk Events")
        top_events = filtered_df.sort_values("mission_risk_score", ascending=False).head(10)

        columns_to_show = [
            "event_id",
            "timestamp",
            "mission_phase",
            "region_type",
            "anomaly_type",
            "mission_risk_score",
            "risk_level",
            "ml_anomaly_probability",
            "radar_health_score",
            "satellite_stability_score",
            "communication_reliability_score",
            "cyber_pressure_index",
        ]

        available_columns = [
            column for column in columns_to_show
            if column in top_events.columns
        ]

        st.dataframe(top_events[available_columns], use_container_width=True)

    with tab3:
        st.subheader("AI-Assisted Mission Briefing")
        briefing_text = load_text_file(BRIEFING_PATH)
        st.text_area("Briefing Output", briefing_text, height=500)

    with tab4:
        st.subheader("Machine Learning Model Summary")
        model_summary = load_text_file(MODEL_SUMMARY_PATH)
        st.text_area("Model Summary", model_summary, height=500)

    with tab5:
        display_figures()


if __name__ == "__main__":
    main()