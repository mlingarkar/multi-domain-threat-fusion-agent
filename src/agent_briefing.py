"""
Agentic Mission Briefing Module

This module generates analyst-style mission risk briefings from the
scored multi-domain mission dataset.

If an OpenAI API key is available, the module can use an LLM to generate
a more natural briefing. If no API key is available, it automatically
uses a local rule-based briefing generator.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv


def identify_top_risk_events(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Identify the highest-risk mission events.

    Args:
        df: Scored mission-event DataFrame.
        top_n: Number of high-risk events to return.

    Returns:
        DataFrame containing the top high-risk events.
    """
    required_column = "mission_risk_score"

    if required_column not in df.columns:
        raise ValueError("mission_risk_score column not found. Run risk scoring first.")

    return df.sort_values("mission_risk_score", ascending=False).head(top_n)


def identify_domain_risk_drivers(row: pd.Series) -> list:
    """
    Identify likely domain-level risk drivers for a single event.

    Args:
        row: A single event row.

    Returns:
        List of readable risk-driver strings.
    """
    drivers = []

    if row.get("radar_health_score", 100) < 45:
        drivers.append("radar degradation")

    if row.get("satellite_stability_score", 100) < 45:
        drivers.append("satellite telemetry instability")

    if row.get("communication_reliability_score", 100) < 55:
        drivers.append("communications disruption")

    if row.get("cyber_pressure_index", 0) > 35:
        drivers.append("elevated cyber pressure")

    if row.get("uav_operational_score", 100) < 45:
        drivers.append("reduced UAV operational health")

    if row.get("weather_severity", 0) >= 4:
        drivers.append("severe environmental conditions")

    if not drivers:
        drivers.append("general multi-domain pressure")

    return drivers


def create_briefing_context(df: pd.DataFrame) -> str:
    """
    Create structured context for the mission briefing.

    Args:
        df: Final mission-event DataFrame.

    Returns:
        Structured text context for either local or LLM briefing generation.
    """
    top_events = identify_top_risk_events(df, top_n=5)

    total_events = len(df)
    anomaly_count = int(df["anomaly_label"].sum()) if "anomaly_label" in df.columns else 0
    average_risk = round(df["mission_risk_score"].mean(), 2)

    risk_distribution = (
        df["risk_level"]
        .value_counts()
        .to_dict()
        if "risk_level" in df.columns
        else {}
    )

    lines = [
        "Mission Dataset Summary",
        "-----------------------",
        f"Total events analyzed: {total_events}",
        f"Known synthetic anomaly count: {anomaly_count}",
        f"Average mission risk score: {average_risk}",
        f"Risk level distribution: {risk_distribution}",
        "",
        "Highest-Risk Mission Events",
        "---------------------------",
    ]

    for _, row in top_events.iterrows():
        drivers = identify_domain_risk_drivers(row)

        lines.extend([
            f"Event ID: {row.get('event_id', 'Unknown')}",
            f"Timestamp: {row.get('timestamp', 'Unknown')}",
            f"Mission Phase: {row.get('mission_phase', 'Unknown')}",
            f"Region Type: {row.get('region_type', 'Unknown')}",
            f"Anomaly Type: {row.get('anomaly_type', 'Unknown')}",
            f"Mission Risk Score: {row.get('mission_risk_score', 'Unknown')}",
            f"Risk Level: {row.get('risk_level', 'Unknown')}",
            f"ML Anomaly Probability: {row.get('ml_anomaly_probability', 'Not Available')}",
            f"Likely Risk Drivers: {', '.join(drivers)}",
            "",
        ])

    return "\n".join(lines)


def generate_local_briefing(df: pd.DataFrame) -> str:
    """
    Generate a local rule-based mission briefing.

    Args:
        df: Final mission-event DataFrame.

    Returns:
        Mission briefing text.
    """
    top_events = identify_top_risk_events(df, top_n=5)
    highest_event = top_events.iloc[0]
    drivers = identify_domain_risk_drivers(highest_event)

    total_events = len(df)
    anomaly_count = int(df["anomaly_label"].sum()) if "anomaly_label" in df.columns else 0
    average_risk = round(df["mission_risk_score"].mean(), 2)

    high_or_critical_count = len(df[df["risk_level"].isin(["High", "Critical"])])

    briefing = [
        "AI Mission Risk Briefing",
        "========================",
        "",
        "Executive Summary",
        "-----------------",
        (
            f"The system analyzed {total_events} synthetic multi-domain mission events. "
            f"The dataset contains {anomaly_count} known synthetic anomalies, with an "
            f"average mission risk score of {average_risk}."
        ),
        "",
        (
            f"{high_or_critical_count} events were classified as High or Critical risk. "
            "These events should be prioritized for analyst review because they show "
            "elevated multi-domain stress across one or more mission systems."
        ),
        "",
        "Highest-Risk Event",
        "------------------",
        f"Event ID: {highest_event.get('event_id', 'Unknown')}",
        f"Timestamp: {highest_event.get('timestamp', 'Unknown')}",
        f"Mission Phase: {highest_event.get('mission_phase', 'Unknown')}",
        f"Region Type: {highest_event.get('region_type', 'Unknown')}",
        f"Anomaly Type: {highest_event.get('anomaly_type', 'Unknown')}",
        f"Mission Risk Score: {highest_event.get('mission_risk_score', 'Unknown')}",
        f"Risk Level: {highest_event.get('risk_level', 'Unknown')}",
        "",
        "Primary Risk Drivers",
        "--------------------",
    ]

    for driver in drivers:
        briefing.append(f"- {driver.title()}")

    briefing.extend([
        "",
        "Recommended Analyst Actions",
        "---------------------------",
        "1. Review the highest-risk event windows and validate whether the anomalies cluster across multiple domains.",
        "2. Compare radar, satellite, communications, cyber, UAV, and environmental indicators for the same timestamp range.",
        "3. Prioritize High and Critical events that show simultaneous domain stress rather than isolated signal noise.",
        "4. Use the model outputs as decision-support indicators, not as autonomous operational decisions.",
        "",
        "Portfolio Safety Note",
        "---------------------",
        "This briefing was generated from synthetic data for demonstration purposes only. "
        "It is designed to show AI-assisted risk analysis, anomaly detection, and analyst-support workflows."
    ])

    return "\n".join(briefing)


def generate_openai_briefing(df: pd.DataFrame, model: str = "gpt-4o-mini") -> Optional[str]:
    """
    Generate a mission briefing using the OpenAI API if an API key is available.

    Args:
        df: Final mission-event DataFrame.
        model: OpenAI model name.

    Returns:
        AI-generated briefing text, or None if unavailable.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        context = create_briefing_context(df)

        prompt = f"""
You are an AI mission-risk analyst assistant.

Generate a professional, concise mission risk briefing from the structured synthetic dataset context below.

Important constraints:
- Do not recommend weapons usage.
- Do not provide targeting instructions.
- Do not claim the data is real.
- Frame all conclusions as analyst-support observations.
- Include executive summary, primary risk drivers, highest-risk events, and recommended analyst review actions.

Context:
{context}
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You generate safe analyst-support briefings from synthetic multi-domain risk data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as error:
        print(f"OpenAI briefing generation failed. Using local fallback. Error: {error}")
        return None


def generate_mission_briefing(df: pd.DataFrame, use_openai: bool = True) -> str:
    """
    Generate a mission briefing using OpenAI when available, otherwise local fallback.

    Args:
        df: Final mission-event DataFrame.
        use_openai: Whether to attempt OpenAI briefing generation.

    Returns:
        Mission briefing text.
    """
    if use_openai:
        openai_briefing = generate_openai_briefing(df)

        if openai_briefing:
            return openai_briefing

    return generate_local_briefing(df)


def save_briefing(
    briefing_text: str,
    output_path: str = "outputs/reports/ai_mission_briefing.txt"
) -> None:
    """
    Save briefing text to a report file.

    Args:
        briefing_text: Mission briefing content.
        output_path: File path for the report.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(briefing_text)


if __name__ == "__main__":
    from preprocessing import load_data, clean_data, add_time_features
    from feature_engineering import create_engineered_features
    from risk_scoring import calculate_mission_risk_score

    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    cleaned_df = add_time_features(cleaned_df)

    engineered_df = create_engineered_features(cleaned_df)
    scored_df = calculate_mission_risk_score(engineered_df)

    # Add a placeholder ML probability for standalone testing.
    scored_df["ml_anomaly_probability"] = scored_df["mission_risk_score"] / 100

    briefing = generate_mission_briefing(scored_df, use_openai=True)
    save_briefing(briefing)

    print("Mission briefing generated successfully.")
    print("Saved to: outputs/reports/ai_mission_briefing.txt")
    print()
    print(briefing[:1000])