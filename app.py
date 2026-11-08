import os
import json
import html
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
import streamlit as st

# IMPORTANT: import features BEFORE loading the model. The saved model may contain
# features.DeltaRegressor, and joblib needs the module to be importable.
import features as F
import advisor


# ============================================================
# AIRWISE — LAHORE AIR QUALITY FORECAST
# ============================================================

st.set_page_config(
    page_title="AIRWISE | Lahore",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACT = ARTIFACTS / "smog_model.joblib"
METRICS = ARTIFACTS / "metrics.json"
TEST_PREDS = ARTIFACTS / "test_predictions.csv"

LAT = 31.5497
LON = 74.3436
LOCAL_TZ = ZoneInfo("Asia/Karachi")

ATTRIBUTION = (
    "Weather and air-quality data by Open-Meteo.com (CC BY 4.0). "
    "PM2.5 here is an atmospheric-model product, not a ground-sensor reading, "
    "so it may differ from regulatory monitoring stations."
)


# ============================================================
# DESIGN
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #eaf2f7;
    --muted: #9fb0bd;
    --panel: #122330;
    --mint: #63e6be;
    --line: rgba(178,205,220,.13);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: radial-gradient(
        ellipse at 12% 0%,
        #17384a 0%,
        #0b1721 43%,
        #08121a 100%
    );
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0b1a24;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] * {
    color: var(--ink);
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif;
    color: #f3f8fb;
}

.metric-card {
    background: linear-gradient(145deg,#172e3d,#10232f);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px;
    min-height: 135px;
}

.metric-label {
    color: #9fb3c0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .08em;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    color: #f4f8fb;
    font-size: 29px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-note {
    color: #9fb3c0;
    font-size: 12px;
    margin-top: 5px;
}

.section-head {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 21px;
    margin: 25px 0 14px;
    color: #f2f7fa;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


def esc(text):
    return html.escape(str(text))


def card(label, value, note="", size=None):
    style = f' style="font-size:{size}px;"' if size else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{esc(label)}</div>
        <div class="metric-value"{style}>{esc(value)}</div>
        <div class="metric-note">{esc(note)}</div>
    </div>
    """


st.html("""
<div style="
    padding:30px 32px;
    border:1px solid rgba(178,205,220,.13);
    border-radius:24px;
    background:linear-gradient(115deg,#1c4658,#122634 62%,#1b3d43);
    margin-bottom:24px;
">
    <div style="
        color:#63e6be;
        text-transform:uppercase;
        font-size:11px;
        letter-spacing:.19em;
        font-weight:700;
    ">
        AIR QUALITY RESEARCH PROTOTYPE / LAHORE
    </div>

    <div style="
        font-family:Manrope,sans-serif;
        font-size:clamp(30px,4vw,48px);
        line-height:1.1;
        font-weight:800;
        margin:15px 0;
        color:white;
    ">
        Tomorrow's air, estimated.<br>
        <span style="color:#83e8c5">
            A rough guide, not a warning.
        </span>
    </div>

    <div style="color:#c2d3dc;font-size:15px;max-width:720px;">
        A machine-learning experiment forecasting next-day PM2.5
        in Lahore from recent pollution and weather patterns.
        Always follow official advisories.
    </div>

    <div style="margin-top:20px;">
        <span style="display:inline-block;border:1px solid rgba(99,230,190,.3);background:rgba(99,230,190,.09);color:#aaf5d9;border-radius:99px;padding:7px 12px;font-size:11px;font-weight:700;margin-right:6px;">NEXT-DAY FORECAST</span>
        <span style="display:inline-block;border:1px solid rgba(99,230,190,.3);background:rgba(99,230,190,.09);color:#aaf5d9;border-radius:99px;padding:7px 12px;font-size:11px;font-weight:700;margin-right:6px;">ML POWERED</span>
        <span style="display:inline-block;border:1px solid rgba(99,230,190,.3);background:rgba(99,230,190,.09);color:#aaf5d9;border-radius:99px;padding:7px 12px;font-size:11px;font-weight:700;">ENGLISH / اردو</span>
    </div>
</div>
""")


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    if not ARTIFACT.exists():
        return None
    return joblib.load(ARTIFACT)


# ============================================================
# FETCH RECENT DATA (hourly; daily features are built by features.py)
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_recent_inputs():
    common = {
        "latitude": LAT,
        "longitude": LON,
        "past_days": 30,
        "forecast_days": 1,
        "timezone": "Asia/Karachi",
    }

    air_response = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={**common, "hourly": "pm2_5"},
        timeout=35,
    )
    air_response.raise_for_status()
    air_json = air_response.json()
    if "hourly" not in air_json:
        raise RuntimeError("Air-quality data is unavailable.")

    air_hourly = pd.DataFrame({
        "time": pd.to_datetime(air_json["hourly"]["time"]),
        "pm2_5": air_json["hourly"]["pm2_5"],
    })

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={**common, "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"},
        timeout=35,
    )
    weather_response.raise_for_status()
    weather_json = weather_response.json()
    if "hourly" not in weather_json:
        raise RuntimeError("Weather data is unavailable.")

    weather_hourly = pd.DataFrame({
        "time": pd.to_datetime(weather_json["hourly"]["time"]),
        "temperature": weather_json["hourly"]["temperature_2m"],
        "humidity": weather_json["hourly"]["relative_humidity_2m"],
        "wind": weather_json["hourly"]["wind_speed_10m"],
    })

    return air_hourly, weather_hourly


# ============================================================
# GenAI SETTINGS (secrets are read here, never written in code)
# ============================================================

def get_secret(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name) or ""


def llm_settings():
    """Returns (provider, api_key, model). Add ANTHROPIC_API_KEY or OPENAI_API_KEY as an app secret."""
    for provider, secret in (("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")):
        key = get_secret(secret)
        if key:
            return provider, key, get_secret("LLM_MODEL") or advisor.DEFAULT_MODELS[provider]
    return "anthropic", "", ""


@st.cache_data(ttl=3600, show_spinner=False)
def cached_advisory(value, lang, provider, model, _key):
    result = advisor.get_advisory(value, lang, api_key=_key, model=model, provider=provider)
    return result.text, result.source


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌿 AIRWISE")
    st.caption("INTELLIGENCE FOR CLEANER AIR")
    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Live forecast",
            "Model performance",
            "About the project"
        ]
    )

    st.markdown("---")

    language = st.selectbox(
        "Advisory language",
        ["English", "اردو"]
    )

    st.markdown("""
    <div style="font-size:13px;line-height:1.9;">
        <b>PROJECT AREA</b><br>
        Lahore, Punjab, Pakistan<br><br>

        <b>FORECAST HORIZON</b><br>
        Next calendar day<br><br>

        <b>MODEL FAMILY</b><br>
        Classical Machine Learning
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Prototype • Not an official warning service")

lang = "ur" if language == "اردو" else "en"


# ============================================================
# CHECK MODEL
# ============================================================

artifact = load_model()

if artifact is None:

    st.error(
        "The trained model is missing. Please check that "
        "artifacts/smog_model.joblib exists inside the AIRWISE folder."
    )

    st.stop()

if artifact.get("sklearn_version") and artifact["sklearn_version"] != sklearn.__version__:
    st.sidebar.warning(
        f"Model was trained with scikit-learn {artifact['sklearn_version']}, "
        f"but this app runs {sklearn.__version__}. Pin the version in requirements.txt."
    )


# ============================================================
# LIVE FORECAST
# ============================================================

if page == "Live forecast":

    st.markdown(
        '<div class="section-head">◉ &nbsp; Tomorrow at a glance</div>',
        unsafe_allow_html=True
    )

    try:

        with st.spinner("Connecting to Lahore air-quality and weather data..."):
            air_hourly, weather_hourly = fetch_recent_inputs()

        # Same feature code as the notebook: complete days only, calendar lags, stale-data check.
        daily, row = F.build_latest_features(air_hourly, weather_hourly)

        if row is None:
            raise RuntimeError(
                "The newest day of data is incomplete or out of date, so no forecast was made. "
                "The app never invents a reading."
            )

        last_day = pd.Timestamp(row["time"].iloc[0])
        tomorrow = (last_day + pd.Timedelta(days=1)).date()
        latest = row.iloc[0]

        X = row[artifact["features"]]
        forecast = max(0.0, float(artifact["model"].predict(X)[0]))
        band = F.pm_category(forecast)

        low_ratio, high_ratio = artifact.get("interval_ratio", [None, None])
        if low_ratio and high_ratio:
            range_note = f"likely range {forecast * low_ratio:.0f}–{forecast * high_ratio:.0f} µg/m³"
        else:
            range_note = "µg/m³ • next-day estimate"

        # METRIC CARDS

        c1, c2, c3, c4 = st.columns(4)

        cards = [
            ("PREDICTED PM2.5", f"{forecast:.0f}", range_note),
            ("AIR QUALITY CATEGORY", band, "US EPA PM2.5 categories (2024)"),
            ("LATEST DAILY MEAN", f"{latest['pm_today']:.0f}", f"µg/m³ • {last_day.date()}"),
            ("MODEL ENGINE", artifact.get("model_name", "ML model"), "Selected using training CV")
        ]

        for col, (label, value, note) in zip([c1, c2, c3, c4], cards):
            with col:
                st.html(card(label, value, note, size=24 if len(str(value)) > 12 else None))

        # CHART AND ADVISORY

        left, right = st.columns([1.35, 1])

        with left:

            st.markdown(
                '<div class="section-head">⌁ &nbsp; Recent pollution pattern and forecast</div>',
                unsafe_allow_html=True
            )

            recent = daily[["time", "pm2_5"]].dropna().tail(21).set_index("time")
            chart = recent.rename(columns={"pm2_5": "Daily PM2.5"})
            chart["Forecast"] = np.nan
            chart.loc[last_day, "Forecast"] = chart.loc[last_day, "Daily PM2.5"]  # join the lines
            chart = pd.concat([
                chart,
                pd.DataFrame({"Daily PM2.5": [np.nan], "Forecast": [forecast]},
                             index=[pd.Timestamp(tomorrow)])
            ])

            st.line_chart(chart, height=290)

            st.caption(
                f"Input data through {last_day.date()}. Daily means of hourly values; "
                "the final day's remaining hours come from the data provider's own forecast."
            )

        with right:

            st.markdown(
                '<div class="section-head">✦ &nbsp; Air-quality briefing</div>',
                unsafe_allow_html=True
            )

            provider, api_key, model = llm_settings()
            advice, source = cached_advisory(int(round(forecast)), lang, provider, model, api_key)

            source_label = (
                "AI-written, checked: no numbers other than the forecast"
                if source == "ai" else "Standard guidance (fixed message)"
            )
            direction = ' dir="rtl" style="text-align:right;"' if lang == "ur" else ""

            st.html(f"""
            <div style="
                background:linear-gradient(120deg,rgba(36,94,91,.34),rgba(22,52,62,.8));
                border:1px solid rgba(99,230,190,.22);
                border-radius:18px;
                padding:20px;
            ">
                <div style="color:#63e6be;font-size:11px;font-weight:700;">
                    FORECAST FOR {esc(tomorrow)}
                </div>
                <h3 style="margin:12px 0;">{esc(band)} conditions</h3>
                <div{direction} style="color:#d4e5e8;line-height:1.65;">
                    {esc(advice)}
                </div>
                <div{direction} style="margin-top:12px;color:#f0d9a8;font-size:12px;line-height:1.5;">
                    {esc(advisor.DISCLAIMER[lang])}
                </div>
                <div style="margin-top:12px;color:#8fb9ad;font-size:11px;">
                    {esc(source_label)}
                </div>
            </div>
            """)

            st.markdown("### Weather context")

            st.write(f"Temperature: **{latest['temperature']:.1f} °C**")
            st.write(f"Relative humidity: **{latest['humidity']:.0f}%**")
            st.write(f"Wind speed: **{latest['wind']:.1f} km/h**")

            # Optional feature: ask about the forecast (answers only from the data shown here)
            with st.expander("Ask about this forecast"):
                if not api_key:
                    st.caption("Add an API key as an app secret to enable this feature.")
                else:
                    question = st.text_input("Your question", placeholder="Why is tomorrow's forecast high?",
                                             max_chars=300)
                    if st.button("Ask") and question.strip():
                        last7 = ", ".join(f"{v:.0f}" for v in daily["pm2_5"].dropna().tail(7))
                        context = (
                            f"Forecast for {tomorrow}: {forecast:.0f} ug/m3, category {band}. "
                            f"Latest daily mean PM2.5: {latest['pm_today']:.0f}. "
                            f"PM2.5 daily means for the last 7 days: {last7}. "
                            f"7-day average: {latest['pm_roll7']:.0f}. "
                            f"Latest weather: temperature {latest['temperature']:.0f} C, "
                            f"humidity {latest['humidity']:.0f} percent, wind {latest['wind']:.0f} km/h. "
                            "The forecast comes from a statistical model that uses recent PM2.5 and weather."
                        )
                        with st.spinner("Thinking..."):
                            answer = advisor.answer_question(
                                question, context, lang, api_key=api_key, model=model, provider=provider)
                        st.write(answer)

        st.caption(
            "This is an educational research prototype, not an official "
            "Pakistan AQI conversion or public warning service."
        )

    except Exception as error:

        st.error(f"Live forecast is temporarily unavailable: {error}")

        st.info(
            "The application will not invent a reading. "
            "Please retry when the data service is available."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model performance":

    st.markdown(
        '<div class="section-head">▥ &nbsp; Historical model evaluation</div>',
        unsafe_allow_html=True
    )

    if METRICS.exists():

        with open(METRICS, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        a, b, c = st.columns(3)

        for column, (label, value) in zip([a, b, c], [
            ("SELECTED MODEL", metrics.get("selected_model", "—")),
            ("TEST MAE", f'{metrics.get("selected_model_test_mae", float("nan")):.2f} µg/m³'),
            ("BASELINE MAE (TOMORROW = TODAY)", f'{metrics.get("persistence_test_mae", float("nan")):.2f} µg/m³'),
        ]):
            with column:
                st.html(card(label, value, size=24))

        st.markdown("")

        gain_ci = metrics.get("gain_95ci")
        gain = metrics.get("gain_over_persistence_mae")

        if metrics.get("ml_beats_persistence") and gain_ci and gain_ci[0] > 0:
            st.success(
                f"The model beats the 'tomorrow = today' baseline by about {gain:.2f} µg/m³ per day "
                f"(95% interval {gain_ci[0]:.2f} to {gain_ci[1]:.2f}), which is unlikely to be luck."
            )
        elif metrics.get("ml_beats_persistence"):
            st.warning(
                "The model's average error is slightly lower than the baseline's, but the 95% interval "
                f"({gain_ci[0]:.2f} to {gain_ci[1]:.2f} µg/m³) includes zero, so the gain is within noise."
                if gain_ci else
                "The model's average error is slightly lower than the baseline's, but the difference may be noise."
            )
        else:
            st.warning("The selected ML model did not outperform the persistence baseline on the held-out period.")

        st.write(
            f'**Test period:** {metrics.get("test_start")} '
            f'to {metrics.get("test_end")} ({metrics.get("test_days")} days)'
        )

        # Predicted vs actual (promised in the proposal)
        if TEST_PREDS.exists():
            st.markdown("### Predicted vs actual (test period)")
            preds = pd.read_csv(TEST_PREDS, parse_dates=["time"]).set_index("time")
            st.line_chart(
                preds[["actual", "persistence", "model"]].rename(columns={
                    "actual": "Actual", "persistence": "Baseline", "model": "Model"}),
                height=300,
            )

        # High-smog days
        high = metrics.get("high_smog")
        if high:
            st.markdown(f"### High-smog days (PM2.5 ≥ {metrics.get('high_smog_threshold')} µg/m³)")
            n_high = metrics.get("high_smog_days", 0)
            st.write(f"Smog days in the test period: **{n_high}**")
            if n_high < 20:
                st.warning("Very few smog days in the test period, so these numbers are unreliable.")
            st.dataframe(pd.DataFrame(high).round(2))
            st.caption(
                "Negative bias means the model under-predicts smog. 'Missed days' are smog days the "
                "forecast did not flag. Treat the forecast as a rough guide on the worst days."
            )

        winters = metrics.get("winter_backtest")
        if winters:
            st.markdown("### Winter-by-winter backtest")
            st.dataframe(pd.DataFrame(winters).round(2))

        cv_rows = metrics.get("cv")
        if cv_rows:
            st.markdown("### Cross-validation results")
            cv_df = pd.DataFrame(cv_rows)
            if "Folds beating persistence" in cv_df:
                cv_df["Folds beating persistence"] = cv_df["Folds beating persistence"].astype(str)
            st.dataframe(cv_df.round(3))

        if metrics.get("likely_range_test_coverage") is not None:
            st.caption(
                f"The 'likely range' on the forecast page contained the real value on "
                f"{metrics['likely_range_test_coverage']:.0%} of test days."
            )

        st.caption(
            "Metrics are historical holdout results, not a guarantee "
            "of future forecast accuracy."
        )

    else:

        st.warning(
            "Evaluation metrics are unavailable. "
            "Check artifacts/metrics.json."
        )


# ============================================================
# ABOUT THE PROJECT
# ============================================================

else:

    st.markdown(
        '<div class="section-head">◎ &nbsp; About AIRWISE</div>',
        unsafe_allow_html=True
    )

    st.html("""
    <div style="
        background:rgba(18,35,48,.86);
        border:1px solid rgba(178,205,220,.13);
        border-radius:18px;
        padding:24px;
        line-height:1.8;
        color:#eaf2f7;
    ">
        <h3>Research question</h3>

        How much better than "tomorrow equals today" can a
        lightweight machine-learning model forecast next-day
        PM2.5 in Lahore?

        <hr style="border-color:rgba(178,205,220,.13);">

        <h3>Project pipeline</h3>

        Open-Meteo data → daily aggregation → data quality checks
        → lag and rolling features → chronological validation
        → baseline and ML comparison → bilingual advisory
        → interactive dashboard.

        <h3>Models</h3>

        Persistence baseline, Linear Regression, Random Forest and
        gradient boosting, including versions that predict the change
        from today. The best model is chosen by time-series
        cross-validation, never by the test set.

        <h3>Evaluation</h3>

        Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE),
        a bootstrap interval for the gain over the baseline,
        expanding-window time-series cross-validation,
        high-smog analysis and a winter-by-winter backtest.

        <h3>The AI advisory</h3>

        The language model only receives the forecast number and its
        category. Its reply is rejected if it contains any other number,
        and a fixed message is shown if anything fails.

        <h3>Responsible use</h3>

        This is an educational research prototype.
        It is not an official alert service and does not provide
        medical advice.
    </div>
    """)


st.caption(ATTRIBUTION)
