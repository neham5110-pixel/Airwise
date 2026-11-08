# AIRWISE: Next-Day PM2.5 (Smog) Forecast for Lahore

A machine-learning experiment that forecasts tomorrow's PM2.5 in Lahore from recent pollution and weather,
tests it fairly against a "tomorrow = today" baseline, and turns the forecast into short English/Urdu advice.

**Live app:** _paste your Streamlit link here_

## Results
See the **Model performance** page of the app, or `artifacts/metrics.json`. Summary (fill in from your own run):
- Test MAE (model vs baseline): _..._
- 95% interval for the gain over the baseline: _..._
- High-smog days caught / missed in the test period: _..._

## Honest limitations
- PM2.5 comes from Open-Meteo, an atmospheric-model product, not ground sensors.
- The model is at best a modest improvement over "tomorrow = today" and tends to under-predict extreme smog days.
- One city, a few winters of data. Not an official warning service and not medical advice.

## Repository layout
```
app.py                 Streamlit app
features.py            feature building + AQI categories + delta model (shared with the notebook)
advisor.py             GenAI advisory with safeguards and fixed fallback messages
requirements.txt
artifacts/             smog_model.joblib, metrics.json, test_predictions.csv, ... (from the notebook's zip)
AIRWISE_smog_forecast_complete.ipynb
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub (never commit API keys).
2. share.streamlit.io -> New app -> pick the repo, branch `main`, main file `app.py`.
3. Advanced settings -> Secrets, add ONE of:
   ```
   ANTHROPIC_API_KEY = "your-key"
   # or
   OPENAI_API_KEY = "your-key"
   ```
   Without a key the app still works and shows the fixed advice messages.

## Data and licence
Weather and air-quality data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).
