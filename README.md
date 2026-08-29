# NIFTY 50 AI Trading System — Clean Refactor

There is only one current application. No V2/V3/V4/V5 compatibility code is included.

## Run

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Flow

1. Paste the Upstox Analytics token.
2. Click **Test & Save Token**.
3. Chart immediately loads real Upstox 5-minute candles.
4. **Download 3 Years** fills SQLite in 30-day API chunks.
5. **Train Model** trains the single current model set.
6. **Predict Now** runs live prediction.
7. Chart refreshes every minute; prediction every 5 minutes.

## Historical chart fix

The chart does NOT depend on SQLite being populated.

- 1D uses Upstox V3 intraday candles.
- 5D / 1M / 3M use Upstox V3 historical candles.
- Longer ranges are split into <=30-day requests because Upstox V3 limits 1–15 minute historical queries to one month.
- SQLite is only a fallback if Upstox returns no candles.

Upstox V3 documents the historical endpoint and the one-month limit for 1–15 minute intervals. It also documents the V3 intraday endpoint used for the current trading day.

## Token behavior

The token is held only in Flask server memory. It is never returned by an API and is never written to the browser/localStorage. If Flask is restarted, enter/test the token again.

## Models

Only these current artifacts are used:

- `models/model_5m.pkl`
- `models/model_10m.pkl`
- `models/model_direction.pkl`
- `models/feature_columns.json`
- `models/metadata.json`

Old V2/V3/V4/V5 filenames are intentionally ignored.
