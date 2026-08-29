NIFTY 50 AI TRADING SYSTEM V5.1

Adds a live confirmation layer:
- Upstox full market quote: live LTP + depth / buy-sell quantities when available.
- Upstox News API: recent instrument news (up to the provider's available history) with a lightweight sentiment score.
- ML + learned pattern model + trend/RSI + market-depth + news confluence.
- Entry / Stop Loss / Target 1 / Target 2 shown on chart and dashboard.
- Live IST clock over the chart.
- Prediction remains paper-trading/research only.

IMPORTANT:
Live news and order-book information are used as a confirmation layer, not as historical training features, because the existing historical dataset does not contain synchronized historical news/depth. No honest system can claim a guaranteed accuracy increase without backtesting those signals on historical data.

Run:
python train_model.py
python app.py
http://127.0.0.1:5000
