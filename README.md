# Deriv V100 CFD Bot (Demo)

Directional trading bot for Deriv synthetic indices using CFD-style multipliers.
Targets Volatility 100 (R_100) with 5m/15m strategy signals.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in your Deriv credentials in `.env`:

```env
DERIV_APP_ID=YOUR_APP_ID
DERIV_DEMO_TOKEN=YOUR_DEMO_TOKEN
DERIV_SYMBOL=R_100
DERIV_GRANULARITY=300
DERIV_MULTIPLIER=100
DERIV_STAKE=10
DRY_RUN=true
```

Run:

```bash
python -m src.simple_arb_bot
```

## Notes

- `DRY_RUN=true` avoids live orders. Use `false` to place demo trades.
- Strategies are long-only (trend-follow + mean-reversion) and evaluated on candle closes.
- Candle granularity: `300` (5m) or `900` (15m).
