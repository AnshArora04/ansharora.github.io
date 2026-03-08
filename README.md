# UFC Betting Predictor

A Python CLI tool that predicts UFC fight outcomes using ML and cross-references against live [Kalshi](https://kalshi.com) prediction market odds to surface value bets.

## How it works

1. **Scrapes** upcoming UFC fight cards + fighter stats from [ufcstats.com](http://ufcstats.com)
2. **Predicts** fight winners using a Gradient Boosting ML model (~63% accuracy)
3. **Fetches** live Kalshi market odds for upcoming fights
4. **Computes** the edge: `model_probability - kalshi_implied_probability`
5. **Flags** bets where the model sees >5% edge over the market

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your Kalshi API key
cp .env.example .env
# Edit .env and set KALSHI_API_KEY=your_key_here
```

## Usage

```bash
# Analyze the next upcoming UFC card
python -m src.main

# Train/retrain the model from scratch (first run, or after clearing cache)
python -m src.main --train

# Skip Kalshi (model predictions only, no API key needed)
python -m src.main --no-kalshi

# Filter to a specific event
python -m src.main --event "UFC 312"
```

## Output

```
╭────────────────────────────────────────────────────────────────────────────╮
│  UFC Betting Predictor                                                     │
│  ML predictions + Kalshi market odds → value bets                         │
╰────────────────────────────────────────────────────────────────────────────╯

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Fight Card Analysis                               │
├───────────────────────┬──────────┬──────────────┬──────┬───────┬───────────┤
│ Matchup               │ Weight   │ Model Pick   │ Conf │ Model │ Kalshi    │
├───────────────────────┼──────────┼──────────────┼──────┼───────┼───────────┤
│ Jon Jones             │          │              │      │       │           │
│ vs.                   │ HW       │ Jon Jones    │ 68%  │  68%  │  55%  +13%│ ✅ BET Jon Jones
│ Stipe Miocic          │          │              │      │       │           │
└───────────────────────┴──────────┴──────────────┴──────┴───────┴───────────┘
```

## Features

- **Fighter stats**: win%, reach, height, experience from ufcstats.com
- **ML model**: Gradient Boosting Classifier with 5-fold cross-validation
- **Caching**: Fighter stats and fight history cached locally to `data/raw/`
- **Smart matching**: Fuzzy name matching between UFC fighters and Kalshi markets

## Data sources

- Fighter stats: [ufcstats.com](http://ufcstats.com)
- Market odds: [Kalshi API](https://docs.kalshi.com)

## Disclaimer

Model accuracy is approximately 63% — better than random but far from certain. This tool is for informational and educational purposes only. Bet responsibly.
