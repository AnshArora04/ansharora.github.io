"""
Combines ML model predictions with Kalshi market odds to identify value bets.
"""

from typing import Optional
import pandas as pd
from src import kalshi as kalshi_client
from src.model import predict_fight, load_model

# Minimum edge (model prob - market implied prob) to flag as a value bet
VALUE_BET_THRESHOLD = 0.05  # 5%


def lookup_fighter(name: str, fighters_df: pd.DataFrame) -> dict:
    """Find a fighter in the DataFrame by name (case-insensitive, partial match)."""
    name_lower = name.lower()
    # Exact match first
    exact = fighters_df[fighters_df["name"].str.lower() == name_lower]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    # Last name match
    last_name = name_lower.split()[-1]
    partial = fighters_df[fighters_df["name"].str.lower().str.contains(last_name, na=False)]
    if not partial.empty:
        return partial.iloc[0].to_dict()

    return {}


def analyze_card(
    fights: list[dict],
    fighters_df: pd.DataFrame,
    model,
    ufc_markets: list[dict],
) -> list[dict]:
    """
    Analyze a full fight card and return bet recommendations.

    Each result dict:
    {
        "fighter1": str,
        "fighter2": str,
        "weight_class": str,
        "event": str,
        "model_f1_prob": float,
        "model_f2_prob": float,
        "predicted_winner": str,
        "confidence": float,
        "kalshi_ticker": str | None,
        "kalshi_f1_prob": float | None,   # market-implied prob for fighter1
        "kalshi_f2_prob": float | None,
        "f1_edge": float | None,          # model_f1_prob - kalshi_f1_prob
        "f2_edge": float | None,
        "bet_recommendation": str,        # "BET F1", "BET F2", "PASS", "NO MARKET"
        "bet_on": str | None,
        "bet_edge": float | None,
    }
    """
    print(f"\n[predictor] Analyzing {len(fights)} fights...")
    results = []

    for fight in fights:
        f1_name = fight["fighter1"]
        f2_name = fight["fighter2"]
        weight = fight.get("weight_class", "")
        event = fight.get("event", "")

        f1_stats = lookup_fighter(f1_name, fighters_df)
        f2_stats = lookup_fighter(f2_name, fighters_df)

        # Model prediction
        if model and (f1_stats or f2_stats):
            pred = predict_fight(f1_stats, f2_stats, model)
            model_f1_prob = pred["fighter1_prob"]
            model_f2_prob = pred["fighter2_prob"]
            predicted_winner = f1_name if pred["predicted_winner"] == "fighter1" else f2_name
            confidence = pred["confidence"]
        else:
            model_f1_prob = model_f2_prob = 0.5
            predicted_winner = "Unknown"
            confidence = 0.5

        # Kalshi market lookup
        market = kalshi_client.match_fighters_to_market(f1_name, f2_name, ufc_markets)
        kalshi_ticker = None
        kalshi_f1_prob = None
        kalshi_f2_prob = None
        f1_edge = None
        f2_edge = None
        bet_recommendation = "NO MARKET"
        bet_on = None
        bet_edge = None

        if market:
            ticker = market.get("ticker")
            kalshi_ticker = ticker
            odds = kalshi_client.get_market_odds(ticker) if ticker else None

            if odds and odds.get("implied_prob_yes") is not None:
                parsed = market.get("parsed_fighters")
                if parsed:
                    # Determine which fighter is the "yes" side
                    yes_fighter_name, no_fighter_name = parsed
                    yes_is_f1 = _names_match(yes_fighter_name, f1_name)

                    if yes_is_f1:
                        kalshi_f1_prob = odds["implied_prob_yes"]
                        kalshi_f2_prob = odds["implied_prob_no"]
                    else:
                        kalshi_f1_prob = odds["implied_prob_no"]
                        kalshi_f2_prob = odds["implied_prob_yes"]
                else:
                    # Assume yes=fighter1 if we can't parse
                    kalshi_f1_prob = odds["implied_prob_yes"]
                    kalshi_f2_prob = odds["implied_prob_no"]

                if kalshi_f1_prob is not None and kalshi_f2_prob is not None:
                    f1_edge = model_f1_prob - kalshi_f1_prob
                    f2_edge = model_f2_prob - kalshi_f2_prob

                    if f1_edge >= VALUE_BET_THRESHOLD:
                        bet_recommendation = f"BET {f1_name}"
                        bet_on = f1_name
                        bet_edge = f1_edge
                    elif f2_edge >= VALUE_BET_THRESHOLD:
                        bet_recommendation = f"BET {f2_name}"
                        bet_on = f2_name
                        bet_edge = f2_edge
                    else:
                        bet_recommendation = "PASS"

        results.append({
            "fighter1": f1_name,
            "fighter2": f2_name,
            "weight_class": weight,
            "event": event,
            "model_f1_prob": model_f1_prob,
            "model_f2_prob": model_f2_prob,
            "predicted_winner": predicted_winner,
            "confidence": confidence,
            "kalshi_ticker": kalshi_ticker,
            "kalshi_f1_prob": kalshi_f1_prob,
            "kalshi_f2_prob": kalshi_f2_prob,
            "f1_edge": f1_edge,
            "f2_edge": f2_edge,
            "bet_recommendation": bet_recommendation,
            "bet_on": bet_on,
            "bet_edge": bet_edge,
        })

    return results


def _names_match(name_a: str, name_b: str) -> bool:
    """Fuzzy name match — checks if last names match."""
    a_last = name_a.lower().split()[-1] if name_a else ""
    b_last = name_b.lower().split()[-1] if name_b else ""
    return a_last == b_last or name_a.lower() in name_b.lower() or name_b.lower() in name_a.lower()
