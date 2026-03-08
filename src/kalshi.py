"""
Kalshi API client — fetches UFC prediction markets and odds.

Docs: https://docs.kalshi.com/api-reference/market/get-markets
Base URL: https://api.elections.kalshi.com/trade-api/v2

Public market data endpoints don't require authentication.
The API key is used for authenticated endpoints if needed in future.
"""

import os
import re
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
UFC_KEYWORDS = ["ufc", "mma", "fight", "fighter", "bout"]


def _get_headers() -> dict:
    api_key = os.getenv("KALSHI_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["KALSHI-ACCESS-KEY"] = api_key
    return headers


def get_all_markets(status: str = "open", limit: int = 1000) -> list[dict]:
    """
    Fetch all open markets from Kalshi (paginated).
    Returns list of market dicts.
    """
    url = f"{BASE_URL}/markets"
    all_markets = []
    cursor = None

    while True:
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(url, headers=_get_headers(), params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            print(f"[kalshi] HTTP error fetching markets: {e}")
            break
        except Exception as e:
            print(f"[kalshi] Error fetching markets: {e}")
            break

        markets = data.get("markets", [])
        all_markets.extend(markets)

        cursor = data.get("cursor")
        if not cursor or not markets:
            break

    return all_markets


def get_ufc_markets() -> list[dict]:
    """
    Filter all open Kalshi markets to find UFC/MMA fight markets.
    Returns list of market dicts with extra 'parsed_fighters' field.
    """
    print("[kalshi] Fetching all open markets...")
    all_markets = get_all_markets()
    print(f"[kalshi] Total open markets: {len(all_markets)}")

    ufc_markets = []
    for market in all_markets:
        title = (market.get("title", "") or "").lower()
        subtitle = (market.get("subtitle", "") or "").lower()
        ticker = (market.get("ticker", "") or "").lower()
        event_ticker = (market.get("event_ticker", "") or "").lower()

        text = f"{title} {subtitle} {ticker} {event_ticker}"
        if any(kw in text for kw in UFC_KEYWORDS):
            market["parsed_fighters"] = _parse_fighters_from_title(
                market.get("title", "") or market.get("subtitle", "")
            )
            ufc_markets.append(market)

    print(f"[kalshi] Found {len(ufc_markets)} UFC/MMA markets.")
    return ufc_markets


def _parse_fighters_from_title(title: str) -> tuple[str, str] | None:
    """
    Extract fighter names from a market title like:
    "Jon Jones vs. Stipe Miocic: who wins?" → ("Jon Jones", "Stipe Miocic")
    """
    title = title.strip()
    # Match "Fighter A vs Fighter B" or "Fighter A vs. Fighter B"
    pattern = r"^(.+?)\s+vs\.?\s+(.+?)(?:\s*[:\?].*)?$"
    match = re.match(pattern, title, re.IGNORECASE)
    if match:
        f1 = match.group(1).strip().rstrip(",")
        f2 = match.group(2).strip().rstrip(",").split(":")[0].split("?")[0].strip()
        return (f1, f2)
    return None


def get_market_odds(ticker: str) -> Optional[dict]:
    """
    Get current odds for a specific market.
    Returns dict with yes_price, no_price, implied probabilities.
    """
    url = f"{BASE_URL}/markets/{ticker}"
    try:
        resp = requests.get(url, headers=_get_headers(), timeout=10)
        resp.raise_for_status()
        market = resp.json().get("market", {})

        yes_bid = _parse_price(market.get("yes_bid_dollars"))
        yes_ask = _parse_price(market.get("yes_ask_dollars"))
        no_bid = _parse_price(market.get("no_bid_dollars"))
        no_ask = _parse_price(market.get("no_ask_dollars"))
        last_price = _parse_price(market.get("last_price_dollars"))

        # Mid-price as best estimate of market probability
        yes_mid = None
        if yes_bid is not None and yes_ask is not None:
            yes_mid = (yes_bid + yes_ask) / 2
        elif last_price is not None:
            yes_mid = last_price

        return {
            "ticker": ticker,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "yes_mid": yes_mid,
            "implied_prob_yes": yes_mid,  # Kalshi prices ARE probabilities (0-1)
            "implied_prob_no": 1 - yes_mid if yes_mid is not None else None,
            "volume": market.get("volume_fp"),
            "status": market.get("status"),
        }
    except Exception as e:
        print(f"[kalshi] Error fetching odds for {ticker}: {e}")
        return None


def _parse_price(price_val) -> Optional[float]:
    """Parse Kalshi price (dollars string or float) to 0-1 probability."""
    if price_val is None:
        return None
    try:
        # Kalshi prices are in dollars (e.g. 0.65 means 65 cents = 65% implied prob)
        return float(price_val)
    except (ValueError, TypeError):
        return None


def match_fighters_to_market(
    fighter1: str, fighter2: str, ufc_markets: list[dict]
) -> Optional[dict]:
    """
    Find the Kalshi market for a given fighter matchup.
    Uses fuzzy name matching on market titles.
    Returns the matched market dict or None.
    """
    f1_lower = fighter1.lower()
    f2_lower = fighter2.lower()

    # Extract last names for matching (more reliable)
    f1_last = f1_lower.split()[-1] if f1_lower.split() else f1_lower
    f2_last = f2_lower.split()[-1] if f2_lower.split() else f2_lower

    best_match = None
    best_score = 0

    for market in ufc_markets:
        title = (market.get("title", "") or "").lower()
        subtitle = (market.get("subtitle", "") or "").lower()
        text = f"{title} {subtitle}"

        score = 0
        if f1_last in text:
            score += 2
        if f2_last in text:
            score += 2
        if f1_lower in text:
            score += 1
        if f2_lower in text:
            score += 1

        if score > best_score:
            best_score = score
            best_match = market

    # Require both fighters to be mentioned
    if best_score >= 4:
        return best_match
    return None
