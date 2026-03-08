"""
Kalshi API client — fetches UFC prediction markets and odds.

Auth: RSA-SHA256 signed requests
  Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE
  Signature: base64( RSA_SHA256_PKCS1v15( timestamp_ms_str + METHOD + /path ) )

Docs: https://docs.kalshi.com/api-reference/market/get-markets
Base URL: https://api.elections.kalshi.com/trade-api/v2
"""

import os
import re
import base64
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

load_dotenv()

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
UFC_KEYWORDS = ["ufc", "mma", "fight", "fighter", "bout"]

_private_key_cache = None


def _load_private_key():
    global _private_key_cache
    if _private_key_cache is not None:
        return _private_key_cache

    pem_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private.pem")
    pem_path = Path(pem_path)

    if not pem_path.exists():
        raise FileNotFoundError(
            f"Kalshi private key not found at {pem_path}. "
            "Set KALSHI_PRIVATE_KEY_PATH in .env or place kalshi_private.pem in the project root."
        )

    with open(pem_path, "rb") as f:
        _private_key_cache = serialization.load_pem_private_key(f.read(), password=None)
    return _private_key_cache


def _sign_request(method: str, path: str) -> dict:
    """
    Build Kalshi auth headers for a request.

    Signature message: str(timestamp_ms) + METHOD.upper() + /path
    Algorithm: RSA-SHA256, PKCS1v15 padding, base64-encoded output
    """
    api_key_id = os.getenv("KALSHI_API_KEY_ID", "")
    if not api_key_id:
        raise ValueError("KALSHI_API_KEY_ID not set in .env")

    timestamp_ms = str(int(time.time() * 1000))
    message = (timestamp_ms + method.upper() + path).encode("utf-8")

    private_key = _load_private_key()
    signature_bytes = private_key.sign(message, padding.PKCS1v15(), hashes.SHA256())
    signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")

    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": signature_b64,
    }


def _api_path(url: str) -> str:
    """Extract just the /trade-api/v2/... path from a full URL."""
    parsed = urlparse(url)
    return parsed.path


def _get(endpoint: str, params: dict | None = None) -> dict:
    """Authenticated GET request to Kalshi API."""
    url = f"{BASE_URL}{endpoint}"
    headers = _sign_request("GET", _api_path(url))
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_all_markets(status: str = "open", limit: int = 1000) -> list[dict]:
    """
    Fetch all open markets from Kalshi (paginated).
    Returns flat list of market dicts.
    """
    all_markets = []
    cursor = None

    while True:
        params: dict = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor

        try:
            data = _get("/markets", params=params)
        except requests.HTTPError as e:
            print(f"[kalshi] HTTP error fetching markets: {e.response.status_code} {e.response.text[:200]}")
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
    Fetch all open Kalshi markets and filter for UFC/MMA fights.
    Returns list of market dicts with added 'parsed_fighters' field.
    """
    print("[kalshi] Fetching all open markets (authenticated)...")
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


def _parse_fighters_from_title(title: str) -> Optional[tuple[str, str]]:
    """
    Extract fighter names from a market title.
    e.g. "Jon Jones vs. Stipe Miocic: who wins?" → ("Jon Jones", "Stipe Miocic")
    """
    title = title.strip()
    pattern = r"^(.+?)\s+vs\.?\s+(.+?)(?:\s*[:\?].*)?$"
    match = re.match(pattern, title, re.IGNORECASE)
    if match:
        f1 = match.group(1).strip().rstrip(",")
        f2 = match.group(2).strip().split(":")[0].split("?")[0].strip().rstrip(",")
        return (f1, f2)
    return None


def get_market(ticker: str) -> Optional[dict]:
    """Fetch a single market by ticker."""
    try:
        data = _get(f"/markets/{ticker}")
        return data.get("market", {})
    except Exception as e:
        print(f"[kalshi] Error fetching market {ticker}: {e}")
        return None


def get_market_odds(ticker: str) -> Optional[dict]:
    """
    Get current odds for a specific market ticker.
    Returns dict with yes/no prices and implied probabilities.
    Kalshi prices are in dollars (0.00–1.00) and ARE the implied probability.
    """
    market = get_market(ticker)
    if not market:
        return None

    yes_bid = _parse_price(market.get("yes_bid_dollars"))
    yes_ask = _parse_price(market.get("yes_ask_dollars"))
    no_bid = _parse_price(market.get("no_bid_dollars"))
    no_ask = _parse_price(market.get("no_ask_dollars"))
    last_price = _parse_price(market.get("last_price_dollars"))

    # Use mid-price as best estimate; fall back to last traded price
    yes_mid = None
    if yes_bid is not None and yes_ask is not None:
        yes_mid = (yes_bid + yes_ask) / 2.0
    elif last_price is not None:
        yes_mid = last_price

    return {
        "ticker": ticker,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "yes_mid": yes_mid,
        "implied_prob_yes": yes_mid,
        "implied_prob_no": (1.0 - yes_mid) if yes_mid is not None else None,
        "volume": market.get("volume_fp"),
        "status": market.get("status"),
        "title": market.get("title"),
    }


def _parse_price(price_val) -> Optional[float]:
    if price_val is None:
        return None
    try:
        return float(price_val)
    except (ValueError, TypeError):
        return None


def match_fighters_to_market(
    fighter1: str, fighter2: str, ufc_markets: list[dict]
) -> Optional[dict]:
    """
    Find the best-matching Kalshi market for a fighter matchup.
    Scores markets by how many name components appear in the title.
    Requires both fighters' last names to match (score >= 4).
    """
    f1_lower = fighter1.lower()
    f2_lower = fighter2.lower()
    f1_last = f1_lower.split()[-1]
    f2_last = f2_lower.split()[-1]

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

    if best_score >= 4:
        return best_match
    return None
