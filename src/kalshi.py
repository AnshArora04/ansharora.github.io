"""
Kalshi API client — fetches UFC prediction markets and odds.

Auth: RSA-PSS signed requests
  Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE
  Signature: base64( RSA_PSS( SHA256, MGF1(SHA256), salt=DIGEST_LENGTH )( timestamp_ms_str + METHOD + /path ) )

Docs: https://docs.kalshi.com/getting_started/api_keys
Base URL: https://trading-api.kalshi.com/trade-api/v2
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

BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
UFC_KEYWORDS = ["ufc", "mma", "fight", "fighter", "bout"]

# Kalshi series ticker for UFC fight-winner markets ("Will X win...")
_UFC_FIGHT_SERIES = "KXUFCFIGHT"

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
    signature_bytes = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
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
    Fetch UFC/MMA fight markets from Kalshi.

    Strategy (fastest first):
    1. Query /events via known UFC series tickers — returns only relevant markets
    2. Fall back to full market scan, excluding cross-category parlay markets
    """
    markets = _get_markets_via_ufc_series()
    if markets:
        return markets
    print("[kalshi] No UFC events found via series; falling back to full scan...")
    return _scan_all_markets_for_ufc()


def _get_markets_via_ufc_series() -> list[dict]:
    """
    Fetch open events under KXUFCFIGHT series, then collect all their markets.
    Each market title is "Will {Fighter} win the {F1} vs {F2} fight...".
    Adds 'parsed_winner_fighter' with the fighter this market's YES resolves for.
    """
    try:
        data = _get("/events", params={"series_ticker": _UFC_FIGHT_SERIES, "status": "open", "limit": 100})
    except Exception as e:
        print(f"[kalshi] Could not reach events endpoint: {e}")
        return []

    events = data.get("events", [])
    if not events:
        return []

    print(f"[kalshi] Found {len(events)} open UFC events (series {_UFC_FIGHT_SERIES})")
    ufc_markets = []
    for event in events:
        event_ticker = event.get("event_ticker") or event.get("ticker", "")
        if not event_ticker:
            continue
        for m in _get_markets_for_event(event_ticker):
            m["parsed_winner_fighter"] = _parse_winner_from_title(m.get("title", ""))
            m["parsed_fighters"] = _parse_fighters_from_title(
                m.get("title", "") or m.get("subtitle", "")
            )
            ufc_markets.append(m)

    print(f"[kalshi] Found {len(ufc_markets)} UFC fight winner markets.")
    return ufc_markets


def _get_markets_for_event(event_ticker: str) -> list[dict]:
    try:
        data = _get("/markets", params={"event_ticker": event_ticker, "status": "open", "limit": 200})
        return data.get("markets", [])
    except Exception:
        return []


def _scan_all_markets_for_ufc() -> list[dict]:
    """Full scan of all open markets, filtering for UFC while excluding cross-category parlays."""
    print("[kalshi] Fetching all open markets (authenticated)...")
    all_markets = get_all_markets()
    print(f"[kalshi] Total open markets: {len(all_markets)}")

    ufc_markets = []
    for market in all_markets:
        ticker = market.get("ticker", "") or ""

        # Cross-category parlay markets (e.g. KXMVECROSSCATEGORY-...) contain UFC fighter
        # names but are not standalone fight winner markets — skip them
        if "KXMVECROSSCATEGORY" in ticker:
            continue

        event_ticker = (market.get("event_ticker", "") or "").lower()
        title = (market.get("title", "") or "").lower()
        subtitle = (market.get("subtitle", "") or "").lower()

        text = f"{event_ticker} {title} {subtitle}"
        if any(kw in text for kw in UFC_KEYWORDS):
            market["parsed_winner_fighter"] = _parse_winner_from_title(market.get("title", ""))
            market["parsed_fighters"] = _parse_fighters_from_title(
                market.get("title", "") or market.get("subtitle", "")
            )
            ufc_markets.append(market)

    print(f"[kalshi] Found {len(ufc_markets)} UFC/MMA markets.")
    return ufc_markets


def _parse_winner_from_title(title: str) -> Optional[str]:
    """
    Extract the fighter from a 'Will X win...' market title.
    e.g. "Will Max Holloway win the Holloway vs Oliveira fight..." → "Max Holloway"
    """
    if not title:
        return None
    match = re.match(r"Will (.+?) win\b", title, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


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

    # Use mid-price as best estimate; fall back to last traded price.
    # Only use bid/ask mid when at least one side is non-zero (active market).
    yes_mid = None
    if yes_bid is not None and yes_ask is not None and (yes_bid > 0 or yes_ask > 0):
        yes_mid = (yes_bid + yes_ask) / 2.0
    if yes_mid is None and last_price is not None and last_price > 0:
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


def match_fight_markets(
    fighter1: str, fighter2: str, ufc_markets: list[dict]
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Find the Kalshi fight-winner markets for both fighters in a matchup.

    With KXUFCFIGHT series each fight has two markets:
      "Will Fighter1 win..." and "Will Fighter2 win..."

    Returns (market_for_fighter1, market_for_fighter2).
    Each market's yes_mid is that fighter's implied win probability.
    """
    return (
        _find_fighter_win_market(fighter1, fighter2, ufc_markets),
        _find_fighter_win_market(fighter2, fighter1, ufc_markets),
    )


def _find_fighter_win_market(
    target: str, opponent: str, ufc_markets: list[dict]
) -> Optional[dict]:
    """
    Find the market where target fighter is the YES outcome.
    Prefers markets where parsed_winner_fighter matches target AND
    the opponent's name also appears (to confirm it's the right fight).
    """
    t_lower = target.lower()
    t_last = t_lower.split()[-1]
    o_last = opponent.lower().split()[-1]

    best = None
    best_score = 0

    for m in ufc_markets:
        winner = (m.get("parsed_winner_fighter") or "").lower()
        title_lower = (m.get("title", "") or "").lower()

        # Target fighter must appear in the winner field
        if t_last not in winner and t_lower not in winner:
            continue

        score = 1  # target is in winner field
        if t_lower in winner:
            score += 1  # full name match is stronger
        if o_last in title_lower:
            score += 2  # opponent is in the title → right fight

        if score > best_score:
            best_score = score
            best = m

    return best if best_score >= 1 else None


# Keep for backwards compatibility
def match_fighters_to_market(
    fighter1: str, fighter2: str, ufc_markets: list[dict]
) -> Optional[dict]:
    """Deprecated: use match_fight_markets instead."""
    f1_market, _ = match_fight_markets(fighter1, fighter2, ufc_markets)
    return f1_market
