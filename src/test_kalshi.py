"""
Kalshi integration diagnostic script.
Run: python -m src.test_kalshi

Checks each layer independently and prints [PASS] or [FAIL: reason].
"""

import os
import sys
import base64
import time
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OK = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
INFO = "\033[33m[INFO]\033[0m"


def step(n: int, desc: str):
    print(f"\n{'='*60}")
    print(f"Step {n}: {desc}")
    print("=" * 60)


# ── Step 1: .env ─────────────────────────────────────────────────
step(1, "Load .env credentials")

api_key_id = os.getenv("KALSHI_API_KEY_ID", "")
pem_path_str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private.pem")
pem_path = Path(pem_path_str)

ok = True
if api_key_id:
    print(f"{OK} KALSHI_API_KEY_ID = {api_key_id[:8]}...{api_key_id[-4:]}")
else:
    print(f"{FAIL} KALSHI_API_KEY_ID not set in .env")
    ok = False

if pem_path.exists():
    print(f"{OK} Private key file found: {pem_path}")
else:
    print(f"{FAIL} Private key file NOT found at: {pem_path}")
    ok = False

if not ok:
    print("\nFix .env and kalshi_private.pem before proceeding.")
    sys.exit(1)

# ── Step 2: Load PEM key ──────────────────────────────────────────
step(2, "Parse RSA private key")

try:
    from cryptography.hazmat.primitives import serialization
    with open(pem_path, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    key_size = private_key.key_size
    print(f"{OK} RSA private key loaded ({key_size}-bit)")
except Exception as e:
    print(f"{FAIL} Could not load private key: {e}")
    sys.exit(1)

# ── Step 3: Sign a test request ───────────────────────────────────
step(3, "Generate RSA-SHA256 signature")

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

    test_path = "/trade-api/v2/markets"
    timestamp_ms = str(int(time.time() * 1000))
    message = (timestamp_ms + "GET" + test_path).encode("utf-8")

    sig_bytes = private_key.sign(message, asym_padding.PKCS1v15(), hashes.SHA256())
    sig_b64 = base64.b64encode(sig_bytes).decode("utf-8")

    print(f"{OK} Signature generated (length: {len(sig_b64)} chars)")
    print(f"    Timestamp  : {timestamp_ms}")
    print(f"    Message    : {timestamp_ms}GET{test_path}")
    print(f"    Sig prefix : {sig_b64[:40]}...")
except Exception as e:
    print(f"{FAIL} Signing failed: {e}")
    sys.exit(1)

# ── Step 4: Live API call ─────────────────────────────────────────
step(4, "Call GET /markets (limit=5)")

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

def make_headers(method: str, path: str) -> dict:
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path).encode("utf-8")
    sig = private_key.sign(msg, asym_padding.PKCS1v15(), hashes.SHA256())
    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
    }

endpoint = "/markets"
url = f"{BASE_URL}{endpoint}"
api_path = "/trade-api/v2/markets"

try:
    headers = make_headers("GET", api_path)
    resp = requests.get(url, headers=headers, params={"status": "open", "limit": 5}, timeout=15)
    print(f"    HTTP status: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        markets = data.get("markets", [])
        print(f"{OK} API responded with {len(markets)} markets (page of 5)")
        print(f"    Cursor present: {bool(data.get('cursor'))}")
        if markets:
            print(f"    Sample market title: {markets[0].get('title', '(no title)')!r}")
            print(f"    Sample market ticker: {markets[0].get('ticker', '(none)')!r}")
    else:
        print(f"{FAIL} Non-200 response")
        try:
            body = resp.json()
            print(f"    Response body: {json.dumps(body, indent=2)[:500]}")
        except Exception:
            print(f"    Raw response: {resp.text[:500]}")
        sys.exit(1)
except Exception as e:
    print(f"{FAIL} Request failed: {e}")
    sys.exit(1)

# ── Step 5: Full market fetch + UFC filter ────────────────────────
step(5, "Fetch all open markets and filter for UFC/MMA")

UFC_KEYWORDS = ["ufc", "mma", "fight", "fighter", "bout", "mixed martial"]
all_markets = []
cursor = None
pages = 0

print("    Fetching pages...")
while True:
    params = {"status": "open", "limit": 1000}
    if cursor:
        params["cursor"] = cursor

    try:
        h = make_headers("GET", "/trade-api/v2/markets")
        r = requests.get(url, headers=h, params=params, timeout=15)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        print(f"{FAIL} Paginated fetch failed on page {pages + 1}: {e}")
        break

    page_markets = d.get("markets", [])
    all_markets.extend(page_markets)
    pages += 1
    cursor = d.get("cursor")
    if not cursor or not page_markets:
        break

print(f"    Total open markets fetched: {len(all_markets)} (across {pages} page(s))")

ufc_markets = []
for m in all_markets:
    text = " ".join([
        (m.get("title") or ""),
        (m.get("subtitle") or ""),
        (m.get("ticker") or ""),
        (m.get("event_ticker") or ""),
    ]).lower()
    if any(kw in text for kw in UFC_KEYWORDS):
        ufc_markets.append(m)

if ufc_markets:
    print(f"{OK} Found {len(ufc_markets)} UFC/MMA market(s):")
    for m in ufc_markets[:10]:
        print(f"    • [{m.get('ticker')}] {m.get('title')}")
        print(f"      event_ticker={m.get('event_ticker')!r}")
    if len(ufc_markets) > 10:
        print(f"    ... and {len(ufc_markets) - 10} more")
else:
    print(f"{INFO} No UFC/MMA markets found right now (may be between events).")
    print("    Showing 5 non-cross-category market titles with their event_tickers:")
    shown = 0
    for m in all_markets:
        if "KXMVECROSSCATEGORY" in (m.get("ticker") or ""):
            continue
        print(f"    • event_ticker={m.get('event_ticker')!r}  title={m.get('title')!r}")
        shown += 1
        if shown >= 5:
            break

# Also try the /events endpoint with UFC series
print(f"\n    Checking /events endpoint for UFC series tickers:")
for series in ["UFCMMA", "UFC", "MMA", "UFCFIGHT"]:
    try:
        h = make_headers("GET", f"/trade-api/v2/events")
        er = requests.get(f"{BASE_URL}/events", headers=h, params={"series_ticker": series, "status": "open", "limit": 5}, timeout=15)
        if er.status_code == 200:
            evts = er.json().get("events", [])
            if evts:
                print(f"    {OK} series={series!r}: {len(evts)} events found")
                for ev in evts[:3]:
                    print(f"        event_ticker={ev.get('event_ticker') or ev.get('ticker')!r}  title={ev.get('title')!r}")
            else:
                print(f"    {INFO} series={series!r}: 0 events")
        else:
            print(f"    {INFO} series={series!r}: HTTP {er.status_code}")
    except Exception as e:
        print(f"    {INFO} series={series!r}: error {e}")

# ── Step 6: Odds for first UFC market ────────────────────────────
step(6, "Fetch odds for first UFC market")

if not ufc_markets:
    print(f"{INFO} Skipping — no UFC markets to inspect")
else:
    sample = ufc_markets[0]
    ticker = sample.get("ticker", "")
    print(f"    Market: {sample.get('title')!r}")
    print(f"    Ticker: {ticker}")

    try:
        mkt_path = f"/trade-api/v2/markets/{ticker}"
        h = make_headers("GET", mkt_path)
        mr = requests.get(f"{BASE_URL}/markets/{ticker}", headers=h, timeout=15)
        mr.raise_for_status()
        mkt = mr.json().get("market", {})

        print(f"    Raw keys in market object: {list(mkt.keys())}")

        # Print all price-related fields raw
        price_keys = [k for k in mkt if "price" in k.lower() or "bid" in k.lower()
                      or "ask" in k.lower() or "dollar" in k.lower() or "yes" in k.lower()
                      or "no" in k.lower()]
        print(f"\n    Price-related fields:")
        for k in price_keys:
            print(f"      {k}: {mkt[k]!r}")

        # Attempt to derive yes mid-price
        yes_bid = mkt.get("yes_bid") or mkt.get("yes_bid_dollars")
        yes_ask = mkt.get("yes_ask") or mkt.get("yes_ask_dollars")
        last = mkt.get("last_price") or mkt.get("last_price_dollars")

        def to_float(v):
            if v is None:
                return None
            try:
                f = float(v)
                # If value > 1, assume it's in cents (0–100) → convert to 0–1
                return f / 100.0 if f > 1.0 else f
            except (ValueError, TypeError):
                return None

        yb = to_float(yes_bid)
        ya = to_float(yes_ask)
        lp = to_float(last)
        mid = (yb + ya) / 2.0 if yb is not None and ya is not None else lp

        print(f"\n    yes_bid (normalized): {yb}")
        print(f"    yes_ask (normalized): {ya}")
        print(f"    last_price (normalized): {lp}")
        print(f"    yes_mid: {mid}")

        if mid is not None and 0.0 <= mid <= 1.0:
            print(f"{OK} Implied prob YES: {mid:.1%}  |  NO: {1 - mid:.1%}")
        else:
            print(f"{INFO} Could not compute valid mid-price (mid={mid})")

    except Exception as e:
        print(f"{FAIL} Odds fetch failed: {e}")

# ── Done ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Diagnostic complete.")
print("=" * 60)
