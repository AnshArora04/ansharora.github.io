"""
Scrapes UFC fighter stats and upcoming fight cards from ufcstats.com
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import time

BASE_URL = "http://ufcstats.com"
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def scrape_fighter_stats() -> pd.DataFrame:
    """Scrape all fighter stats from ufcstats.com. Returns DataFrame."""
    cache_path = DATA_DIR / "fighters.csv"
    if cache_path.exists():
        print("[scraper] Loading cached fighter stats...")
        return pd.read_csv(cache_path)

    print("[scraper] Scraping fighter stats (this may take a minute)...")
    all_fighters = []

    # ufcstats paginates by first letter
    for char in "abcdefghijklmnopqrstuvwxyz":
        url = f"{BASE_URL}/statistics/fighters?char={char}&page=all"
        try:
            soup = get_soup(url)
            table = soup.find("table", class_="b-statistics__table")
            if not table:
                continue
            rows = table.find("tbody").find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 10:
                    continue
                fighter = {
                    "first_name": cells[0].get_text(strip=True),
                    "last_name": cells[1].get_text(strip=True),
                    "nickname": cells[2].get_text(strip=True),
                    "height": cells[3].get_text(strip=True),
                    "weight": cells[4].get_text(strip=True),
                    "reach": cells[5].get_text(strip=True),
                    "stance": cells[6].get_text(strip=True),
                    "wins": cells[7].get_text(strip=True),
                    "losses": cells[8].get_text(strip=True),
                    "draws": cells[9].get_text(strip=True),
                    "belt": cells[10].get_text(strip=True) if len(cells) > 10 else "",
                }
                fighter["name"] = f"{fighter['first_name']} {fighter['last_name']}".strip()
                all_fighters.append(fighter)
            time.sleep(0.3)
        except Exception as e:
            print(f"[scraper] Warning: failed on char={char}: {e}")
            continue

    df = pd.DataFrame(all_fighters)
    df = _clean_fighter_df(df)
    df.to_csv(cache_path, index=False)
    print(f"[scraper] Saved {len(df)} fighters to {cache_path}")
    return df


def _clean_fighter_df(df: pd.DataFrame) -> pd.DataFrame:
    def parse_height(h):
        """Convert '5' 11"' to inches."""
        try:
            parts = h.replace('"', "").split("'")
            return int(parts[0]) * 12 + int(parts[1].strip())
        except Exception:
            return None

    def parse_reach(r):
        try:
            return float(r.replace('"', "").strip())
        except Exception:
            return None

    def parse_record(v):
        try:
            return int(v)
        except Exception:
            return 0

    df["height_in"] = df["height"].apply(parse_height)
    df["reach_in"] = df["reach"].apply(parse_reach)
    df["wins"] = df["wins"].apply(parse_record)
    df["losses"] = df["losses"].apply(parse_record)
    df["draws"] = df["draws"].apply(parse_record)
    df["total_fights"] = df["wins"] + df["losses"] + df["draws"]
    df["win_pct"] = df.apply(
        lambda r: r["wins"] / r["total_fights"] if r["total_fights"] > 0 else 0.5, axis=1
    )
    return df


def scrape_upcoming_card() -> list[dict]:
    """
    Scrape the next upcoming UFC event and return a list of matchups.
    Each item: {"fighter1": str, "fighter2": str, "weight_class": str, "event": str}
    """
    print("[scraper] Fetching upcoming UFC event...")
    try:
        soup = get_soup(f"{BASE_URL}/statistics/events/upcoming")
        rows = soup.select("tr.b-statistics__table-row")
        if not rows:
            print("[scraper] No upcoming events found.")
            return []

        # First row is the next event
        first_event_link = None
        for row in rows:
            link = row.find("a")
            if link and link.get("href"):
                first_event_link = link["href"]
                event_name = link.get_text(strip=True)
                break

        if not first_event_link:
            print("[scraper] Could not find event link.")
            return []

        print(f"[scraper] Found event: {event_name}")
        event_soup = get_soup(first_event_link)

        fights = []
        fight_rows = event_soup.select("tr.b-fight-details__table-row")
        for row in fight_rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            fighters = cols[1].find_all("p")
            weight_cells = cols[6].find_all("p") if len(cols) > 6 else []

            if len(fighters) >= 2:
                f1 = fighters[0].get_text(strip=True)
                f2 = fighters[1].get_text(strip=True)
                weight = weight_cells[0].get_text(strip=True) if weight_cells else "Unknown"
                if f1 and f2:
                    fights.append({
                        "fighter1": f1,
                        "fighter2": f2,
                        "weight_class": weight,
                        "event": event_name,
                    })

        print(f"[scraper] Found {len(fights)} fights on the card.")
        return fights

    except Exception as e:
        print(f"[scraper] Error scraping event: {e}")
        return []


def scrape_fight_history() -> pd.DataFrame:
    """
    Scrape completed fight results for model training.
    Returns DataFrame with fight outcome data.
    """
    cache_path = DATA_DIR / "fight_history.csv"
    if cache_path.exists():
        print("[scraper] Loading cached fight history...")
        return pd.read_csv(cache_path)

    print("[scraper] Scraping fight history (this takes a while)...")
    all_fights = []

    soup = get_soup(f"{BASE_URL}/statistics/events/completed?page=all")
    event_links = [
        a["href"] for a in soup.select("tr.b-statistics__table-row a")
        if a.get("href", "").startswith("http://ufcstats.com/event-details/")
    ]

    print(f"[scraper] Found {len(event_links)} completed events.")

    for i, event_url in enumerate(event_links[:200]):  # cap at 200 events
        try:
            event_soup = get_soup(event_url)
            fight_rows = event_soup.select("tr.b-fight-details__table-row")

            for row in fight_rows:
                cols = row.find_all("td")
                if len(cols) < 8:
                    continue
                fighters = cols[1].find_all("p")
                if len(fighters) < 2:
                    continue

                win_indicator = cols[0].find_all("p")
                winner_idx = 0
                if win_indicator and len(win_indicator) >= 2:
                    w_text = win_indicator[0].get_text(strip=True).lower()
                    winner_idx = 0 if w_text == "w" else 1

                f1 = fighters[0].get_text(strip=True)
                f2 = fighters[1].get_text(strip=True)
                weight_cells = cols[6].find_all("p") if len(cols) > 6 else []
                weight = weight_cells[0].get_text(strip=True) if weight_cells else ""

                all_fights.append({
                    "fighter1": f1,
                    "fighter2": f2,
                    "winner": f1 if winner_idx == 0 else f2,
                    "weight_class": weight,
                })

            time.sleep(0.2)
            if (i + 1) % 20 == 0:
                print(f"[scraper] Processed {i + 1}/{min(200, len(event_links))} events...")

        except Exception as e:
            print(f"[scraper] Warning on event {event_url}: {e}")
            continue

    df = pd.DataFrame(all_fights)
    df.to_csv(cache_path, index=False)
    print(f"[scraper] Saved {len(df)} fight records to {cache_path}")
    return df
