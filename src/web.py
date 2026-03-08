"""
UFC Betting Predictor — Flask Web UI

Usage:
    python -m src.web          # starts on http://0.0.0.0:5000
    python -m src.web --port 8080
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, render_template_string

from src.scraper import scrape_fighter_stats, scrape_upcoming_card, scrape_fight_history
from src.model import train, load_model
from src import predictor as pred_module
from src import kalshi as kalshi_client

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UFC Betting Predictor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0d0d0d;
    color: #e0e0e0;
    min-height: 100vh;
  }
  header {
    background: linear-gradient(135deg, #1a0000, #3d0000);
    border-bottom: 2px solid #cc0000;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
  }
  h1 { font-size: 1.4rem; color: #ff4444; letter-spacing: 0.04em; }
  h1 span { color: #999; font-weight: 400; font-size: 0.85rem; display: block; margin-top: 2px; }
  #refresh-btn {
    background: #cc0000;
    color: #fff;
    border: none;
    padding: 0.5rem 1.1rem;
    border-radius: 6px;
    font-size: 0.95rem;
    cursor: pointer;
    font-weight: 600;
  }
  #refresh-btn:disabled { background: #555; cursor: not-allowed; }
  #status {
    padding: 0.75rem 1.5rem;
    font-size: 0.85rem;
    color: #666;
    min-height: 2.2rem;
  }
  #status.error { color: #ff6b6b; }
  .spinner {
    display: inline-block;
    width: 13px; height: 13px;
    border: 2px solid #333;
    border-top-color: #cc0000;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin-right: 6px;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #results { padding: 0 0.75rem 2rem; }

  /* Fight card */
  .card {
    background: #161616;
    border: 1px solid #242424;
    border-radius: 12px;
    margin: 0.65rem 0;
    overflow: hidden;
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 1rem;
    background: #1c1c1c;
    border-bottom: 1px solid #242424;
  }
  .event-name {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #666;
  }
  .volume-tag {
    font-size: 0.7rem;
    color: #555;
  }
  .volume-tag span { color: #888; font-weight: 600; }

  /* Odds section — two fighters side by side */
  .odds-row {
    display: flex;
    align-items: stretch;
  }
  .fighter-col {
    flex: 1;
    padding: 0.85rem 1rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .fighter-col.right {
    text-align: right;
    border-left: 1px solid #1e1e1e;
  }
  .fighter-name {
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    line-height: 1.2;
  }
  .fighter-name.fav { color: #fff; }
  .odds-pct {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
  }
  .odds-pct.high { color: #ff4444; }
  .odds-pct.low  { color: #777; }
  .odds-label {
    font-size: 0.65rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* Odds bar */
  .bar-wrap {
    height: 4px;
    background: #222;
    display: flex;
  }
  .bar-f1 { background: #cc0000; transition: width 0.4s; }
  .bar-f2 { background: #444; flex: 1; }

  /* Bottom meta row */
  .meta-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 1rem;
    border-top: 1px solid #1e1e1e;
    font-size: 0.72rem;
    color: #555;
  }
  .pick { color: #aaa; }
  .pick strong { color: #ddd; }
  .weight-class { color: #444; }
  .no-market { color: #444; font-style: italic; }

  .note {
    font-size: 0.7rem;
    color: #3a3a3a;
    text-align: center;
    padding: 1rem 0 0.5rem;
  }

  @media (min-width: 600px) {
    #results { padding: 0 1.25rem 2rem; }
  }
</style>
</head>
<body>
<header>
  <h1>UFC Fight Odds <span>Kalshi market odds · v11:35</span></h1>
  <button id="refresh-btn" onclick="loadData()">Refresh</button>
</header>
<div id="status"></div>
<div id="results"></div>

<script>
function pct(v) {
  if (v === null || v === undefined) return null;
  return Math.round(v * 100);
}
function fmtVol(v) {
  if (v === null || v === undefined) return null;
  if (v >= 1000) return '$' + (v / 1000).toFixed(1) + 'k';
  return '$' + Math.round(v);
}

function renderResults(data) {
  const container = document.getElementById('results');
  if (!data.results || data.results.length === 0) {
    container.innerHTML = '<p style="padding:1.5rem;color:#555;text-align:center">No fights found.</p>';
    return;
  }

  // Sort by Kalshi volume descending (most traded first), no-market fights at bottom
  const sorted = [...data.results].sort((a, b) => {
    const va = a.kalshi_volume ?? -1;
    const vb = b.kalshi_volume ?? -1;
    return vb - va;
  });

  let html = '';

  for (const r of sorted) {
    const f1k = pct(r.kalshi_f1_prob);
    const f2k = pct(r.kalshi_f2_prob);
    const hasOdds = f1k !== null && f2k !== null;

    // Determine favourite by Kalshi odds
    const f1Fav = hasOdds && f1k >= f2k;
    const f1PctClass = hasOdds ? (f1Fav ? 'high' : 'low') : 'low';
    const f2PctClass = hasOdds ? (!f1Fav ? 'high' : 'low') : 'low';

    const vol = fmtVol(r.kalshi_volume);
    const barWidth = hasOdds ? f1k : 50;

    const pick = r.predicted_winner && r.predicted_winner !== 'Unknown'
      ? `<span class="pick">ML pick: <strong>${r.predicted_winner}</strong></span>`
      : '';

    html += `<div class="card">
      <div class="card-header">
        <span class="event-name">${r.event || 'Upcoming'}</span>
        ${vol ? `<span class="volume-tag">Kalshi volume <span>${vol}</span></span>` : '<span class="volume-tag no-market">No market</span>'}
      </div>
      <div class="odds-row">
        <div class="fighter-col left">
          <div class="fighter-name">${r.fighter1}</div>
          <div class="odds-pct ${f1PctClass}">${hasOdds ? f1k + '%' : '—'}</div>
          <div class="odds-label">win prob</div>
        </div>
        <div class="fighter-col right">
          <div class="fighter-name">${r.fighter2}</div>
          <div class="odds-pct ${f2PctClass}">${hasOdds ? f2k + '%' : '—'}</div>
          <div class="odds-label">win prob</div>
        </div>
      </div>
      <div class="bar-wrap">
        <div class="bar-f1" style="width:${barWidth}%"></div>
        <div class="bar-f2"></div>
      </div>
      <div class="meta-row">
        ${pick}
        <span class="weight-class">${r.weight_class || ''}</span>
      </div>
    </div>`;
  }

  html += '<p class="note">Odds from Kalshi prediction markets. For informational purposes only.</p>';
  container.innerHTML = html;
}

async function loadData() {
  const btn = document.getElementById('refresh-btn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.className = '';
  status.innerHTML = '<span class="spinner"></span> Loading…';

  try {
    const resp = await fetch('/api/analyze');
    const data = await resp.json();
    if (data.error) {
      status.className = 'error';
      status.textContent = 'Error: ' + data.error;
    } else {
      const with_odds = data.results.filter(r => r.kalshi_f1_prob !== null).length;
      status.textContent = `${data.results.length} fights · ${with_odds} with live odds · ${new Date().toLocaleTimeString()}`;
      renderResults(data);
    }
  } catch (e) {
    status.className = 'error';
    status.textContent = 'Failed to fetch: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

loadData();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/analyze")
def analyze():
    try:
        fighters_df = scrape_fighter_stats()
        model = load_model()
        if model is None:
            fights_df = scrape_fight_history()
            if fights_df.empty:
                return jsonify({"error": "No fight history for training"}), 500
            model = train(fights_df, fighters_df)

        fights = scrape_upcoming_card()
        if not fights:
            return jsonify({"results": [], "message": "No upcoming fights found"})

        try:
            ufc_markets = kalshi_client.get_ufc_markets()
        except Exception as kalshi_err:
            print(f"[web] Kalshi unavailable: {kalshi_err}")
            ufc_markets = []
        results = pred_module.analyze_card(fights, fighters_df, model, ufc_markets)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="UFC Predictor Web UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"\n  UFC Betting Predictor — Web UI")
    print(f"  Open on this device:  http://localhost:{args.port}")
    print(f"  Open on your phone:   http://<your-local-ip>:{args.port}")
    print(f"  (Find your IP with: hostname -I)\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
