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
    padding: 1rem 1.5rem;
    font-size: 0.9rem;
    color: #aaa;
    min-height: 2.5rem;
  }
  #status.error { color: #ff6b6b; }
  .spinner {
    display: inline-block;
    width: 14px; height: 14px;
    border: 2px solid #555;
    border-top-color: #cc0000;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin-right: 6px;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #results { padding: 0 0.5rem 2rem; }
  .event-group { margin-bottom: 2rem; }
  .event-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    padding: 0.4rem 1rem;
    background: #1a1a1a;
    border-left: 3px solid #cc0000;
    margin: 0.5rem 0;
  }
  .card {
    background: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    margin: 0.6rem 0.5rem;
    overflow: hidden;
  }
  .card-top {
    padding: 0.85rem 1rem 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.5rem;
  }
  .matchup { flex: 1; }
  .fighter { font-size: 1rem; font-weight: 600; color: #fff; }
  .vs { font-size: 0.7rem; color: #666; margin: 1px 0; }
  .weight { font-size: 0.72rem; color: #777; margin-top: 4px; }
  .badge {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    white-space: nowrap;
    align-self: flex-start;
  }
  .badge-bet { background: #0d3d1a; color: #4caf50; border: 1px solid #2e7d32; }
  .badge-pass { background: #3d3000; color: #ffc107; border: 1px solid #7d6000; }
  .badge-nomarket { background: #222; color: #777; border: 1px solid #333; }
  .card-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid #252525;
  }
  .stat {
    padding: 0.55rem 0.5rem;
    text-align: center;
    border-right: 1px solid #252525;
  }
  .stat:last-child { border-right: none; }
  .stat-label { font-size: 0.65rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat-value { font-size: 0.95rem; font-weight: 600; margin-top: 2px; }
  .winner-row {
    padding: 0.5rem 1rem;
    border-top: 1px solid #252525;
    font-size: 0.82rem;
    color: #aaa;
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }
  .winner-name { color: #fff; font-weight: 600; }
  .edge-pos { color: #4caf50; }
  .edge-neg { color: #f44336; }
  .summary {
    margin: 1rem;
    padding: 0.85rem 1rem;
    background: #0d1f0d;
    border: 1px solid #1e4d1e;
    border-radius: 8px;
    font-size: 0.85rem;
    color: #81c784;
  }
  .summary h3 { font-size: 0.9rem; margin-bottom: 0.4rem; color: #a5d6a7; }
  .summary ul { padding-left: 1.1rem; }
  .summary li { margin: 0.2rem 0; }
  .note {
    margin: 0 1rem;
    font-size: 0.72rem;
    color: #555;
    padding-bottom: 1rem;
  }
  @media (min-width: 600px) {
    #results { padding: 0 1rem 2rem; }
    .card { margin: 0.6rem 0; }
  }
</style>
</head>
<body>
<header>
  <h1>UFC Betting Predictor <span>ML predictions + Kalshi market odds · v11:35</span></h1>
  <button id="refresh-btn" onclick="loadData()">Refresh</button>
</header>
<div id="status"></div>
<div id="results"></div>

<script>
function fmt(v, asPercent=true) {
  if (v === null || v === undefined) return '—';
  if (asPercent) return (v * 100).toFixed(0) + '%';
  return v;
}
function edge(v) {
  if (v === null || v === undefined) return '<span>—</span>';
  const p = (v * 100).toFixed(0);
  const cls = v >= 0 ? 'edge-pos' : 'edge-neg';
  const sign = v >= 0 ? '+' : '';
  return `<span class="${cls}">${sign}${p}%</span>`;
}
function badge(rec) {
  if (rec && rec.startsWith('BET')) return `<span class="badge badge-bet">${rec}</span>`;
  if (rec === 'PASS') return `<span class="badge badge-pass">PASS</span>`;
  return `<span class="badge badge-nomarket">NO MARKET</span>`;
}

function renderResults(data) {
  const container = document.getElementById('results');
  if (!data.results || data.results.length === 0) {
    container.innerHTML = '<p style="padding:1rem;color:#888">No fights found.</p>';
    return;
  }

  // Group by event
  const events = {};
  for (const r of data.results) {
    const ev = r.event || 'Upcoming';
    if (!events[ev]) events[ev] = [];
    events[ev].push(r);
  }

  const valueBets = data.results.filter(r => r.bet_recommendation && r.bet_recommendation.startsWith('BET'));

  let html = '';

  // Summary box
  if (valueBets.length > 0) {
    html += '<div class="summary"><h3>Value Bets Found (' + valueBets.length + ')</h3><ul>';
    for (const vb of valueBets) {
      const mProb = vb.bet_on === vb.fighter1 ? vb.model_f1_prob : vb.model_f2_prob;
      const kProb = vb.bet_on === vb.fighter1 ? vb.kalshi_f1_prob : vb.kalshi_f2_prob;
      html += `<li><strong>${vb.bet_on}</strong> — model ${fmt(mProb)}, market ${fmt(kProb)}, edge +${(vb.bet_edge*100).toFixed(1)}%</li>`;
    }
    html += '</ul></div>';
  }

  for (const [evName, fights] of Object.entries(events)) {
    html += `<div class="event-group"><div class="event-title">${evName}</div>`;
    for (const r of fights) {
      const f1Win = r.predicted_winner === r.fighter1;
      const modelPct = fmt(f1Win ? r.model_f1_prob : r.model_f2_prob);
      const kalshiPct = fmt(f1Win ? r.kalshi_f1_prob : r.kalshi_f2_prob);
      const edgeVal = f1Win ? r.f1_edge : r.f2_edge;

      html += `<div class="card">
        <div class="card-top">
          <div class="matchup">
            <div class="fighter">${r.fighter1}</div>
            <div class="vs">vs.</div>
            <div class="fighter">${r.fighter2}</div>
            <div class="weight">${r.weight_class || ''}</div>
          </div>
          ${badge(r.bet_recommendation)}
        </div>
        <div class="winner-row">
          Pick: <span class="winner-name">${r.predicted_winner}</span>
          &nbsp;·&nbsp; conf ${fmt(r.confidence)} &nbsp;·&nbsp; edge ${edge(edgeVal)}
        </div>
        <div class="card-stats">
          <div class="stat"><div class="stat-label">Model</div><div class="stat-value">${modelPct}</div></div>
          <div class="stat"><div class="stat-label">Kalshi</div><div class="stat-value">${kalshiPct}</div></div>
          <div class="stat"><div class="stat-label">F1 Prob</div><div class="stat-value">${fmt(r.model_f1_prob)}</div></div>
          <div class="stat"><div class="stat-label">F2 Prob</div><div class="stat-value">${fmt(r.model_f2_prob)}</div></div>
        </div>
      </div>`;
    }
    html += '</div>';
  }

  html += '<p class="note">Model accuracy ~63%. For informational purposes only. Bet responsibly.</p>';
  container.innerHTML = html;
}

async function loadData() {
  const btn = document.getElementById('refresh-btn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.className = '';
  status.innerHTML = '<span class="spinner"></span> Fetching fights & running model… (may take ~30s)';

  try {
    const resp = await fetch('/api/analyze');
    const data = await resp.json();
    if (data.error) {
      status.className = 'error';
      status.textContent = 'Error: ' + data.error;
    } else {
      status.textContent = `Loaded ${data.results.length} fights · ${new Date().toLocaleTimeString()}`;
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
