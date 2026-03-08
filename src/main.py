"""
UFC Betting Predictor — CLI entrypoint

Usage:
    python -m src.main                  # Analyze upcoming card
    python -m src.main --train          # Retrain model from scratch
    python -m src.main --no-kalshi      # Skip Kalshi (model predictions only)
    python -m src.main --event "UFC 312" # Filter by event name (partial match)
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper import scrape_fighter_stats, scrape_upcoming_card, scrape_fight_history
from src.model import train, load_model
from src import predictor as pred_module
from src import kalshi as kalshi_client

console = Console()


def print_banner():
    console.print(Panel(
        "[bold red]UFC Betting Predictor[/bold red]\n"
        "[dim]ML predictions + Kalshi market odds → value bets[/dim]",
        border_style="red",
        expand=False,
    ))


def print_results(results: list[dict]):
    if not results:
        console.print("[yellow]No fights to display.[/yellow]")
        return

    table = Table(
        title="Fight Card Analysis",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )

    table.add_column("Matchup", style="white", min_width=30)
    table.add_column("Weight", style="dim", min_width=12)
    table.add_column("Model Pick", style="bold", min_width=20)
    table.add_column("Conf.", justify="right", min_width=6)
    table.add_column("Model %", justify="right", min_width=9)
    table.add_column("Kalshi %", justify="right", min_width=9)
    table.add_column("Edge", justify="right", min_width=7)
    table.add_column("Recommendation", style="bold", min_width=22)

    for r in results:
        f1 = r["fighter1"]
        f2 = r["fighter2"]
        matchup = f"{f1}\nvs.\n{f2}"

        winner = r["predicted_winner"]
        conf = f"{r['confidence']:.0%}"

        # Model probability for predicted winner
        if r["predicted_winner"] == f1:
            model_pct = f"{r['model_f1_prob']:.0%}"
            kalshi_pct = f"{r['kalshi_f1_prob']:.0%}" if r["kalshi_f1_prob"] is not None else "—"
            edge_val = r["f1_edge"]
        else:
            model_pct = f"{r['model_f2_prob']:.0%}"
            kalshi_pct = f"{r['kalshi_f2_prob']:.0%}" if r["kalshi_f2_prob"] is not None else "—"
            edge_val = r["f2_edge"]

        edge_str = f"+{edge_val:.0%}" if edge_val and edge_val > 0 else (
            f"{edge_val:.0%}" if edge_val is not None else "—"
        )

        rec = r["bet_recommendation"]
        if rec.startswith("BET"):
            rec_styled = f"[green bold]{rec}[/green bold]"
            edge_styled = f"[green]{edge_str}[/green]"
        elif rec == "PASS":
            rec_styled = f"[yellow]{rec}[/yellow]"
            edge_styled = edge_str
        else:
            rec_styled = f"[dim]{rec}[/dim]"
            edge_styled = "—"

        table.add_row(
            matchup,
            r.get("weight_class", ""),
            winner,
            conf,
            model_pct,
            kalshi_pct,
            edge_styled,
            rec_styled,
        )

    console.print(table)

    # Summary
    value_bets = [r for r in results if r["bet_recommendation"].startswith("BET")]
    if value_bets:
        console.print(f"\n[green bold]Value bets found: {len(value_bets)}[/green bold]")
        for vb in value_bets:
            edge = f"{vb['bet_edge']:.1%}" if vb["bet_edge"] else "?"
            console.print(
                f"  → [green]{vb['bet_on']}[/green] "
                f"(model: {vb['model_f1_prob' if vb['bet_on'] == vb['fighter1'] else 'model_f2_prob']:.0%}, "
                f"Kalshi: {vb['kalshi_f1_prob' if vb['bet_on'] == vb['fighter1'] else 'kalshi_f2_prob']:.0%}, "
                f"edge: +{edge})"
            )
    else:
        no_market = [r for r in results if r["bet_recommendation"] == "NO MARKET"]
        if no_market:
            console.print(
                f"\n[dim]{len(no_market)} fight(s) had no matching Kalshi market.[/dim]"
            )
        console.print("\n[yellow]No strong value bets found on this card.[/yellow]")

    console.print(
        "\n[dim]Note: Model accuracy is ~63%. This is for informational purposes only. "
        "Bet responsibly.[/dim]"
    )


def main():
    parser = argparse.ArgumentParser(description="UFC Betting Predictor")
    parser.add_argument("--train", action="store_true", help="Retrain model from scratch")
    parser.add_argument("--no-kalshi", action="store_true", help="Skip Kalshi market lookup")
    parser.add_argument("--event", type=str, default=None, help="Filter by event name")
    args = parser.parse_args()

    print_banner()

    # 1. Load fighter stats
    console.print("[cyan]Loading fighter stats...[/cyan]")
    fighters_df = scrape_fighter_stats()
    console.print(f"[dim]Loaded {len(fighters_df)} fighters.[/dim]")

    # 2. Load or train model
    model = load_model()
    if model is None or args.train:
        console.print("[cyan]Training prediction model...[/cyan]")
        fights_df = scrape_fight_history()
        if fights_df.empty:
            console.print("[red]No fight history available for training.[/red]")
            sys.exit(1)
        model = train(fights_df, fighters_df)
    else:
        console.print("[dim]Loaded existing model.[/dim]")

    # 3. Get upcoming fight card
    console.print("[cyan]Fetching upcoming fight card...[/cyan]")
    fights = scrape_upcoming_card()

    if not fights:
        console.print("[red]No upcoming fights found.[/red]")
        sys.exit(0)

    if args.event:
        fights = [f for f in fights if args.event.lower() in f.get("event", "").lower()]
        if not fights:
            console.print(f"[red]No fights found for event '{args.event}'.[/red]")
            sys.exit(0)

    console.print(f"[dim]Found {len(fights)} fights.[/dim]")

    # 4. Fetch Kalshi markets
    ufc_markets = []
    if not args.no_kalshi:
        console.print("[cyan]Fetching Kalshi UFC markets...[/cyan]")
        ufc_markets = kalshi_client.get_ufc_markets()

    # 5. Analyze and display
    results = pred_module.analyze_card(fights, fighters_df, model, ufc_markets)
    print_results(results)


if __name__ == "__main__":
    main()
