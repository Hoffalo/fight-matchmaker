"""
main.py
═══════════════════════════════════════════════════════════════════════════════
UFC Matchmaker — Command Line Interface

All commands are defined here using Typer.
Run `python main.py --help` to see available commands.

Commands
────────
  collect   — Run the full data scraping pipeline
  train     — Train the Fight Quality Neural Network
  predict   — Generate matchup predictions for a weight class
  matchup   — Score one specific named fighter pairing
  card      — Build a suggested full event card
  stats     — Show current database record counts
  evaluate  — Evaluate a trained model on the full dataset
  demo      — Run with synthetic demo data (no scraping needed)
═══════════════════════════════════════════════════════════════════════════════
"""
import sys
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from utils.logger import setup_logging
from data.db import Database
from config import WEIGHT_CLASSES, NN, BASE_DIR

# ── App setup ────────────────────────────────────────────────────────────────
app     = typer.Typer(
    name="ufc-matchmaker",
    help=(
        "AI-powered UFC fight pairing predictor.\n\n"
        "Scrapes fighter and fight data, trains a Neural Network on fight quality,\n"
        "and predicts the most entertaining / commercially successful matchups."
    ),
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# collect
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def collect(
    events: Optional[int]   = typer.Option(None,  "--events",   "-e", help="Max events to scrape (default: all)"),
    fighters: Optional[int] = typer.Option(None,  "--fighters", "-f", help="Max fighter profiles to enrich"),
    skip_rankings: bool     = typer.Option(False, "--skip-rankings",  help="Skip Tapology rankings scrape"),
    headless: bool          = typer.Option(True,  "--headless/--no-headless", help="Run browser headlessly"),
    verbose: bool           = typer.Option(False, "--verbose",  "-v", help="Debug-level logging"),
):
    """
    [bold]Scrape UFC data and populate the local database.[/bold]

    Runs the full data collection pipeline:
      1. Events + fights from UFCStats.com (Selenium)
      2. Fighter career stats from UFCStats.com (Selenium)
      3. Rankings from Tapology.com (Selenium)
      4. Derived metrics + fight quality scores computed locally

    This command should be run [italic]before[/italic] training.
    """
    setup_logging("DEBUG" if verbose else "INFO",
                  log_file=str(BASE_DIR / "logs" / "collect.log"))

    from data.pipeline import DataPipeline

    db = Database()
    console.print("\n[bold cyan]Starting data collection pipeline...[/bold cyan]\n")

    pipeline = DataPipeline(db, headless=headless)
    pipeline.run_full_collection(
        max_events=events,
        max_fighters=fighters,
        skip_rankings=skip_rankings,
    )

    from dashboard import print_db_stats
    print_db_stats(db.get_stats())


# ─────────────────────────────────────────────────────────────────────────────
# train
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def train(
    epochs: int   = typer.Option(NN["epochs"],        "--epochs",    "-e", help="Number of training epochs"),
    lr: float     = typer.Option(NN["learning_rate"], "--lr",              help="Learning rate"),
    batch: int    = typer.Option(NN["batch_size"],    "--batch",     "-b", help="Batch size"),
    evaluate_after: bool = typer.Option(True, "--eval/--no-eval",         help="Evaluate model after training"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    [bold]Train the Fight Quality Neural Network.[/bold]

    Requires the database to have fight records with computed quality scores.
    Run [cyan]collect[/cyan] first.

    The trained model is saved to [dim]models/fight_quality_nn.pt[/dim].
    A feature scaler is saved to [dim]models/feature_scaler.pkl[/dim].
    """
    setup_logging("DEBUG" if verbose else "INFO",
                  log_file=str(BASE_DIR / "logs" / "train.log"))

    from models.training import train as run_training, evaluate as eval_model
    from dashboard import print_training_eval, console as dash_console

    # Allow CLI overrides to config
    cfg = dict(NN)
    cfg["epochs"]        = epochs
    cfg["learning_rate"] = lr
    cfg["batch_size"]    = batch

    db = Database()
    dash_console.print("\n[bold cyan]Training Fight Quality Neural Network...[/bold cyan]\n")

    model = run_training(db=db, cfg=cfg)

    if evaluate_after:
        metrics = eval_model(model, db, cfg=cfg)
        print_training_eval(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# predict
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def predict(
    weight_class: str       = typer.Option("Lightweight", "--weight-class", "-w",
                                           help=f"One of: {', '.join(WEIGHT_CLASSES)}"),
    top: int                = typer.Option(20,    "--top",     "-n", help="Number of matchups to show"),
    min_fights: int         = typer.Option(3,     "--min-fights",   help="Minimum career fights to include a fighter"),
    ranked_only: bool       = typer.Option(False, "--ranked-only",  help="Only include currently ranked fighters"),
    heuristic: bool         = typer.Option(False, "--heuristic",    help="Use heuristic scorer (no trained model needed)"),
    no_narrative: bool      = typer.Option(False, "--no-narrative", help="Hide narrative descriptions"),
    verbose: bool           = typer.Option(False, "--verbose", "-v"),
):
    """
    [bold]Predict the best matchups for a weight class.[/bold]

    Generates all valid pairings within the division and ranks them by
    predicted fight quality + business value.

    Requires a trained model unless [cyan]--heuristic[/cyan] is passed.
    """
    setup_logging("DEBUG" if verbose else "INFO")

    from dashboard import print_matchup_report

    # Validate weight class
    wc_lower = [w.lower() for w in WEIGHT_CLASSES]
    if weight_class.lower() not in wc_lower:
        console.print(
            f"[red]Unknown weight class '{weight_class}'.[/red]\n"
            f"Valid options: {', '.join(WEIGHT_CLASSES)}"
        )
        raise typer.Exit(1)
    canonical_wc = WEIGHT_CLASSES[wc_lower.index(weight_class.lower())]

    db = Database()

    if heuristic:
        from models.matchmaker import HeuristicMatchmaker
        console.print(f"\n[yellow]Using heuristic scorer (no model)[/yellow]")
        mm = HeuristicMatchmaker(db)
        results = mm.predict_weight_class(canonical_wc, top_n=top, min_fights=min_fights)
        print_matchup_report(results, canonical_wc, show_narrative=not no_narrative)
    else:
        from models.matchmaker import Matchmaker
        try:
            with Matchmaker(db) as mm:
                results = mm.predict_weight_class(
                    canonical_wc,
                    top_n=top,
                    min_fights=min_fights,
                    ranked_only=ranked_only,
                )
            print_matchup_report(results, canonical_wc, show_narrative=not no_narrative)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            console.print("[dim]Tip: use --heuristic to run without a trained model.[/dim]")
            raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# matchup (specific named pairing)
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def matchup(
    fighter_a: str = typer.Argument(..., help="First fighter name (as in DB)"),
    fighter_b: str = typer.Argument(..., help="Second fighter name"),
    heuristic: bool = typer.Option(False, "--heuristic"),
    verbose: bool   = typer.Option(False, "--verbose", "-v"),
):
    """
    [bold]Score one specific fighter pairing.[/bold]

    Example:
      python main.py matchup "Dustin Poirier" "Justin Gaethje"
    """
    setup_logging("DEBUG" if verbose else "INFO")

    from dashboard import print_specific_matchup

    db = Database()

    if heuristic:
        from models.matchmaker import HeuristicMatchmaker
        mm = HeuristicMatchmaker(db)
        # HeuristicMatchmaker doesn't expose predict_specific_matchup,
        # so we use the full NN matchmaker path but will fall back
        pass

    from models.matchmaker import Matchmaker
    try:
        with Matchmaker(db) as mm:
            result = mm.predict_specific_matchup(fighter_a, fighter_b)
        if result is None:
            console.print(f"[red]Could not score matchup. Check fighter names.[/red]")
            raise typer.Exit(1)
        print_specific_matchup(result)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# card (suggested full event)
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def card(
    weight_classes: Optional[str] = typer.Option(
        None, "--weight-classes", "-w",
        help="Comma-separated weight classes. Default: all."
    ),
    fights_per_class: int = typer.Option(2, "--per-class", "-n",
                                         help="Fights per weight class"),
    heuristic: bool = typer.Option(False, "--heuristic"),
    verbose: bool   = typer.Option(False, "--verbose", "-v"),
):
    """
    [bold]Build a suggested full event fight card.[/bold]

    Selects the best non-overlapping matchups across all weight classes
    (no fighter appears twice on the card).
    """
    setup_logging("DEBUG" if verbose else "INFO")

    from dashboard import print_fight_card

    wcs = (
        [w.strip() for w in weight_classes.split(",")]
        if weight_classes
        else WEIGHT_CLASSES
    )

    db = Database()

    if heuristic:
        from models.matchmaker import HeuristicMatchmaker
        mm = HeuristicMatchmaker(db)
        fight_card = {}
        used = set()
        for wc in wcs:
            candidates = mm.predict_weight_class(wc, top_n=fights_per_class * 5)
            selected = []
            for m in candidates:
                if m.fighter_a.id not in used and m.fighter_b.id not in used:
                    selected.append(m)
                    used.add(m.fighter_a.id)
                    used.add(m.fighter_b.id)
                if len(selected) >= fights_per_class:
                    break
            if selected:
                fight_card[wc] = selected
    else:
        from models.matchmaker import Matchmaker
        try:
            with Matchmaker(db) as mm:
                fight_card = mm.predict_card(
                    weight_classes=wcs,
                    n_per_class=fights_per_class,
                )
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    print_fight_card(fight_card)


# ─────────────────────────────────────────────────────────────────────────────
# stats
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def stats():
    """[bold]Show current database record counts.[/bold]"""
    from dashboard import print_db_stats
    db = Database()
    print_db_stats(db.get_stats())


# ─────────────────────────────────────────────────────────────────────────────
# evaluate
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def evaluate(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """
    [bold]Evaluate the trained model on the full dataset.[/bold]

    Prints MAE, RMSE, and R² metrics.
    """
    setup_logging("DEBUG" if verbose else "INFO")

    from models.training import load_model, evaluate as eval_model
    from dashboard import print_training_eval

    db = Database()
    try:
        model = load_model()
        metrics = eval_model(model, db)
        print_training_eval(metrics)
    except FileNotFoundError:
        console.print("[red]No trained model found. Run: python main.py train[/red]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# demo — synthetic data, no scraping, no model needed
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def demo(
    weight_class: str = typer.Option("Lightweight", "--weight-class", "-w"),
    top: int          = typer.Option(10, "--top", "-n"),
):
    """
    [bold]Run a demo with synthetic fighter data.[/bold]

    Inserts a set of realistic fake fighters into the DB and runs the
    heuristic matchmaker — no internet access or trained model needed.
    Useful for testing the dashboard output format.
    """
    from dashboard import print_matchup_report, console as dash_console
    from models.matchmaker import HeuristicMatchmaker

    dash_console.print("\n[bold yellow]Running DEMO with synthetic data...[/bold yellow]\n")

    db = Database()
    _insert_demo_data(db, weight_class)

    mm = HeuristicMatchmaker(db)
    results = mm.predict_weight_class(weight_class, top_n=top, min_fights=0)
    print_matchup_report(results, weight_class)


def _insert_demo_data(db: Database, weight_class: str):
    """
    Insert a realistic set of synthetic fighters for demo purposes.
    Covers a range of styles: strikers, grapplers, balanced fighters.
    """
    synthetic_fighters = [
        # Strikers (low grapple_ratio, high sig_strikes_pm, high ko_rate)
        dict(name="Demo: Apex Striker",   weight_class=weight_class, ranking=1,
             wins_total=22, losses_total=2, ko_rate=0.72, sub_rate=0.05, dec_rate=0.23,
             finish_rate=0.77, sig_strikes_pm=8.4, sig_strike_acc=0.52,
             sig_strikes_abs_pm=3.1, sig_strike_def=0.62, td_avg=0.4, td_acc=0.30,
             td_def=0.78, sub_avg=0.1, grapple_ratio=0.08, ctrl_time_avg=15.0,
             height_cm=177.0, reach_cm=183.0, weight_lbs=155.0),

        dict(name="Demo: Iron Chin",       weight_class=weight_class, ranking=3,
             wins_total=19, losses_total=4, ko_rate=0.58, sub_rate=0.11, dec_rate=0.31,
             finish_rate=0.69, sig_strikes_pm=7.1, sig_strike_acc=0.47,
             sig_strikes_abs_pm=5.2, sig_strike_def=0.54, td_avg=0.6, td_acc=0.35,
             td_def=0.65, sub_avg=0.3, grapple_ratio=0.12, ctrl_time_avg=20.0,
             height_cm=175.0, reach_cm=178.0, weight_lbs=155.0),

        dict(name="Demo: Vicious Volume",  weight_class=weight_class, ranking=5,
             wins_total=17, losses_total=3, ko_rate=0.41, sub_rate=0.06, dec_rate=0.53,
             finish_rate=0.47, sig_strikes_pm=9.8, sig_strike_acc=0.44,
             sig_strikes_abs_pm=5.8, sig_strike_def=0.50, td_avg=0.3, td_acc=0.28,
             td_def=0.72, sub_avg=0.1, grapple_ratio=0.05, ctrl_time_avg=10.0,
             height_cm=180.0, reach_cm=185.0, weight_lbs=155.0),

        # Grapplers (high grapple_ratio, high td_avg, high sub_rate)
        dict(name="Demo: Ground Tyrant",   weight_class=weight_class, ranking=2,
             wins_total=24, losses_total=1, ko_rate=0.13, sub_rate=0.62, dec_rate=0.25,
             finish_rate=0.75, sig_strikes_pm=2.8, sig_strike_acc=0.49,
             sig_strikes_abs_pm=2.0, sig_strike_def=0.71, td_avg=6.1, td_acc=0.58,
             td_def=0.87, sub_avg=3.4, grapple_ratio=0.72, ctrl_time_avg=210.0,
             height_cm=176.0, reach_cm=177.0, weight_lbs=155.0),

        dict(name="Demo: The Submission",  weight_class=weight_class, ranking=4,
             wins_total=21, losses_total=3, ko_rate=0.10, sub_rate=0.71, dec_rate=0.19,
             finish_rate=0.81, sig_strikes_pm=3.2, sig_strike_acc=0.46,
             sig_strikes_abs_pm=2.5, sig_strike_def=0.68, td_avg=5.2, td_acc=0.51,
             td_def=0.80, sub_avg=4.1, grapple_ratio=0.68, ctrl_time_avg=190.0,
             height_cm=178.0, reach_cm=179.0, weight_lbs=155.0),

        # Balanced / all-rounders
        dict(name="Demo: Complete Package", weight_class=weight_class, ranking=6,
             wins_total=15, losses_total=3, ko_rate=0.40, sub_rate=0.33, dec_rate=0.27,
             finish_rate=0.73, sig_strikes_pm=5.5, sig_strike_acc=0.50,
             sig_strikes_abs_pm=3.4, sig_strike_def=0.60, td_avg=2.8, td_acc=0.45,
             td_def=0.74, sub_avg=1.2, grapple_ratio=0.35, ctrl_time_avg=80.0,
             height_cm=177.0, reach_cm=181.0, weight_lbs=155.0),

        dict(name="Demo: War Machine",     weight_class=weight_class, ranking=7,
             wins_total=18, losses_total=5, ko_rate=0.44, sub_rate=0.22, dec_rate=0.34,
             finish_rate=0.66, sig_strikes_pm=6.2, sig_strike_acc=0.46,
             sig_strikes_abs_pm=4.5, sig_strike_def=0.55, td_avg=2.1, td_acc=0.42,
             td_def=0.68, sub_avg=0.8, grapple_ratio=0.25, ctrl_time_avg=60.0,
             height_cm=176.0, reach_cm=180.0, weight_lbs=155.0),

        dict(name="Demo: Chaos Factor",    weight_class=weight_class, ranking=9,
             wins_total=14, losses_total=4, ko_rate=0.64, sub_rate=0.14, dec_rate=0.22,
             finish_rate=0.78, sig_strikes_pm=7.4, sig_strike_acc=0.43,
             sig_strikes_abs_pm=5.6, sig_strike_def=0.48, td_avg=1.2, td_acc=0.38,
             td_def=0.59, sub_avg=0.4, grapple_ratio=0.13, ctrl_time_avg=25.0,
             height_cm=174.0, reach_cm=176.0, weight_lbs=155.0),

        dict(name="Demo: The Surgeon",     weight_class=weight_class, ranking=10,
             wins_total=20, losses_total=2, ko_rate=0.35, sub_rate=0.10, dec_rate=0.55,
             finish_rate=0.45, sig_strikes_pm=5.8, sig_strike_acc=0.57,
             sig_strikes_abs_pm=2.8, sig_strike_def=0.67, td_avg=1.5, td_acc=0.48,
             td_def=0.76, sub_avg=0.6, grapple_ratio=0.20, ctrl_time_avg=45.0,
             height_cm=179.0, reach_cm=184.0, weight_lbs=155.0),

        dict(name="Demo: Unranked Danger", weight_class=weight_class, ranking=None,
             wins_total=12, losses_total=2, ko_rate=0.67, sub_rate=0.08, dec_rate=0.25,
             finish_rate=0.75, sig_strikes_pm=6.9, sig_strike_acc=0.49,
             sig_strikes_abs_pm=3.9, sig_strike_def=0.57, td_avg=0.9, td_acc=0.40,
             td_def=0.70, sub_avg=0.2, grapple_ratio=0.11, ctrl_time_avg=22.0,
             height_cm=178.0, reach_cm=182.0, weight_lbs=155.0),
    ]

    for f in synthetic_fighters:
        db.upsert_fighter(f)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
