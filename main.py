"""
main.py
═══════════════════════════════════════════════════════════════════════════════
UFC Matchmaker — Command Line Interface

Run `python main.py --help` to see available commands.

Commands
────────
  collect       Run the full data scraping pipeline
  train         Retrain the 12-feat NN classifier on current data
  matchmake     Rank matchups in a weight class
  dreamcard     Build a dream card across divisions
  evaluate      Evaluate the trained NN on the test split
  backtest      Per-event backtest from a chosen cutoff date
  experiments   Generate experiment summary tables and plots
  demo          Quick demo: top 10 Lightweight matchups + dream card
  stats         Show current database record counts
  seed-db       Create a minimal SQLite DB for offline tests (no scraping)
  pca           Cumulative variance + PC1 loadings on training features
═══════════════════════════════════════════════════════════════════════════════
"""
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from utils.logger import setup_logging
from data.db import Database
from config import WEIGHT_CLASSES, BASE_DIR

app = typer.Typer(
    name="ufc-matchmaker",
    help=(
        "AI-powered UFC fight matchmaker. Trains a 12-feature neural network on "
        "fighter stats and ranks matchups by predicted entertainment value."
    ),
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# collect — scrape data into the local SQLite DB
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def collect(
    events: Optional[int]   = typer.Option(None,  "--events",   "-e", help="Max events to scrape (default: all)"),
    fighters: Optional[int] = typer.Option(None,  "--fighters", "-f", help="Max fighter profiles to enrich"),
    skip_rankings: bool     = typer.Option(False, "--skip-rankings",  help="Skip Tapology rankings scrape"),
    headless: bool          = typer.Option(True,  "--headless/--no-headless", help="Run browser headlessly"),
    verbose: bool           = typer.Option(False, "--verbose",  "-v", help="Debug-level logging"),
):
    """[bold]Scrape UFC data and populate the local database.[/bold]"""
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
# train — retrain the 12-feat binary NN
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def train(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """
    [bold]Retrain the 12-feature NN binary classifier.[/bold]

    Runs the small-data hyperparameter sweep on the RFECV-selected features and
    saves the best model + scaler to [dim]models/checkpoints/[/dim].
    """
    setup_logging("DEBUG" if verbose else "INFO",
                  log_file=str(BASE_DIR / "logs" / "train.log"))

    from models.nn_binary import run_twelve_feature_comparison

    console.print("\n[bold cyan]Training 12-feature binary classifier...[/bold cyan]\n")
    result = run_twelve_feature_comparison()
    console.print(
        f"\n[green]Done.[/green] best val AUC = {result['best_val_auc']:.4f}, "
        f"params = {result['n_parameters']}, ckpt = {result['checkpoint']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# matchmake — rank matchups in a weight class
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def matchmake(
    weight_class: str = typer.Argument(..., help=f"One of: {', '.join(WEIGHT_CLASSES)}"),
    top: int          = typer.Option(10, "--top", "-n", help="Number of matchups to show"),
    five_round: bool  = typer.Option(False, "--five-round/--three-round",
                                     help="Score as a 5-round main-event-style fight"),
):
    """[bold]Rank all possible pairings in a weight class.[/bold]"""
    wc_lower = [w.lower() for w in WEIGHT_CLASSES]
    if weight_class.lower() not in wc_lower:
        console.print(
            f"[red]Unknown weight class '{weight_class}'.[/red]\n"
            f"Valid options: {', '.join(WEIGHT_CLASSES)}"
        )
        raise typer.Exit(1)
    canonical_wc = WEIGHT_CLASSES[wc_lower.index(weight_class.lower())]

    from models.matchmaker_v2 import MatchmakerV2
    mm = MatchmakerV2()
    mm.rank_weight_class(canonical_wc, top_n=top, is_five_rounder=five_round)


# ─────────────────────────────────────────────────────────────────────────────
# dreamcard — build a dream card across divisions
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def dreamcard(
    fights: int = typer.Option(5, "--fights", "-n", help="Number of fights on the card"),
    divisions: Optional[str] = typer.Option(
        None, "--divisions", "-d",
        help="Comma-separated list of weight classes (default: all eight men's divisions)",
    ),
    max_per_division: int = typer.Option(
        1,
        "--max-per-division",
        "-m",
        help="Cap fights per weight class (0 = no cap; default 1 spreads the card across divisions).",
    ),
):
    """[bold]Build a dream card — best fights across divisions, no fighter repeats.[/bold]"""
    wcs = (
        [w.strip() for w in divisions.split(",")] if divisions else None
    )
    from models.matchmaker_v2 import MatchmakerV2
    mm = MatchmakerV2()
    mm.build_card(
        weight_classes=wcs,
        total_fights=fights,
        max_per_weight_class=None if max_per_division <= 0 else max_per_division,
    )


# ─────────────────────────────────────────────────────────────────────────────
# demo — quick top-10 + dream card
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def demo():
    """[bold]Quick demo: top 10 Lightweight matchups + 5-fight dream card.[/bold]"""
    from models.matchmaker_v2 import MatchmakerV2
    mm = MatchmakerV2()
    mm.rank_weight_class("Lightweight", top_n=10)
    mm.build_card(total_fights=5)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate — NN on test split
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def evaluate(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """[bold]Evaluate the trained 12-feat NN on the held-out test split.[/bold]"""
    setup_logging("DEBUG" if verbose else "INFO")

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, f1_score

    from config import FINAL_MODEL
    from models.data_loader import get_canonical_splits
    from models.nn_binary import FightBonusNN

    ckpt_path = Path(FINAL_MODEL["checkpoint"])
    if not ckpt_path.is_file():
        console.print(
            f"[red]Checkpoint not found: {ckpt_path}[/red]\n"
            "Run: [cyan]python main.py train[/cyan]"
        )
        raise typer.Exit(1)

    splits = get_canonical_splits()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = FightBonusNN(
        input_dim=cfg["input_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_te = torch.from_numpy(splits["X_test"].astype(np.float32))
    y_te = splits["y_test"].astype(np.int32)
    with torch.no_grad():
        logits = model(X_te).squeeze(-1).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int32)

    auc = roc_auc_score(y_te, probs)
    f1 = f1_score(y_te, preds, zero_division=0)
    pos_rate = float(y_te.mean())

    console.print(
        f"\n[bold]Test split[/bold]  N={len(y_te)}  pos_rate={pos_rate:.3f}\n"
        f"  AUC = [green]{auc:.4f}[/green]\n"
        f"  F1  = [green]{f1:.4f}[/green]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# backtest — per-event simulation from a cutoff
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def backtest(
    test_from: str = typer.Option("2026-01-01", "--from", help="Cutoff date (YYYY-MM-DD)"),
    k: int         = typer.Option(3, "--k", help="Top-K per event for hit-rate"),
    verbose: bool  = typer.Option(False, "--verbose", "-v"),
):
    """[bold]Per-event backtest from a chosen cutoff.[/bold]"""
    setup_logging("DEBUG" if verbose else "INFO")
    from models.backtesting import backtest as run_backtest

    db = Database()
    summary = run_backtest(db, test_date_from=test_from, k=k)
    console.print("\n[bold]Backtest summary:[/bold]")
    for key, val in summary.items():
        console.print(f"  {key}: {val}")


# ─────────────────────────────────────────────────────────────────────────────
# experiments — summary table + plots
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def experiments():
    """[bold]Generate experiment summary tables and plots.[/bold]"""
    from models.experiment_summary import main as run_experiments
    run_experiments()


# ─────────────────────────────────────────────────────────────────────────────
# stats — DB record counts
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def stats():
    """[bold]Show current database record counts.[/bold]"""
    from dashboard import print_db_stats
    db = Database()
    print_db_stats(db.get_stats())


# ─────────────────────────────────────────────────────────────────────────────
# seed-db — minimal SQLite for offline tests (no scraping)
# ─────────────────────────────────────────────────────────────────────────────

@app.command("seed-db")
def seed_db(
    db: Optional[str] = typer.Option(None, "--db", help="Database path (default: data/ufc_matchmaker.db)"),
):
    """[bold]Create a minimal database with train/val/test fights so PCA/modelling can run without scraping.[/bold]"""
    from data.seed_minimal_splits_db import seed
    path = seed(db_path=db)
    console.print(f"[green]Seeded minimal database:[/green] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# pca — cumulative variance / PC1 loadings on scaled 115-dim training features
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def pca(
    db: Optional[str] = typer.Option(None, "--db", help="Path to ufc_matchmaker.db"),
):
    """[bold]PCA on training features — cumulative variance and PC1 loadings.[/bold]"""
    from models.pca_analysis import format_pca_report, run_pca_from_db

    path = db or str(Path(__file__).resolve().parent / "data" / "ufc_matchmaker.db")
    try:
        result = run_pca_from_db(db_path=path)
        console.print(format_pca_report(result))
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
