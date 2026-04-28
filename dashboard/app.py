"""
dashboard/app.py
═══════════════════════════════════════════════════════════════════════════════
Terminal Dashboard for UFC Matchmaker

Built with the `rich` library — renders colourful, formatted tables and panels
directly in the terminal.  No browser needed; everything runs in a shell.

Key screens:
  • print_matchup_report  — ranked list of predicted matchups for a division
  • print_fight_card      — full suggested card across weight classes
  • print_specific_matchup — detailed breakdown of one named matchup
  • print_db_stats        — database health / record counts
  • print_training_eval   — model evaluation metrics after training
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
)
from rich import print as rprint

from models.matchmaker import MatchupResult, FighterProfile

logger = logging.getLogger(__name__)

# Shared console — use this everywhere so output order is deterministic
console = Console()

# ── Brand colours (matches the project's tactical/HUD aesthetic) ─────────────
CLR_GOLD    = "bold yellow"
CLR_RED     = "bold red"
CLR_GREEN   = "bold green"
CLR_CYAN    = "bold cyan"
CLR_DIM     = "dim white"
CLR_TITLE   = "bold white on dark_red"
CLR_HEADER  = "bold white"
CLR_SCORE_HI = "bold green"
CLR_SCORE_MID = "yellow"
CLR_SCORE_LO  = "red"


# ─────────────────────────────────────────────────────────────────────────────
# Score colouring helper
# ─────────────────────────────────────────────────────────────────────────────

def _score_style(score: float) -> str:
    """Return a Rich colour style string based on score value."""
    if score >= 80:
        return CLR_SCORE_HI
    elif score >= 60:
        return CLR_SCORE_MID
    else:
        return CLR_SCORE_LO


def _score_text(score: float) -> Text:
    """Render a score as a coloured Rich Text object."""
    t = Text(f"{score:.1f}", style=_score_style(score))
    return t


def _bar(value: float, width: int = 10, filled_char: str = "█", empty_char: str = "░") -> str:
    """
    ASCII progress bar.
      value : 0.0 – 1.0
      width : total bar length in characters
    """
    filled = round(value * width)
    return filled_char * filled + empty_char * (width - filled)


def _stars(value: float, n: int = 5) -> str:
    """Convert 0–1 float to star string: ★★★★☆"""
    filled = round(value * n)
    return "★" * filled + "☆" * (n - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Main matchup report
# ─────────────────────────────────────────────────────────────────────────────

def print_matchup_report(
    results: list[MatchupResult],
    weight_class: str,
    show_narrative: bool = True,
    show_subscores: bool = True,
):
    """
    Print a full ranked matchup report for a weight class.

    Each row shows:
      Rank | Fighter A vs Fighter B | Score | Stars breakdown | Finish% | Narrative
    """
    if not results:
        console.print(f"[{CLR_RED}]No matchups found for {weight_class}.[/]")
        return

    # ── Header ───────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Rule(
            f"[{CLR_TITLE}]  UFC MATCHMAKER — {weight_class.upper()}  [{CLR_TITLE}]",
            style="red",
        )
    )
    console.print(
        f"  [dim]Top {len(results)} predicted matchups   "
        f"72 features (including matchup cross-features)   "
        f"model blend: 70% NN score | 30% business overlay[/dim]"
    )
    console.print()

    # ── Table ─────────────────────────────────────────────────────────────────
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style=CLR_HEADER,
        expand=True,
        padding=(0, 1),
    )

    table.add_column("#",        style="dim", width=3,  justify="right")
    table.add_column("MATCHUP",              width=38)
    table.add_column("SCORE",    width=7,    justify="center")
    table.add_column("NN",       width=6,    justify="center", style="dim")
    table.add_column("BIZ",      width=6,    justify="center", style="dim")
    table.add_column("CLASH",    width=12,   justify="left")
    table.add_column("FINISH%",  width=9,    justify="center")
    table.add_column("BALANCE",  width=12,   justify="left")
    table.add_column("PACE",     width=12,   justify="left")
    table.add_column("TITLE?",   width=7,    justify="center")

    for rank, result in enumerate(results, start=1):
        fa = result.fighter_a
        fb = result.fighter_b

        # Fighter A label: "Islam Makhachev (C) 25-2"
        def fighter_label(f: FighterProfile) -> str:
            rank_str = f" [{f.ranked_label}]" if f.ranking or f.is_champion else ""
            return f"{f.name}{rank_str}  {f.record}"

        matchup_text = Text()
        matchup_text.append(fighter_label(fa), style="bold")
        matchup_text.append("\n  vs  ", style="dim")
        matchup_text.append(fighter_label(fb), style="bold")

        # Score cell — coloured
        score_cell = Text(f"{result.final_score:.1f}", style=_score_style(result.final_score))
        score_cell.append("\n/100", style="dim")

        # Sub-score bars
        clash_cell   = f"[cyan]{_stars(result.style_clash)}[/cyan]"
        balance_cell = f"[green]{_stars(result.competitive_balance)}[/green]"
        pace_cell    = f"[yellow]{_stars(result.action_density)}[/yellow]"

        finish_pct = f"{result.finish_probability * 100:.0f}%"
        title_cell = "[bold green]YES[/bold green]" if result.title_implications else "[dim]—[/dim]"

        table.add_row(
            str(rank),
            matchup_text,
            score_cell,
            f"[dim]{result.nn_score:.1f}[/dim]",
            f"[dim]{result.business_score:.1f}[/dim]",
            clash_cell,
            finish_pct,
            balance_cell,
            pace_cell,
            title_cell,
        )

        # Optional narrative row (dimmer, indented)
        if show_narrative and result.narrative:
            table.add_row(
                "", Text(f"   ↳ {result.narrative}", style="dim italic"), "", "", "", "", "", "", "", ""
            )
            table.add_row("", "", "", "", "", "", "", "", "", "")  # spacer

    console.print(table)
    _print_legend()


def _print_legend():
    """Print a small column-header legend below the table."""
    console.print(
        "  [dim]CLASH = style contrast  "
        "BALANCE = competitive evenness  "
        "PACE = predicted action density  "
        "BIZ = business overlay score[/dim]"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Single matchup deep-dive
# ─────────────────────────────────────────────────────────────────────────────

def print_specific_matchup(result: MatchupResult):
    """
    Print a detailed breakdown for one specific matchup.
    Shows fighter cards side-by-side and a full sub-score breakdown.
    """
    fa = result.fighter_a
    fb = result.fighter_b

    console.print()
    console.print(Rule(style="red"))

    # ── Title panel ──────────────────────────────────────────────────────────
    title = (
        f"[bold white]  {fa.name.upper()}  [{fa.ranked_label}]  "
        f"[red]VS[/red]  "
        f"{fb.name.upper()}  [{fb.ranked_label}]  [/bold white]"
    )
    console.print(Panel(title, style="red", expand=True))

    # ── Score panel ──────────────────────────────────────────────────────────
    score_panel_content = (
        f"[bold]FINAL SCORE[/bold]  "
        f"[{_score_style(result.final_score)}]{result.final_score:.1f} / 100[/]    "
        f"[dim]NN: {result.nn_score:.1f}  |  Business: {result.business_score:.1f}[/dim]"
    )
    console.print(Panel(score_panel_content, box=box.SIMPLE))

    # ── Fighter cards side by side ────────────────────────────────────────────
    def fighter_card(f: FighterProfile) -> Panel:
        total = max(f.wins_total + f.losses_total, 1)
        wp = f.wins_total / total * 100

        lines = [
            f"[bold]{f.name}[/bold]  [{f.ranked_label}]",
            f"  Weight class : {f.weight_class}",
            f"  Record       : {f.record}  ({wp:.0f}% win rate)",
            f"  Finish rate  : {f.finish_rate * 100:.0f}%  "
            f"(KO: {f.raw.get('ko_rate', 0) * 100:.0f}%  "
            f"Sub: {f.raw.get('sub_rate', 0) * 100:.0f}%)",
            f"  Sig str/min  : {f.sig_strikes_pm:.1f}",
            f"  Grapple ratio: {f.raw.get('grapple_ratio', 0):.2f}  "
            f"({'grappler' if (f.raw.get('grapple_ratio') or 0) > 0.45 else 'striker'})",
        ]
        return Panel("\n".join(lines), title=f.name, border_style="cyan")

    console.print(Columns([fighter_card(fa), fighter_card(fb)]))

    # ── Sub-score breakdown ───────────────────────────────────────────────────
    breakdown = Table(box=box.SIMPLE, show_header=False, expand=False, padding=(0, 2))
    breakdown.add_column("Metric",  style="bold", width=22)
    breakdown.add_column("Bar",     width=14)
    breakdown.add_column("Score",   width=6, justify="right")
    breakdown.add_column("Notes",   style="dim", width=40)

    sub_scores = [
        (
            "Style Clash",
            result.style_clash,
            "cyan",
            _style_clash_label(fa, fb),
        ),
        (
            "Finish Probability",
            result.finish_probability,
            "red",
            f"~{result.finish_probability * 100:.0f}% chance of finish",
        ),
        (
            "Competitive Balance",
            result.competitive_balance,
            "green",
            "How evenly matched on paper",
        ),
        (
            "Action Density",
            result.action_density,
            "yellow",
            f"~{(fa.sig_strikes_pm or 3) + (fb.sig_strikes_pm or 3):.1f} combined sig str/min",
        ),
    ]

    for label, value, colour, note in sub_scores:
        bar_str = f"[{colour}]{_bar(value)}[/{colour}]"
        pct_str = f"{value * 100:.0f}%"
        breakdown.add_row(label, bar_str, pct_str, note)

    console.print(Panel(breakdown, title="Sub-Score Breakdown", border_style="dim"))

    # ── Narrative ─────────────────────────────────────────────────────────────
    if result.narrative:
        console.print(
            Panel(
                f"[italic]{result.narrative}[/italic]",
                title="Fight Narrative",
                border_style="dim",
            )
        )

    if result.title_implications:
        console.print(
            Panel(
                "[bold yellow]⚑  This fight has TITLE IMPLICATIONS[/bold yellow]",
                border_style="yellow",
            )
        )

    console.print()


def _style_clash_label(fa: FighterProfile, fb: FighterProfile) -> str:
    """Human-readable style clash description."""
    gr_a = fa.raw.get("grapple_ratio") or 0.3
    gr_b = fb.raw.get("grapple_ratio") or 0.3
    diff = abs(gr_a - gr_b)
    if diff > 0.5:
        s = fa if gr_a < gr_b else fb
        g = fb if gr_a < gr_b else fa
        return f"Strong: {s.name} (striker) vs {g.name} (grappler)"
    elif diff > 0.25:
        return "Moderate style contrast"
    else:
        return "Similar styles — technical chess match likely"


# ─────────────────────────────────────────────────────────────────────────────
# Full card view
# ─────────────────────────────────────────────────────────────────────────────

def print_fight_card(card: dict[str, list[MatchupResult]]):
    """
    Print a suggested full fight card across all weight classes.
    Formatted like an official UFC fight card poster.
    """
    console.print()
    console.print(
        Panel(
            "[bold white on dark_red]   UFC MATCHMAKER — SUGGESTED FIGHT CARD   [/bold white on dark_red]",
            expand=True,
        )
    )

    fight_num = 1
    for weight_class, matchups in card.items():
        if not matchups:
            continue

        console.print(f"\n  [bold cyan]{weight_class.upper()}[/bold cyan]")
        console.print(f"  {'─' * 60}")

        for matchup in matchups:
            fa, fb = matchup.fighter_a, matchup.fighter_b
            score_str = f"[{_score_style(matchup.final_score)}]{matchup.final_score:.1f}[/]"

            console.print(
                f"  [{fight_num:2d}]  "
                f"[bold]{fa.name}[/bold] [{fa.ranked_label}]  {fa.record}"
                f"  [dim]vs[/dim]  "
                f"[bold]{fb.name}[/bold] [{fb.ranked_label}]  {fb.record}"
                f"  [dim]│[/dim]  Score: {score_str}"
                f"  [dim]│  Finish: {matchup.finish_probability * 100:.0f}%[/dim]"
            )
            fight_num += 1

    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# DB stats
# ─────────────────────────────────────────────────────────────────────────────

def print_db_stats(stats: dict):
    """Print a summary of what's currently in the database."""
    console.print()
    console.print(
        Panel(
            f"[bold]DATABASE STATUS[/bold]\n\n"
            f"  Fighters : [bold green]{stats['fighters']:,}[/bold green]\n"
            f"  Fights   : [bold green]{stats['fights']:,}[/bold green]\n"
            f"  Events   : [bold green]{stats['events']:,}[/bold green]",
            title="UFC Matchmaker DB",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation output
# ─────────────────────────────────────────────────────────────────────────────

def print_training_eval(metrics: dict):
    """
    Print regression metrics after FightQualityNN training (fight-quality targets).

    Displays MAE, RMSE, and R² on the 0–100 scale — applies when training.py's
    regression loop has run successfully.
    """
    console.print()

    r2 = metrics.get("r2", 0)
    r2_style = "bold green" if r2 > 0.8 else ("yellow" if r2 > 0.6 else "red")

    console.print(
        Panel(
            f"[bold]MODEL EVALUATION[/bold]\n\n"
            f"  MAE   : [bold]{metrics.get('mae', 0):.2f}[/bold] points   "
            f"[dim](avg absolute error on 0-100 scale)[/dim]\n"
            f"  RMSE  : [bold]{metrics.get('rmse', 0):.2f}[/bold] points   "
            f"[dim](penalises large errors more)[/dim]\n"
            f"  R²    : [{r2_style}]{r2:.3f}[/{r2_style}]        "
            f"[dim](1.0 = perfect fit)[/dim]",
            title="Training Complete",
            border_style="green",
            expand=False,
        )
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Rich progress bar (re-exported for convenience)
# ─────────────────────────────────────────────────────────────────────────────

def make_progress() -> Progress:
    """Create a styled Rich progress bar for use in long scraping loops."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
