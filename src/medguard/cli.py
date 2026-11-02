"""MedGuard command-line interface.

Examples::

    medguard datasets                       # list the dataset registry
    medguard analyze "Tab Dolo 650mg, Tab Azithral 500mg"
    medguard analyze --language hi "Tab Metformin 500mg, Tab Gluconil 5mg"
    medguard lookup "warfarin" "aspirin"
    medguard splits --strategy cold_start   # build leak-free splits
    medguard eval --table data/processed/interactions.parquet
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from medguard.config import get_settings
from medguard.data.registry import describe_all
from medguard.data.splits import SplitStrategy, make_splits
from medguard.pipeline import MedGuardPipeline, PipelineConfig

app = typer.Typer(
    name="medguard",
    help="MedGuard - prescription safety AI (decision support, not a diagnostic device).",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def datasets() -> None:
    """Show the registry of public datasets MedGuard uses."""
    console.print(describe_all())


@app.command()
def analyze(
    text: str = typer.Argument(..., help="Raw prescription text."),
    language: str = typer.Option("en", "--language", "-l", help="en | hi"),
    table: Path | None = typer.Option(None, "--table", help="Interaction table parquet."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Analyse a prescription given as text."""
    _setup_logging(verbose)
    settings = get_settings()
    pipeline = MedGuardPipeline(
        settings=settings,
        config=PipelineConfig(language=language),
    )

    interactions = table or settings.interactions_path
    if interactions.exists():
        import pandas as pd

        from medguard.interaction.retrieve import KnowledgeBase

        pipeline.kb = KnowledgeBase.from_frame(pd.read_parquet(interactions))
    else:
        console.print("[yellow]No knowledge base found. Run `make data` to download one.[/yellow]")
    result = pipeline.analyze_text(text, language=language)

    table_out = Table(title=f"MedGuard analysis ({language})")
    table_out.add_column("Medicine", style="cyan")
    table_out.add_column("Dose")
    table_out.add_column("Confidence")
    for drug in result.drugs:
        conf = f"{drug.confidence.calibrated:.2f}" if drug.confidence else "-"
        table_out.add_row(drug.normalized or drug.surface, drug.dose or "-", conf)
    console.print(table_out)

    if result.findings:
        findings_table = Table(title="Interaction findings")
        findings_table.add_column("Pair")
        findings_table.add_column("Severity", style="bold")
        findings_table.add_column("Mechanism")
        findings_table.add_column("Source")
        for finding in result.findings:
            findings_table.add_row(
                f"{finding.drug_a} + {finding.drug_b}",
                finding.severity.value,
                finding.event_type or "-",
                finding.source,
            )
        console.print(findings_table)

    for explanation in result.explanations:
        console.print(f"\n[bold]{explanation.text}[/bold]\n")

    if result.abstention:
        console.print(f"[yellow]Abstained: {result.abstention.reason}[/yellow]")
    console.print(f"[dim]processed in {result.processing_ms} ms[/dim]")


@app.command()
def lookup(
    drug_a: str = typer.Argument(...),
    drug_b: str = typer.Argument(...),
    table: Path | None = typer.Option(None, "--table"),
) -> None:
    """Look up one drug pair in the knowledge base."""
    settings = get_settings()
    interactions = table or settings.interactions_path
    if not interactions.exists():
        console.print("[red]Knowledge base not found. Run `make data` first.[/red]")
        raise typer.Exit(code=1)

    import pandas as pd

    from medguard.interaction.retrieve import KnowledgeBase, rows_to_findings

    kb = KnowledgeBase.from_frame(pd.read_parquet(interactions))
    findings = rows_to_findings(kb.lookup(drug_a, drug_b))
    if not findings:
        console.print(f"[yellow]No interaction recorded for {drug_a} + {drug_b}.[/yellow]")
        raise typer.Exit(code=0)
    console.print_json(json.dumps([f.model_dump(mode="json") for f in findings], indent=2))


@app.command()
def splits(
    table: Path = typer.Option(..., "--table", help="Canonical interaction table."),
    out_dir: Path = typer.Option(Path("data/processed"), "--out-dir"),
    strategy: str = typer.Option("pairwise", "--strategy", help="random | pairwise | cold_start"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Build leak-free train/validation/test splits."""
    import pandas as pd

    frame = pd.read_parquet(table)
    train, val, test, report = make_splits(
        frame,
        strategy=SplitStrategy(strategy),
        seed=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    (out_dir / "split_report.json").write_text(json.dumps(report.to_dict(), indent=2))

    console.print_json(json.dumps(report.to_dict(), indent=2))
    for warning in report.warnings:
        console.print(f"[yellow]WARNING: {warning}[/yellow]")
    console.print(f"[green]Saved splits to {out_dir}[/green]")


@app.command()
def eval(
    table: Path = typer.Option(..., "--table", help="Canonical interaction table."),
    strategy: str = typer.Option("pairwise", "--strategy"),
    seed: int = typer.Option(42, "--seed"),
    out: Path | None = typer.Option(None, "--out", help="Write metrics JSON here."),
) -> None:
    """Run the baseline evaluation and print the metrics table."""
    import numpy as np
    import pandas as pd

    from medguard.evaluation.metrics import (
        calibration_metrics,
        classification_metrics,
        selective_risk_coverage,
    )
    from medguard.modeling.baseline import DrugPairTfidfBaseline, build_baseline_df

    frame = pd.read_parquet(table)
    train, val, test, report = make_splits(frame, strategy=SplitStrategy(strategy), seed=seed)

    # Build a labelled training set from the KB plus sampled negatives.
    all_drugs = sorted({str(d) for d in frame["drug_a"]} | {str(d) for d in frame["drug_b"]})
    negatives = [
        (all_drugs[i], all_drugs[j])
        for i in range(len(all_drugs))
        for j in range(i + 1, len(all_drugs))
    ]
    train_df = build_baseline_df(train, negatives)

    model = DrugPairTfidfBaseline().fit(train_df)

    test_pairs = list(zip(test["drug_a"], test["drug_b"], strict=False))
    labels = [1] * len(test_pairs)  # KB pairs are positives
    preds = model.predict(test_pairs)
    probs = model.predict_proba(test_pairs)[:, -1]

    metrics = classification_metrics(np.array(labels), preds)
    calib = calibration_metrics(np.array(labels), probs)
    curve = selective_risk_coverage(np.array(labels), probs)

    console.print(f"\n[bold]Split report[/bold]\n{json.dumps(report.to_dict(), indent=2)}")
    for warning in report.warnings:
        console.print(f"[yellow]WARNING: {warning}[/yellow]")
    console.print(f"\n[bold]Classification[/bold]\n{json.dumps(metrics.to_dict(), indent=2)}")
    console.print(f"\n[bold]Calibration[/bold]\n{json.dumps(calib.to_dict(), indent=2)}")

    curve_table = Table(title="Risk-coverage")
    curve_table.add_column("Threshold")
    curve_table.add_column("Coverage")
    curve_table.add_column("Risk")
    for row in curve[::4]:
        curve_table.add_row(
            f"{row['threshold']:.2f}",
            f"{row['coverage']:.2f}",
            f"{row['risk']:.3f}",
        )
    console.print(curve_table)

    if out:
        out.write_text(
            json.dumps(
                {
                    "split": report.to_dict(),
                    "classification": metrics.to_dict(),
                    "calibration": calib.to_dict(),
                    "risk_coverage": curve,
                },
                indent=2,
            )
        )
        console.print(f"[green]Wrote {out}[/green]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the API server."""
    import uvicorn

    uvicorn.run("medguard.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
