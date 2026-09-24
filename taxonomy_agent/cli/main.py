"""
taxonomy-agent CLI

Commands:
  build-index     Embed taxonomy nodes and build HNSW index (run once, or after taxonomy changes)
  scan            Classify all files in a folder (batch mode)
  watch           Watch folders for new files and classify continuously
  review          Show files queued for review and action them
  stats           Show classification statistics
  correct         Tell the agent it made a mistake (feedback loop)
"""

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from taxonomy_agent.core.taxonomy_loader import TaxonomyLoader
from taxonomy_agent.core.embedder import TaxonomyEmbedder
from taxonomy_agent.core.classifier import FileClassifier, ClassificationTier
from taxonomy_agent.core.organiser import FileOrganiser, build_taxonomy_map
from taxonomy_agent.core.watcher import FolderWatcher

console = Console()

# ── Config defaults ──────────────────────────────────────────────────────────
DEFAULT_TAXONOMY = Path(__file__).parent.parent / "taxonomy" / "default_taxonomy.yaml"
DEFAULT_OUTPUT = Path.home() / "Organised"
DEFAULT_WATCH_DIRS = [
    Path.home() / "Downloads",
    Path.home() / "Desktop",
]


def get_components(taxonomy_path, output_dir, api_key=None, dry_run=True):
    """Initialise all components. Shared across commands."""
    taxonomy = TaxonomyLoader(taxonomy_path)
    embedder = TaxonomyEmbedder()
    taxonomy_map = build_taxonomy_map(taxonomy.get_all_nodes())
    organiser = FileOrganiser(
        root_output_dir=Path(output_dir),
        taxonomy_map=taxonomy_map,
        dry_run=dry_run,
    )
    classifier = FileClassifier(
        taxonomy=taxonomy,
        embedder=embedder,
        anthropic_api_key=api_key,
        dry_run=dry_run,
    )
    return taxonomy, embedder, classifier, organiser


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Taxonomy Agent — local-first AI file organiser."""
    pass


@cli.command()
@click.option("--taxonomy", default=str(DEFAULT_TAXONOMY), help="Path to taxonomy YAML")
def build_index(taxonomy):
    """Embed taxonomy nodes and build HNSW index. Run this once after editing taxonomy.yaml."""
    console.rule("[bold blue]Building HNSW Index")
    loader = TaxonomyLoader(taxonomy)
    nodes = loader.get_all_nodes()
    console.print(f"Loaded [bold]{len(nodes)}[/bold] taxonomy nodes")

    embedder = TaxonomyEmbedder()
    embedder.build_taxonomy_index(nodes)

    console.print(f"[green]✓[/green] Index built at: {embedder.index_path}")
    
    # Show taxonomy tree
    table = Table(title="Taxonomy Nodes", show_lines=True)
    table.add_column("Path", style="cyan")
    table.add_column("Auto-rules", style="yellow")
    table.add_column("Keywords")
    
    for node in sorted(nodes, key=lambda n: n.full_path):
        indent = "  " * node.depth
        rules = str(len(node.auto_rules)) if node.auto_rules else "-"
        kws = ", ".join(node.keywords[:4]) + ("..." if len(node.keywords) > 4 else "")
        table.add_row(f"{indent}{node.full_path}", rules, kws)
    
    console.print(table)


@cli.command()
@click.argument("folder", type=click.Path(exists=True))
@click.option("--taxonomy", default=str(DEFAULT_TAXONOMY))
@click.option("--output", default=str(DEFAULT_OUTPUT))
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY")
@click.option("--dry-run/--no-dry-run", default=True, help="Preview without moving files")
@click.option("--recursive/--no-recursive", default=False)
@click.option("--limit", default=0, help="Max files to process (0 = all)")
def scan(folder, taxonomy, output, api_key, dry_run, recursive, limit):
    """Classify all files in FOLDER."""
    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[red]LIVE[/red]"
    console.rule(f"[bold blue]Scanning {folder} {mode}")

    taxonomy_obj, embedder, classifier, organiser = get_components(
        taxonomy, output, api_key, dry_run
    )

    folder_path = Path(folder)
    glob = "**/*" if recursive else "*"
    files = [
        f for f in folder_path.glob(glob)
        if f.is_file() and not f.name.startswith(".")
    ]

    if limit:
        files = files[:limit]

    console.print(f"Found [bold]{len(files)}[/bold] files")

    table = Table(show_lines=True)
    table.add_column("File", style="cyan", max_width=35)
    table.add_column("→ Category", style="green", max_width=30)
    table.add_column("Conf", justify="right")
    table.add_column("Tier", style="dim")
    table.add_column("Review?", justify="center")

    tier_counts = {t: 0 for t in ClassificationTier}
    
    for file_path in files:
        result = classifier.classify(file_path)
        action = organiser.execute(result)
        tier_counts[result.tier] += 1

        review_flag = "[yellow]⚠[/yellow]" if result.needs_review else "[green]✓[/green]"
        tier_label = {
            ClassificationTier.AUTO_RULE: "[dim]rule[/dim]",
            ClassificationTier.EMBEDDING: "[blue]embed[/blue]",
            ClassificationTier.CLAUDE: "[magenta]claude[/magenta]",
        }[result.tier]

        table.add_row(
            file_path.name[:35],
            result.node_path,
            f"{result.confidence:.0%}",
            tier_label,
            review_flag,
        )

    console.print(table)

    # Summary
    console.print("\n[bold]Summary[/bold]")
    console.print(f"  Auto-rules:  {tier_counts[ClassificationTier.AUTO_RULE]}")
    console.print(f"  Embedding:   {tier_counts[ClassificationTier.EMBEDDING]}")
    console.print(f"  Claude:      {tier_counts[ClassificationTier.CLAUDE]}")

    if dry_run:
        console.print(
            "\n[yellow]Dry run complete. Run with --no-dry-run to move files.[/yellow]"
        )


@cli.command()
@click.option("--taxonomy", default=str(DEFAULT_TAXONOMY))
@click.option("--output", default=str(DEFAULT_OUTPUT))
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY")
@click.option(
    "--dirs",
    multiple=True,
    default=[str(d) for d in DEFAULT_WATCH_DIRS],
    help="Directories to watch",
)
def watch(taxonomy, output, api_key, dirs):
    """Watch folders for new files and classify them automatically."""
    console.rule("[bold blue]Starting Watcher")

    taxonomy_obj, embedder, classifier, organiser = get_components(
        taxonomy, output, api_key, dry_run=False
    )

    def on_new_file(file_path: Path):
        try:
            result = classifier.classify(file_path)
            action = organiser.execute(result)

            icon = "⚠" if result.needs_review else "✓"
            color = "yellow" if result.needs_review else "green"
            console.print(
                f"[{color}]{icon}[/{color}] {file_path.name} "
                f"→ [cyan]{result.node_path}[/cyan] "
                f"({result.confidence:.0%}, {result.tier.value}, "
                f"{result.latency_ms:.0f}ms)"
            )
        except Exception as e:
            console.print(f"[red]✗[/red] {file_path.name}: {e}")

    watcher = FolderWatcher(
        watch_dirs=[Path(d) for d in dirs],
        classify_callback=on_new_file,
    )
    watcher.start()


@cli.command()
@click.option("--taxonomy", default=str(DEFAULT_TAXONOMY))
@click.option("--output", default=str(DEFAULT_OUTPUT))
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY")
def review(taxonomy, output, api_key):
    """Show files queued for manual review."""
    console.rule("[bold blue]Review Queue")

    _, _, _, organiser = get_components(taxonomy, output, api_key, dry_run=False)
    queue = organiser.get_review_queue()

    if not queue:
        console.print("[green]No files pending review.[/green]")
        return

    console.print(f"[yellow]{len(queue)} files need review:[/yellow]\n")
    for i, record in enumerate(queue, 1):
        console.print(f"[bold]{i}.[/bold] {Path(record['source']).name}")
        console.print(f"   Category: {record['node_path']}")
        console.print(f"   Confidence: {record['confidence']:.0%}")
        console.print(f"   Reason: {record['reasoning']}\n")


@cli.command()
def stats():
    """Show classification statistics from audit log."""
    log_path = Path.home() / ".taxonomy_agent" / "audit.jsonl"
    organiser = FileOrganiser(
        root_output_dir=Path.home() / "Organised",
        taxonomy_map={},
        log_path=log_path,
    )
    s = organiser.get_stats()
    if not s:
        console.print("No data yet. Run a scan first.")
        return

    console.print(f"\n[bold]Classification Stats[/bold]")
    console.print(f"  Total files processed: {s['total_files']}")
    console.print(f"  By tier:")
    for tier, count in s["by_tier"].items():
        console.print(f"    {tier}: {count}")
    console.print(f"  Needs review: {s['needs_review']}")
    console.print(f"  User corrections: {s['corrections']}")
    console.print(f"  Avg latency: {s['avg_latency_ms']}ms")
    console.print(f"  Claude escalation rate: {s['claude_calls_pct']}%")


if __name__ == "__main__":
    cli()
