"""
Handles the physical movement and renaming of files.

Key design decisions:
- NEVER deletes anything
- Always moves to a new destination, never overwrites
- Maintains a full audit log of every action
- Dry-run mode by default — shows what WOULD happen without doing it
- Correction tracking: when you move a file it placed wrong,
  it logs that as a correction to improve future classifications
"""

from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .classifier import ClassificationResult


class FileOrganiser:
    def __init__(
        self,
        root_output_dir: Path,
        taxonomy_map: dict[str, str],  # node_id → relative folder path
        dry_run: bool = True,
        log_path: Optional[Path] = None,
    ):
        self.root_output_dir = Path(root_output_dir)
        self.taxonomy_map = taxonomy_map
        self.dry_run = dry_run
        self.log_path = log_path or Path.home() / ".taxonomy_agent" / "audit.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.corrections: list[dict] = []  # user corrections for feedback loop

    def node_to_folder(self, node_id: str) -> Path:
        """Convert a taxonomy node ID to a filesystem folder path."""
        # node_id like "primitive_ai/research" → "Primitive AI/Research"
        # We use the taxonomy_map if available, otherwise derive from ID
        if node_id in self.taxonomy_map:
            rel_path = self.taxonomy_map[node_id]
        else:
            # Fallback: convert ID back to readable path
            parts = node_id.split("/")
            rel_path = "/".join(p.replace("_", " ").title() for p in parts)
        return self.root_output_dir / rel_path

    def execute(self, result: ClassificationResult) -> dict:
        """
        Move (or in dry-run, preview) a file based on its classification.
        Returns an action record.
        """
        src = Path(result.file_path)
        dest_dir = self.node_to_folder(result.node_id)

        # If review needed, put in a _review subfolder
        if result.needs_review:
            dest_dir = dest_dir / "_needs_review"

        # Handle filename suggestion
        dest_filename = result.suggested_filename or src.name
        dest = dest_dir / dest_filename

        # Handle duplicates — never overwrite
        if not self.dry_run and dest.exists():
            stem = Path(dest_filename).stem
            suffix = Path(dest_filename).suffix
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        action = {
            "timestamp": datetime.now().isoformat(),
            "source": str(src),
            "destination": str(dest),
            "node_id": result.node_id,
            "node_path": result.node_path,
            "confidence": round(result.confidence, 4),
            "tier": result.tier.value,
            "is_certain": result.is_certain,
            "needs_review": result.needs_review,
            "reasoning": result.reasoning,
            "latency_ms": round(result.latency_ms, 1),
            "dry_run": self.dry_run,
            "renamed": dest_filename != src.name,
            "status": "pending",
        }

        if self.dry_run:
            action["status"] = "dry_run"
        else:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                action["status"] = "moved"
            except Exception as e:
                action["status"] = f"error: {e}"

        self._log(action)
        return action

    def record_correction(
        self, file_path: str, wrong_node_id: str, correct_node_id: str
    ):
        """
        User tells us we got it wrong.
        This is stored and used to:
        1. Fine-tune the confidence thresholds over time
        2. Update the taxonomy enrichment descriptions
        3. (Future) Few-shot examples in Claude prompts
        """
        correction = {
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "wrong_node_id": wrong_node_id,
            "correct_node_id": correct_node_id,
        }
        self.corrections.append(correction)
        self._log({**correction, "type": "correction"})

    def get_review_queue(self) -> list[dict]:
        """Return all files pending review from the audit log."""
        if not self.log_path.exists():
            return []
        queue = []
        with open(self.log_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("needs_review") and record.get("status") == "moved":
                        queue.append(record)
                except json.JSONDecodeError:
                    continue
        return queue

    def get_stats(self) -> dict:
        """Summary statistics from audit log."""
        if not self.log_path.exists():
            return {}
        
        records = []
        with open(self.log_path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        actions = [r for r in records if r.get("type") != "correction"]
        corrections = [r for r in records if r.get("type") == "correction"]
        
        tier_counts = {}
        for r in actions:
            t = r.get("tier", "unknown")
            tier_counts[t] = tier_counts.get(t, 0) + 1

        avg_latency = (
            sum(r.get("latency_ms", 0) for r in actions) / len(actions)
            if actions else 0
        )

        return {
            "total_files": len(actions),
            "by_tier": tier_counts,
            "needs_review": sum(1 for r in actions if r.get("needs_review")),
            "corrections": len(corrections),
            "avg_latency_ms": round(avg_latency, 1),
            "claude_calls_pct": round(
                tier_counts.get("claude", 0) / max(len(actions), 1) * 100, 1
            ),
        }

    def _log(self, record: dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def build_taxonomy_map(taxonomy_nodes: list) -> dict[str, str]:
    """
    Build a mapping from node_id to filesystem path.
    Uses the node's full_path for readable folder names.
    """
    mapping = {}
    for node in taxonomy_nodes:
        # "Primitive AI / Research" → "Primitive AI/Research"
        folder_path = node.full_path.replace(" / ", "/")
        mapping[node.id] = folder_path
    return mapping
