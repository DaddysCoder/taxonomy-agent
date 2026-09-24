"""
The classification pipeline.

Three-tier decision system (fastest to slowest):

TIER 1 — Auto-rules (0ms)
    Pattern matching on filename/content.
    e.g. "*.py" → Primitive AI / Code
    No ML, no API. Instant.

TIER 2 — HNSW Embedding Search (5-50ms local)
    Embed file text, find nearest taxonomy node.
    Confidence >= HIGH_THRESHOLD → classify and move on.
    Confidence between LOW and HIGH → classify to parent level, queue for review.
    
TIER 3 — H3Prompt Claude Escalation (<1% of files)
    Confidence < LOW_THRESHOLD → send to Claude.
    Uses hierarchical 3-step prompting:
      Step 1: Which top-level category?
      Step 2: Which subcategory?
      Step 3: Confirm and explain.
    Only genuinely ambiguous files reach here.

This mirrors the TELEClass + Hierarchical Selective Classification
approach from the research: abstain at leaf level when uncertain,
output parent prediction, escalate hardest cases to stronger model.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from .taxonomy_loader import TaxonomyLoader, TaxonomyNode
from .embedder import TaxonomyEmbedder
from .file_reader import FileReader


# Confidence thresholds
HIGH_THRESHOLD = 0.82
LOW_THRESHOLD = 0.60


class ClassificationTier(Enum):
    AUTO_RULE = "auto_rule"
    EMBEDDING = "embedding"
    CLAUDE = "claude"


@dataclass
class ClassificationResult:
    file_path: str
    node_id: str
    node_path: str
    confidence: float
    tier: ClassificationTier
    is_certain: bool
    needs_review: bool
    reasoning: str
    latency_ms: float
    suggested_filename: Optional[str] = None


class FileClassifier:
    def __init__(
        self,
        taxonomy: TaxonomyLoader,
        embedder: TaxonomyEmbedder,
        anthropic_api_key: Optional[str] = None,
        dry_run: bool = True,
    ):
        self.taxonomy = taxonomy
        self.embedder = embedder
        self.reader = FileReader()
        self.api_key = anthropic_api_key
        self.dry_run = dry_run
        self._claude_client = None

    @property
    def claude(self):
        if self._claude_client is None:
            try:
                import anthropic
                self._claude_client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("anthropic package not installed: pip install anthropic")
        return self._claude_client

    def classify(self, file_path: Path) -> ClassificationResult:
        t0 = time.perf_counter()
        file_data = self.reader.extract(file_path)

        for node in self.taxonomy.get_all_nodes():
            if node.matches_auto_rule(file_data["filename"], file_data["snippet"]):
                latency = (time.perf_counter() - t0) * 1000
                return ClassificationResult(
                    file_path=str(file_path),
                    node_id=node.id,
                    node_path=node.full_path,
                    confidence=0.99,
                    tier=ClassificationTier.AUTO_RULE,
                    is_certain=True,
                    needs_review=False,
                    reasoning=f"Matched auto-rule for: {node.name}",
                    latency_ms=latency,
                )

        embed_text = file_data["text"]
        all_leaf_nodes = self.taxonomy.get_leaf_nodes()
        leaf_ids = [n.id for n in all_leaf_nodes]

        results = self.embedder.find_nearest(embed_text, k=3)

        if results:
            best_id, best_score = results[0]
            best_node = self.taxonomy.get_node(best_id)

            if best_score >= HIGH_THRESHOLD:
                latency = (time.perf_counter() - t0) * 1000
                return ClassificationResult(
                    file_path=str(file_path),
                    node_id=best_id,
                    node_path=best_node.full_path if best_node else best_id,
                    confidence=best_score,
                    tier=ClassificationTier.EMBEDDING,
                    is_certain=True,
                    needs_review=False,
                    reasoning=f"Embedding similarity: {best_score:.3f}",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

            elif best_score >= LOW_THRESHOLD:
                parent_id = best_node.parent_id if best_node else None
                if parent_id:
                    parent_node = self.taxonomy.get_node(parent_id)
                    latency = (time.perf_counter() - t0) * 1000
                    return ClassificationResult(
                        file_path=str(file_path),
                        node_id=parent_id,
                        node_path=parent_node.full_path if parent_node else parent_id,
                        confidence=best_score,
                        tier=ClassificationTier.EMBEDDING,
                        is_certain=False,
                        needs_review=True,
                        reasoning=(
                            f"Best match: {best_node.full_path if best_node else best_id} "
                            f"(score: {best_score:.3f}) — classified to parent level pending review. "
                            f"Runner-up: {results[1][0] if len(results) > 1 else 'none'}"
                        ),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        return self._classify_with_claude(file_path, file_data, t0)

    def _classify_with_claude(
        self, file_path: Path, file_data: dict, t0: float
    ) -> ClassificationResult:
        if not self.api_key:
            results = self.embedder.find_nearest(file_data["text"], k=1)
            if results:
                best_id, best_score = results[0]
                node = self.taxonomy.get_node(best_id)
                return ClassificationResult(
                    file_path=str(file_path),
                    node_id=best_id,
                    node_path=node.full_path if node else best_id,
                    confidence=best_score,
                    tier=ClassificationTier.EMBEDDING,
                    is_certain=False,
                    needs_review=True,
                    reasoning="Low confidence embedding result (no API key for Claude escalation)",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        file_context = (
            f"Filename: {file_data['filename']}\n"
            f"Extension: {file_data['extension']}\n"
            f"Content preview:\n{file_data['text'][:800]}"
        )

        try:
            top_nodes = self.taxonomy.get_children(None)
            top_options = "\n".join(
                f"- {n.name}: {n.description[:100].strip()}"
                for n in top_nodes
            )

            step1_response = self.claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Classify this file into ONE top-level category.\n\n"
                        f"FILE:\n{file_context}\n\n"
                        f"CATEGORIES:\n{top_options}\n\n"
                        f"Reply with ONLY the exact category name, nothing else."
                    )
                }]
            )
            top_choice = step1_response.content[0].text.strip()
            top_node = next(
                (n for n in top_nodes if n.name.lower() == top_choice.lower()),
                top_nodes[0]
            )

            children = self.taxonomy.get_children(top_node.id)
            if not children:
                return ClassificationResult(
                    file_path=str(file_path),
                    node_id=top_node.id,
                    node_path=top_node.full_path,
                    confidence=0.75,
                    tier=ClassificationTier.CLAUDE,
                    is_certain=True,
                    needs_review=False,
                    reasoning=f"Claude H3Prompt: {top_node.name} (no subcategories)",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

            child_options = "\n".join(
                f"- {n.name}: {n.description[:100].strip()}"
                for n in children
            )

            step2_response = self.claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": (
                        f"This file belongs in '{top_node.name}'. "
                        f"Now pick the most specific subcategory.\n\n"
                        f"FILE:\n{file_context}\n\n"
                        f"SUBCATEGORIES:\n{child_options}\n\n"
                        f"Reply with ONLY the exact subcategory name, nothing else."
                    )
                }]
            )
            sub_choice = step2_response.content[0].text.strip()
            sub_node = next(
                (n for n in children if n.name.lower() == sub_choice.lower()),
                children[0]
            )

            step3_response = self.claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": (
                        f"FILE: {file_context}\n\n"
                        f"PROPOSED CATEGORY: {sub_node.full_path}\n\n"
                        f"1. Is this correct? (yes/no + brief reason)\n"
                        f"2. Suggest a cleaner filename if the current one is unclear "
                        f"(format: YYYY-MM-DD_descriptive-name.ext or keep original)\n\n"
                        f"Reply in this exact format:\n"
                        f"CORRECT: yes\n"
                        f"REASON: <one sentence>\n"
                        f"FILENAME: <suggested name or KEEP>"
                    )
                }]
            )

            step3_text = step3_response.content[0].text.strip()
            lines = {
                line.split(":")[0].strip(): ":".join(line.split(":")[1:]).strip()
                for line in step3_text.splitlines()
                if ":" in line
            }

            is_correct = lines.get("CORRECT", "yes").lower().startswith("y")
            reason = lines.get("REASON", "")
            filename_suggestion = lines.get("FILENAME", "KEEP")
            suggested_filename = (
                None if filename_suggestion == "KEEP" else filename_suggestion
            )

            final_node = sub_node if is_correct else top_node

            return ClassificationResult(
                file_path=str(file_path),
                node_id=final_node.id,
                node_path=final_node.full_path,
                confidence=0.90 if is_correct else 0.65,
                tier=ClassificationTier.CLAUDE,
                is_certain=is_correct,
                needs_review=not is_correct,
                reasoning=f"Claude H3Prompt: {reason}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                suggested_filename=suggested_filename,
            )

        except Exception as e:
            results = self.embedder.find_nearest(file_data["text"], k=1)
            best_id = results[0][0] if results else list(self.taxonomy.nodes.keys())[0]
            best_score = results[0][1] if results else 0.0
            node = self.taxonomy.get_node(best_id)
            return ClassificationResult(
                file_path=str(file_path),
                node_id=best_id,
                node_path=node.full_path if node else best_id,
                confidence=best_score,
                tier=ClassificationTier.EMBEDDING,
                is_certain=False,
                needs_review=True,
                reasoning=f"Claude call failed ({e}), embedding fallback",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
