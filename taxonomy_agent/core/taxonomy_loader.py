"""
Loads taxonomy YAML and flattens it into nodes ready for embedding.
Each node gets a rich text representation that makes embedding similarity
more accurate than just using the label name.
"""

from __future__ import annotations
import yaml
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TaxonomyNode:
    id: str                          # e.g. "primitive_ai/research"
    name: str                        # e.g. "Research"
    full_path: str                   # e.g. "Primitive AI / Research"
    description: str                 # rich text for embedding
    keywords: list[str]
    auto_rules: list[dict]           # fast-path rules that bypass ML
    parent_id: Optional[str]
    children: list[str] = field(default_factory=list)
    depth: int = 0

    def embed_text(self) -> str:
        """
        The text we actually embed. Combining path + description + keywords
        gives the embedder much more signal than just the label name.
        This is the TELEClass enrichment idea in practice.
        """
        parts = [
            f"Category: {self.full_path}",
            f"Description: {self.description.strip()}",
        ]
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        return "\n".join(parts)

    def matches_auto_rule(self, filename: str, content_snippet: str = "") -> bool:
        """
        Fast-path: check filename patterns and content keywords
        before touching any ML. Zero latency.
        """
        for rule in self.auto_rules:
            pattern = rule.get("pattern", "")
            if pattern and fnmatch.fnmatch(filename.lower(), pattern.lower()):
                # If there's a content constraint too, check it
                contains_any = rule.get("contains_any", [])
                if not contains_any:
                    return True
                snippet_lower = content_snippet.lower()
                if any(kw.lower() in snippet_lower for kw in contains_any):
                    return True
        return False


class TaxonomyLoader:
    def __init__(self, yaml_path: str | Path):
        self.yaml_path = Path(yaml_path)
        self.nodes: dict[str, TaxonomyNode] = {}
        self._load()

    def _load(self):
        with open(self.yaml_path) as f:
            data = yaml.safe_load(f)
        
        for top_level in data["taxonomy"]:
            self._parse_node(top_level, parent_id=None, depth=0)

    def _parse_node(
        self,
        node_data: dict,
        parent_id: Optional[str],
        depth: int,
        parent_path: str = "",
    ) -> str:
        name = node_data["name"]
        slug = name.lower().replace(" ", "_").replace("/", "_")
        node_id = f"{parent_id}/{slug}" if parent_id else slug
        full_path = f"{parent_path} / {name}" if parent_path else name

        node = TaxonomyNode(
            id=node_id,
            name=name,
            full_path=full_path,
            description=node_data.get("description", name),
            keywords=node_data.get("keywords", []),
            auto_rules=node_data.get("auto_rules", []),
            parent_id=parent_id,
            depth=depth,
        )
        self.nodes[node_id] = node

        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node_id)

        for child_data in node_data.get("children", []):
            child_id = self._parse_node(
                child_data,
                parent_id=node_id,
                depth=depth + 1,
                parent_path=full_path,
            )
            node.children.append(child_id)

        return node_id

    def get_leaf_nodes(self) -> list[TaxonomyNode]:
        """Nodes with no children — most specific classification targets."""
        return [n for n in self.nodes.values() if not n.children]

    def get_all_nodes(self) -> list[TaxonomyNode]:
        return list(self.nodes.values())

    def get_node(self, node_id: str) -> Optional[TaxonomyNode]:
        return self.nodes.get(node_id)

    def get_path_to_root(self, node_id: str) -> list[TaxonomyNode]:
        """Walk from a leaf up to the root — used for H3Prompt top-down classification."""
        path = []
        current = self.nodes.get(node_id)
        while current:
            path.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        return list(reversed(path))

    def get_children(self, node_id: Optional[str] = None) -> list[TaxonomyNode]:
        """Get direct children of a node, or top-level nodes if node_id is None."""
        if node_id is None:
            return [n for n in self.nodes.values() if n.parent_id is None]
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[c] for c in node.children if c in self.nodes]
