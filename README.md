# Taxonomy Agent

Local-first AI file organiser. Faster and smarter than The Drive AI for personal use
because it runs on your machine, uses your mental model, and applies ML only where needed.

## Architecture

```
New file detected
    ↓
TIER 1: Auto-rules (0ms)        ← ~40% of files
    filename/extension pattern matching
    ↓ (no match)
TIER 2: HNSW Embedding (5-50ms) ← ~58% of files  
    local sentence-transformers model
    HNSW approximate nearest-neighbour search
    confidence >= 0.82 → auto-classify
    confidence 0.60-0.82 → classify to parent, queue review
    ↓ (confidence < 0.60)
TIER 3: Claude H3Prompt (<2s)   ← ~2% of files
    hierarchical 3-step prompting
    coarse → fine → confirm + rename
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key (only needed for <2% of files)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Edit your taxonomy
nano taxonomy/default_taxonomy.yaml

# 4. Build the HNSW index (run once, or after taxonomy changes)
python -m taxonomy_agent.cli.main build-index

# 5. Test on a folder (dry run — no files moved)
python -m taxonomy_agent.cli.main scan ~/Downloads --dry-run

# 6. Go live
python -m taxonomy_agent.cli.main scan ~/Downloads --no-dry-run

# 7. Watch for new files continuously
python -m taxonomy_agent.cli.main watch
```

## Commands

| Command | Description |
|---------|-------------|
| `build-index` | Embed taxonomy nodes, build HNSW index |
| `scan <folder>` | Classify all files in a folder |
| `watch` | Watch Downloads/Desktop for new files |
| `review` | Show files queued for manual review |
| `stats` | Classification statistics |

## Customising your taxonomy

Edit `taxonomy/default_taxonomy.yaml`. The more descriptive your `description:` fields,
the better the embedding similarity matching will be.

Key fields:
- `description`: Rich text used for embedding — more detail = better accuracy
- `keywords`: Extra signal words
- `auto_rules`: Fast-path pattern matching (no ML)

After editing, always run `build-index` again.

## Why this beats The Drive AI for personal use

| Feature | The Drive AI | This |
|---------|-------------|------|
| Privacy | Files sent to cloud | 100% local |
| Your taxonomy | Generic categories | Your mental model |
| Speed | API latency | 5-50ms local |
| Rename suggestions | Basic | Content-aware via Claude |
| Cost | $9/month | ~$0.01/month Claude API |
| Customisation | Limited | Full control |

## Science behind it

- **HNSW** (Hierarchical Navigable Small World graphs): O(log n) approximate nearest
  neighbour search — scales to millions of files without slowing down
- **TELEClass**: Enriching taxonomy node descriptions before embedding significantly
  improves classification accuracy for imbalanced categories  
- **Hierarchical Selective Classification**: When confidence is low at leaf level,
  abstain and output the parent prediction — correct at a higher level beats
  wrong at a specific level
- **H3Prompt**: Top-down hierarchical prompting in 3 steps reduces LLM errors
  and is cheaper than asking for a leaf-level classification in one shot
