# CLAUDE.md

Context for Claude Code (or any Claude instance) working in this repository.
Read this before making changes — several design choices below look like
they could be "cleaned up" but are load-bearing for the thesis's fairness
claims. Breaking them silently would undermine the actual academic
contribution, not just the code.

## What this project is

Implementation for an M.Tech thesis: **cost-normalized benchmarking of
self-hosted vector databases (pgvector, Qdrant, FAISS) at small-to-mid
scale (50K/250K/1M vectors), producing a practitioner decision framework.**

- **Base paper:** Kuffo & Boncz (2025), *"Bang for the Buck: Vector Search
  on Cloud CPUs"* (DaMoN '25). They introduce a queries-per-dollar cost
  metric for raw CPUs/index libraries and explicitly name cross-database
  cost comparison as future work — this project is that follow-up, applied
  to full self-hosted database systems instead of raw hardware.
- **The gap being filled:** existing benchmarks are either large-scale +
  managed-cloud + not cost-aware (academic ANN-Benchmarks-style work), or
  cost-aware + managed-cloud-only + explicitly non-prescriptive (Zilliz's
  VectorDBBench Cost Leaderboard). Nobody benchmarks self-hosted systems at
  small scale on commodity hardware and translates it into an actual
  decision framework.
- **Evaluation workload:** MS MARCO passage retrieval (`BeIR/msmarco` on
  Hugging Face — verified structure: `corpus` split = `{_id, title, text}`
  passages, `queries` split = `{_id, title, text}` real search queries).

Full narrative context lives in the accompanying thesis proposal, IEEE
paper draft, and Phase II review PPT (not in this repo) — this file covers
only what's needed to work on the code correctly.

## Current status (as of last run)

| Component | Status |
|---|---|
| `data_prep.py` | Verified working — ran for real against live MS MARCO data |
| FAISS benchmarking | Verified working — ran for real, produced `results/raw_results.csv` |
| Cost analysis | Verified working — ran for real, produced `results/cost_summary.csv` |
| pgvector adapter | Written, syntax/import-checked, **not run against a live server** |
| Qdrant adapter | Written, syntax/import-checked, **not run against a live server** |
| Full 50K/250K/1M scale run | **Not done yet** — only a 500-passage smoke test has been run |
| `app.py` (Streamlit decision tool) | Written, not yet run (needs `results/cost_summary.csv` from a real multi-system run to be useful) |

**The immediate next task is almost always:** get pgvector and Qdrant
actually running via `docker compose up -d`, confirm both adapters work
against live servers, then run the full-scale benchmark. Don't assume
those two adapters are bug-free just because they're syntactically valid —
they've never been executed.

## Architecture — do not casually change these invariants

1. **Every system uses the same HNSW `M` and `ef_construction`** (from
   `config/config.yaml`, currently `M=16`, `ef_construction=200`). This is
   what makes the comparison fair. If you change how one system gets its
   parameters, change `config.yaml`, not a hardcoded value in one adapter.

2. **`build_index()` times the FULL path** from raw vectors to a queryable
   index — for pgvector that's table creation + bulk load + `CREATE INDEX`,
   not just the index statement. This matches what FAISS's single
   `index.add()` call does. If you "optimize" pgvector's timing to only
   measure `CREATE INDEX`, you'll make pgvector look artificially fast
   relative to FAISS — that's a real methodological bug, not a performance
   win.

3. **All three systems use inner product as the distance metric**
   (`vector_ip_ops` in pgvector, `Distance.DOT` in Qdrant,
   `METRIC_INNER_PRODUCT` in FAISS). Embeddings must stay normalized for
   this to behave like cosine similarity. Don't switch one system to
   cosine/L2 without switching all three — they'd stop ranking neighbors
   identically and recall comparisons would become meaningless.

4. **Ground truth is brute-force exact search, recomputed per scale**
   (`harness/ground_truth.py`), cached to `results/ground_truth_{scale}_k{k}.npy`.
   The correct top-10 for 50K passages is a different answer than for 1M
   passages (different corpus) — never reuse one scale's ground truth for
   another. Each cache has a `.meta.json` fingerprint (shapes + vector
   hashes); it's only reused when that matches the current `data/`, so
   re-running `data_prep.py` can't silently leave a stale answer key behind.
   Don't remove that check to "save time".

5. **Cost is computed at the cheapest `ef_search` that meets the target
   recall** (`cost.pick_operating_point`), not the fastest ef_search
   available. Reporting cost at the fastest setting would be reporting
   cost at whatever recall that happens to produce — not a fair, fixed
   comparison point.

6. **`VectorStore` (`harness/base.py`) is the contract every adapter
   implements identically.** The runner (`run_benchmark.py`) never
   branches on which system it's talking to beyond `build_store()`. If you
   add a fourth system (Milvus, etc.), implement the same abstract methods
   — don't special-case it in the runner.

## Known quirks (not bugs — don't "fix" these)

- **FAISS's `IndexHNSWFlat.hnsw` object has no `M` attribute** you can read
  back after construction (checked directly against the real FAISS API —
  it exposes `efConstruction`, `efSearch`, etc., but not `M`). That's why
  `FaissStore` stores `self.M` itself in `build_index()` rather than
  reading it back from the C++ object.
- **A `Fatal Python error: PyGILState_Release` message may appear** after
  any script using `datasets` streaming finishes and the process exits.
  This happens after your data is already saved — it's a benign
  interpreter-shutdown race in the `datasets`/`pyarrow` streaming
  internals, confirmed harmless. Don't spend time chasing it.
- **Qdrant has no persistent "set ef_search" setting** like pgvector's
  `SET hnsw.ef_search`. It's passed per-search-request instead
  (`search_params=SearchParams(hnsw_ef=...)`), so `QdrantStore.set_ef_search()`
  just stores the value for the next `search()` call rather than issuing
  any command immediately. This is correct, not a missing feature.
- **Memory reporting differs by system.** pgvector reports real
  `pg_relation_size`; Qdrant tries a real reported field and falls back to
  an estimate; FAISS has no memory API at all, so it's always an
  analytical estimate (vectors + HNSW graph links). This asymmetry is
  worth a line in the thesis's limitations section, not something to paper
  over by making them all "consistent" with fake precision.

## Setup on your machine

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-benchmark.txt   # app-only: requirements.txt
docker compose up -d          # pgvector + Qdrant, matched resource limits
```

Unlike the environment this was originally built in, your machine should
have a normal multi-core CPU and Docker — both were the actual constraints
during initial development (single core, no Docker), not anything wrong
with the approach. The full 1M-passage embedding pass is slow on a single
core; on a real multi-core instance it should be dramatically faster —
worth timing once and noting in the methodology section if you want a
concrete "why we didn't do this on a laptop" justification.

## Running things

```bash
python data_prep.py --smoke-test     # fast sanity check (~500 passages)
python data_prep.py                    # full run, sized by config.yaml

python run_benchmark.py --smoke-test    # matches smoke-test data
python run_benchmark.py                  # full sweep, all systems, all scales
python run_benchmark.py --systems pgvector qdrant   # subset

python cost_report.py                     # -> results/cost_summary.csv + printed recommendation
streamlit run app.py                        # panel-facing demo
```

## Code style for this project

- Plain, readable Python — moderate docstrings/comments are welcome here
  (unlike some other contexts, this is not a "no comments" codebase; the
  existing files use comments specifically to explain *why*, not what).
- Every config value (scales, HNSW params, pricing, connection info) goes
  in `config/config.yaml` — no hardcoded parameters scattered across
  adapter files.
- New systems/experiments should extend `VectorStore`, not bypass it.

## Where results feed back out

Once a full-scale run completes, the numbers belong in three places
outside this repo, all currently holding honest placeholders:

1. The IEEE paper draft's **Section IV (Results and Discussion)** — currently
   an explicit "pending" note.
2. The Phase II review PPT's **Implementation Status / Preliminary
   Results / Milestones slides** — currently marked "Not started / Planned."
3. The one-page thesis proposal, if its scope/timeline needs adjusting
   based on what the full run actually showed.

Don't regenerate those documents' full content from scratch — the
structure, formatting, and everything else in them is already finished;
only the results-dependent sections need real numbers dropped in.
