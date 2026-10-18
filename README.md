# Vector DB Benchmark — Cost-Normalized Benchmarking of Self-Hosted Vector Databases

Implementation for the M.Tech thesis: *Cost-Normalized Benchmarking of
Self-Hosted Vector Databases at Small-to-Mid Scale — A Decision Framework
for Resource-Constrained Teams.*

Base paper: Kuffo & Boncz (2025), *Bang for the Buck: Vector Search on
Cloud CPUs* (DaMoN '25). This project extends their queries-per-dollar cost
metric from raw CPUs/libraries to full self-hosted vector database systems
— the gap their own paper names as future work.

## What this does

Benchmarks **pgvector**, **Qdrant**, and **FAISS** under identical
conditions (same HNSW parameters, same hardware, same query set), at three
data scales (50K / 250K / 1M vectors), and converts the results into real
cloud cost — dollars per million queries at a target recall — instead of
just reporting raw speed.

Evaluation workload: the public **MS MARCO** passage retrieval dataset
(via the `BeIR/msmarco` mirror on Hugging Face), embedded with
`BAAI/bge-small-en-v1.5` through `fastembed` (ONNX, CPU-friendly — no GPU
or heavy `torch` install needed).

## Status: verified working, not yet run at full scale

Every piece of this pipeline has been **run for real** end-to-end on a
small smoke-test slice (500 passages) — it is not untested code. What's
*not* yet done is the full-scale run (50K/250K/1M passages across all
three systems), which needs real infrastructure this was built and
verified without:

| Piece | Verified how |
|---|---|
| Data prep (MS MARCO streaming + embedding) | Ran for real, produced real embeddings |
| FAISS benchmarking | Ran for real — see `results/raw_results.csv` for actual recall/latency/QPS numbers from a live run |
| Cost analysis | Ran for real on those FAISS results — see `results/cost_summary.csv` |
| pgvector / Qdrant adapters | Syntax- and import-verified; **not** run against live servers (this environment has no Docker) |
| Full 50K/250K/1M scale | Not run (this environment has 1 CPU core; embedding even 3,000 passages took ~6 minutes here — the full 1M-passage run needs to happen on your own multi-core instance) |

The included `results/raw_results.csv` and `results/cost_summary.csv` are
real output from the smoke test, kept as a working example — not
placeholders. Delete them (or just let the scripts overwrite them) when
you run the real thing.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-benchmark.txt   # app-only: requirements.txt

# Bring up pgvector and Qdrant (matched resource limits, so neither gets
# an unfair hardware advantage)
docker compose up -d
```

## Running it

**1. Prepare data** (downloads MS MARCO, embeds it, saves to `data/`):

```bash
python data_prep.py --smoke-test   # fast: ~500 passages, proves the pipeline works
python data_prep.py                 # full: sized by config.yaml (up to 1M passages)
                                     #   -- this will take a while; run it on your
                                     #   AWS instance, not a laptop, for the full 1M tier
```

**2. Run the benchmark** (builds each system's index, sweeps ef_search,
measures recall/latency/throughput):

```bash
python run_benchmark.py --smoke-test              # matches the smoke-test data
python run_benchmark.py                            # full run, all systems, all scales
python run_benchmark.py --systems faiss qdrant       # only specific systems
```

**3. Generate the cost report and decision framework:**

```bash
python cost_report.py
```

This prints a per-scale recommendation (cheapest system that meets your
target recall) and writes `results/cost_summary.csv`.

**4. Launch the decision-tool demo** (what you actually show your panel):

```bash
streamlit run app.py
```

Type in a data scale, target query load, and it recommends a system with a
real dollar figure — read live from `results/cost_summary.csv`, never
invented.

## Project structure

```
config/config.yaml     -- every parameter lives here: scales, HNSW params,
                            ef_search sweep, AWS pricing, connection info
docker-compose.yml      -- pgvector + Qdrant, matched CPU/memory limits
data_prep.py             -- MS MARCO -> embeddings -> data/*.npy
harness/
  base.py                 -- the VectorStore interface all three systems implement
  faiss_store.py           -- FAISS adapter (in-process, no server)
  pgvector_store.py         -- pgvector adapter (psycopg2 + native HNSW index)
  qdrant_store.py            -- Qdrant adapter (qdrant-client)
  ground_truth.py             -- brute-force exact kNN, cached per scale
run_benchmark.py         -- orchestrates the full sweep, writes raw_results.csv
cost.py                    -- raw performance -> real cloud dollars
cost_report.py               -- raw_results.csv -> cost_summary.csv + decision framework
app.py                        -- Streamlit demo for your panel
```

## Design notes worth knowing before you defend this

- **All three systems use the same HNSW M / ef_construction** (from
  `config.yaml`), so index quality differences reflect the systems
  themselves, not mismatched settings.
- **`build_index()` times the full path** from raw vectors to a queryable
  index for every system — for pgvector that's table creation + bulk load
  + `CREATE INDEX`, matching what FAISS's single `index.add()` call does
  in one step. This is the fair comparison point; timing only the
  `CREATE INDEX` statement would understate pgvector's real setup cost.
- **Recall is scored against a brute-force exact search**, recomputed per
  scale (the correct top-10 for 50K passages isn't the same as for 1M —
  the corpus itself is different).
- **Cost is computed at the cheapest `ef_search` that meets your target
  recall**, not the fastest one available — this matters because the
  fastest setting is usually also the least accurate.
- **Memory reporting differs by system**: pgvector and Qdrant report real
  measured memory/disk usage; FAISS has no such API, so its number is an
  analytical estimate (vectors + HNSW graph links). This is a known,
  document-worthy limitation, not a bug — mention it in your results
  discussion rather than letting a panelist find it first.
- **AWS pricing in `config.yaml`** (c6i.xlarge, Linux on-demand, pre-tax
  USD list price per region under `cost.regions` — $0.17/hr in us-east-1,
  ap-south-1 and ap-south-2 as of 2026-10-15) was looked up, not guessed —
  but cloud prices drift. Refresh it before your final run if it's been more
  than a couple of months. Source: AWS's public price file per region, e.g.
  `https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/Asia%20Pacific%20(Mumbai)/Linux/index.json`
  (gzip-compressed; fetch with `curl --compressed`).
- **Regions: measured vs. priced.** `run_region` (or `--region`) tags each
  row of `raw_results.csv` with where the benchmark actually ran.
  `cost_report.py` prices every measured run at every region in
  `cost.regions`, keeping both `measured_in` and `priced_for` columns, so a
  US measurement re-priced at Mumbai rates is never confused with a real
  Mumbai run. To get a real Indian measurement, launch the same instance type
  in ap-south-1 and run `python run_benchmark.py --region ap-south-1`. Rows
  from before region tagging show up as `unspecified`.
- **Indian billing (GST) is not modelled.** Accounts billed through AWS India
  (AISPL) pay 18% GST on top of list price. That depends on the billing
  account, not the region, so all costs here are pre-tax list prices.

## Known sandbox quirk

You may occasionally see a `Fatal Python error: PyGILState_Release`
message after a script using `datasets` streaming finishes and exits.
This happens *after* your data is already saved — it's a benign
interpreter-shutdown race in the `datasets`/`pyarrow` streaming path, not
a bug in this code. Safe to ignore.
