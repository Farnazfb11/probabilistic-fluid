# graph-summaries-for-medical-temporal-kg

## HiWi RDF Summarizer (FLUID-based)

This project implements **RDF graph summarization** based on the
[FLUID framework](https://github.com/t-blume/fluid-framework/tree/master) (Till Blume, 2020).
It parses **N-Quads** datasets (e.g., from the [Dynamic Linked Data Observatory](http://dyldo.org)),
builds a property graph with [NetworkX](https://networkx.org/), and applies different
**summarization strategies** (Attribute, Class, Complex Attribute+Class) to produce **FLUID-style summary graphs**.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage (CLI)](#usage-cli)
- [Output JSON Schema](#output-json-schema)
- [Strategies](#strategies)
- [Testing (Gold-Standard Mini-Graph + pytest)](#testing-gold-standard-mini-graph--pytest)
- [Comparison & Metrics (main_compare.py)](#comparison--metrics-main_comparepy)
- [Batch Processing (server / HPC)](#batch-processing-server--hpc)
- [Plotting & Visualization](#plotting--visualization)
- [Data Filtering & Preprocessing](#data-filtering--preprocessing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Authors & Acknowledgment](#authors--acknowledgment)
- [License](#license)
- [Project Status](#project-status)

---

## Features

- Parses N-Quads / N-Quads.GZ and constructs a **NetworkX MultiDiGraph** with optional literal nodes, numeric IDs, and reverse edges.  
- Handles literal values flexibly: collapse all literals into a single node (default) or preserve each literal as a distinct node.
- Exports **FLUID-style summaries** with:
  - **Schema:** `types`, `predicates`, `pred_to_dst_types`
  - **Payload:** `vcp` (class size), `dsp` (data sources / named graphs)
  - **(Optional)** membership map (raw node → supernode)
  - **Lightweight mode:** `only_id` (just `sid` and `vcp`)
- Provides a **one-shot metrics** tool to compare summary packages (avg size, std, edge counts, out-degree, TMH, components).

---

## Project Structure

**Core RDF Processing:**
- `RDFparser.py` — permissive N-Quads parser (handles gzip; reports unmatched lines).  
- `RDFGraphBuilder.py` — graph builder with flags: include literals, numeric IDs, reverse edges.  

**Summarization & Strategies:**
- `VertexSummary.py` — per-vertex FLUID summary (types, predicates, dst types).  
- `SchemaExtraction.py` — protocol for strategies; plug-in style.  
- `AttCollection.py` — **SEAttributeCollection** (predicate-only).  
- `ClassCollection.py` — **SEClassCollection** (class-only).  
- `CompAttClassCollection.py` — **SEComplexAttributeClassCollection** (types + predicates + dst types).  
- `Collector.py` — GraphX-style collector (send/merge messages).  
- `Summarizer.py` — grouping and superedge construction helpers.  

**I/O & Utilities:**
- `save.py` — `save_summary_for_strategy` / `load_summary_package`.  
- `Unary_measure.py` — single-summary metrics (EQC size, TMH, components, connectivity).  
- `Binary_measure.py` — pairwise comparison metrics (Jaccard, GED, KLD, EMD).  

**Analysis & Visualization:**
- `unary_plot.py` — **CLI**: plot unary metrics (EQC size, TMH, components) over time.
- `Binary_plot.py` — **CLI**: plot pairwise comparison metrics (Jaccard, GED, KLD, EMD).
- `kary_plot.py` — **CLI**: plot k-ary temporal metrics across multiple snapshots.
- `plot_CSV.py` — **CLI**: regenerate plots from metrics.csv (lightweight re-visualization).

**Batch Processing:**
- `main_build.py` — **CLI**: parse → build → summarize → save.  
- `main_compare.py` — **CLI**: compute one-shot summary metrics.  
- `parallel.sh` — batch process all weeks under a base directory.

**Data Filtering:**
- `save_filtered_quads.py` — **CLI**: batch filter N-Quads by host/domain across weeks.
- `filtering/build_domain_files_from_lod_json.py` — **CLI**: extract domain-specific datasets from LOD Cloud JSON.
- `filtering/extract_hosts.py` — **CLI**: list unique hosts in N-Quads data.
- `filtering/pick_portals.py` — **CLI**: filter life sciences datasets to BioPortal and Eagle-I portals.
- `filtering/bioportal_prefixes.py` — **CLI**: BioPortal ontology prefix utilities.  

---

## Requirements
- Python **3.9+**
- OS: Linux (server), Windows 10/11 (dev), macOS (dev) — tested internally
- Install runtime deps via: `pip install -r requirements.txt`
- (For tests) Dev deps via: `pip install -r requirements-dev.txt`

---

## Installation

Create a virtual environment with Python 3.9 (recommended on the server):

```bash
cd ~/graph-summaries-for-medical-temporal-kg
/usr/bin/python3.9 -m venv .venv39
```
Activate it:
```bash
# Linux/macOS
source .venv39/bin/activate

# Windows (PowerShell)
.\.venv39\Scripts\Activate.ps1
```
Install dependencies:
```bash
python -m pip install -r requirements.txt
```

---

## Quickstart

Summarize a single DyLDO file with default strategy (SchemEX) and 10k triples limit:

```bash
python main_build.py --input "data/2012-05-06/data-seedlist.nq.gz"
```

Expected console (abridged):

```csharp
[main] Parsed triples: ~10,000
[main] Graph built. Nodes=…  Edges=…
[main] Strategy: complexattributeclass
[save] wrote summary → summaries/2012-05-06_complexattributeclass.json
```


---

## Usage (CLI)

`main_build.py` accepts **one or more** input paths and creates a single summary for all inputs.

```bash
python main_build.py --input PATH [PATH ...] [options]
```

**Required**

All provided `--input` files are **merged into one property graph** before summarization (useful for `data.nq.gz` + `data-seedlist.nq.gz` of the same week).

* `--input, -i` One or more paths to `.nq` or `.nq.gz` files.

**Options**

* `--limit, -n N` Max triples to read across all inputs (default: 10000; `0` = all).
* `--include-literals` Include literal objects as their own nodes (default: off). When off, all literal values are collapsed into a single canonical `__LITERAL__` node (preserves edges, saves memory). When on, each literal value becomes its own node.
* `--use-numeric-ids` Hash node IRIs into numeric IDs to save memory (default: off; use flag to enable).
* `--reverse-edges` Add reverse edges for non-literal objects (default: off).
* `--strategy, -s {attribute,class,attributeclass,complexattributeclass}` Select summary strategy (default: `complexattributeclass`).
* `--out, -o PATH` Output **file** or **directory** (default: `summaries/`).

  * If a directory or ends with `/`, final name is `<WEEK>_<STRATEGY>.json`.
  * `<WEEK>` is inferred from any `YYYY-MM-DD` in input paths or parent folder name.
* `--include-membership` Include raw node → supernode membership map (large).
* `--only-id` Export lightweight package with only `sid` and `vcp`.
* `--no-edges` Do NOT include super-edges in the saved summary (default: edges included).
* `--load-test` Reload the saved JSON to smoke-test it.

**Examples**

Minimal run (complex strategy, 10k triples):

```bash
python main_build.py --input "data/2012-05-06/data-seedlist.nq.gz"
```

Attribute strategy with 50k triples and explicit output file:

```bash
python main_build.py --input "data/2012-05-06/data-seedlist.nq.gz" \
  --limit 50000 --strategy attribute \
  --out summaries/seedlist_attribute.json
```

Class strategy with literals + reverse edges + numeric IDs, multi-file input:

```bash
python main_build.py --input \
  "data/2012-05-06/data.nq.gz" \
  "data/2012-05-06/data-seedlist.nq.gz" \
  --strategy class --include-literals --reverse-edges --use-numeric-ids \
  --out summaries/
```
### Output naming rules
- If `-o` points to a **directory** (or ends with `/`): file is saved as `<WEEK>_<STRATEGY>.json` in that directory.
- `<WEEK>` is inferred from:
  1) any `YYYY-MM-DD` substring in the provided input paths (first match wins), else
  2) the parent directory name, else
  3) the input filename stem.

---

## Output JSON Schema

**Meta + Nodes (full export)**

```json
{
  "meta": {
    "strategy": "attributeclass",
    "only_id": false,
    "datasets": ["data.nq.gz","data-seedlist.nq.gz"],
    "include_literals": false,
    "use_numeric_ids": false,
    "add_reverse_edges": false,
    "limit": 10000,
    "include_membership": false
  },
  "nodes": {
    "139647895282323149896797688874731797517": {
      "sid": 139647895282323149896797688874731797517,
      "vcp": 1,
      "dsp": ["G3"],
      "types": [],
      "predicates": ["http://ex/name"],
      "pred_to_dst_types": {}
    }
  }
}
```

**Lightweight “only-id” variant**

```json
"nodes": {
  "197971857087053301556285727432889687635": {
    "sid": 197971857087053301556285727432889687635,
    "vcp": 4
  }
}
```

Notes:

* In full export, each supernode stores the signature defining fields (`types`, `predicates`, `pred_to_dst_types`) depending on strategy.

* In only id export, those signature fields are not saved, only `sid` and `vcp` are kept.

* Membership map (optional) adds a top-level `"membership": { "<label or id>": <sid>, ... }`.

| Mode      | Fields kept in `nodes[*]`                       | When to use                     |
|-----------|--------------------------------------------------|---------------------------------|
| Full      | `sid`, `vcp`, `dsp`, `types`, `predicates`, `pred_to_dst_types` | Offline analysis, schema checks |
| Only-ID   | `sid`, `vcp`                                    | Large-scale runs, quick stats   |

---

## Strategies

* **Attribute (`--strategy attribute`)**
  Groups by **outgoing predicates only** (predicate-set signature).
  Implementation: `SEAttributeCollection`. 

* **Class (`--strategy class`)**
  Groups by **RDF type set**.
  Implementation: `SEClassCollection`. 

* **Attribute + Class (`--strategy attributeclass`)**
  Groups by **types + outgoing predicates**.
  Implementation: `SEAttributeClassCollection`. 

* **SchemEX (`--strategy complexattributeclass`)**
  Groups by **types + outgoing predicates + per-predicate destination types**.
  Implementation: `SEComplexAttributeClassCollection`. 

---

## Testing (Gold-Standard Mini-Graph + pytest)

**Gold-standard mini-graph (tests/conftest.py):**

Subjects: `a(Person)`, `b(Person)`, `e(Person)`, `g(Org)`, `d(Person)`  
Objects: `x(Person)`, `y(Org)`, `n1(untyped)`

Edges:
```
a → x (knows),    a → y (worksAt)
b → y (knows),    b → y (worksAt)  [note: both to same object y]
e → n1 (name)
g → x (knows)
d (no outgoing edges)
```

Validates edge cases: identical predicate sets with different destination types (forces splits in `attributeclass`), empty signatures, untyped targets, class distinctions, etc.

**How to run:**

```bash
pip install -r requirements-dev.txt
pytest -vv                          # all tests
pytest -vv tests/test_minigraph_strategies.py  # just strategy tests
pytest -vv tests/test_rdfparser.py  # just parser tests
```

**Example test (attribute strategy grouping):**

The `test_attribute_predicate_only` test shows how the attribute-only strategy groups subjects by their predicate sets:

```python
# a and b both have {knows, worksAt} → same group
assert subj["a"] == subj["b"]

# g has only {knows} → different group
assert subj["g"] != subj["a"]

# d has empty predicate set → unique group
assert subj["d"] not in {subj["a"], subj["e"], subj["g"]}

# Result: 4 distinct equivalence classes: {a,b}, {e}, {g}, {d}
assert len(set(subj.values())) == 4
```

Test files cover: strategy grouping, RDF parsing (N-Triples/N-Quads/gzip), JSON save/load, and metrics computation.

**All metrics have test coverage:**
- Unary metrics (EQC size, TMH, components, etc.) → `test_unary_measure.py`
- Binary metrics (Jaccard, GED, KLD, EMD) → `test_binary_measure.py`

For detailed metric logic and expected behaviors, see the test implementations—they are self-documenting with inline assertions and comments.

---

## Comparison & Metrics (`main_compare.py`)

Compute one-shot metrics over a saved summary:

```bash
python main_compare.py --input "summaries/2012-05-06_attributeclass_onlyid.json"
```

It reports:

* `eqc_count`, `avg_eqc_size`, `std_eqc_size`
* `edge_count_unweighted`, `edge_count_weighted`
* `avg_out_degree_weighted`, `tmh_weighted`
* `components_weak`, `components_strong`
* sanity: number of sizes used

**Metric notes**
- `eqc_count`: number of equivalence classes (supernodes).
- `avg_eqc_size`, `std_eqc_size`: mean and std of class sizes (`vcp`).
- `edge_count_unweighted`: number of distinct superedges (ignores multiplicity).
- `edge_count_weighted`: total edge weight after roll-up (multiplicity).
- `avg_out_degree_weighted`: average weighted out-degree over supernodes.
- `tmh_weighted`: total multiplicity *highlights* (aggregate weight proxy).
- `components_weak` / `components_strong`: weakly/strongly connected components in the summary graph.


Optionally save to JSON:

```bash
python main_compare.py -i "..." --out-json results/metrics.json
```

(See implementation in `Unary_comp.py` and CLI wrapper in `main_compare.py`.)

---

## Batch Processing (server / HPC)

Use `parallel.sh` to iterate all weekly folders under `$BASE`, validate `.gz`, and run summaries in parallel:

```bash
# on server
source .venv/bin/activate
bash parallel.sh
```

* Skips already done weeks (`summaries/<WEEK>_attributeclass_onlyid.json` exists and non-empty).
* Detects **missing / corrupt** inputs and logs to `logs/_missing.log` and `logs/_corrupt.log`.
* Writes week logs to `logs/<WEEK>.log`.
* Adjust `PARALLEL` (default 8) to your RAM budget.

Logs and status:
- Per-week run logs: `logs/<WEEK>.log`
- Missing inputs: `logs/_missing.log`
- Corrupt inputs: `logs/_corrupt.log`


---

## Plotting & Visualization

After generating summary JSONs and metrics, use these tools to create publication-quality plots over time or between snapshots.

### Unary Metrics (single summary strategy)

Plot evolving metrics for a single summarization strategy across multiple time points:

```bash
python unary_plot.py \
  --input-dir summaries/ \
  --out-dir results/
```

**Generates:**
- **EQC (Equivalence Class) metrics:**
  - `avg_and_eqc_count.pdf` — equivalence class count and average size over time
  - `eqc_std.pdf` — standard deviation of equivalence class sizes
- **Tendency-to-Make-Hubs (TMH):**
  - `tmh_weighted.pdf` — weighted degree TMH trends
  - `tmh_unweighted.pdf` — unweighted degree TMH trends
  - `tmh_weighted_2015_2020.pdf` — weighted TMH restricted to 2015–2020
- **Connected Components analysis:**
  - `components_unfiltered.pdf` — weakly vs strongly connected components (all nodes)
  - `components_payload_filtered.pdf` — components after filtering nodes by minimum payload (vcp ≥ 40)
  - `components_weak_thresholded.pdf` — weakly connected components with node-count thresholds
  - `components_strong_thresholded.pdf` — strongly connected components with node-count thresholds
  - `components_no_isolated_out.pdf` — components after removing nodes with out-degree 0
- **Payload:**
  - `total_payload.pdf` — total payload (sum of vcp) over time

**Key options:**
- `--title-*` — customize individual plot titles
- `--no-avg-eqc`, `--no-std-eqc`, `--no-tmh-weighted`, etc. — skip specific plots
- `--max-xticks N` — control x-axis label density (default: 40)
- `--component-min-payload N` — minimum node payload for payload-filtered component plot (default: 40)
- `--csv FILE` — optionally export metrics to CSV for downstream processing

### Binary Metrics (comparing consecutive summaries)

Compare consecutive snapshots to measure how summary graphs change over time:

```bash
python Binary_plot.py \
  --input-dir summaries/ \
  --out-dir results_binary/
```

**Generates:**
- **Jaccard similarity (structural):**
  - `jaccard_vertices.pdf` — Jaccard distance over entity vertices
  - `jaccard_edges.pdf` — Jaccard distance over superedges
- **Graph structure:**
  - `ged_structural.pdf` — graph edit distance approximation
- **Distributions:**
  - `kld_symmetric.pdf` — symmetric KL divergence of EQC size distributions
- **Earth Mover's Distance (optional):**
  - `emd_degree.pdf` — EMD based on node degree distributions
  - `emd_signature.pdf` — EMD based on signature distributions
  - `emd_sid_degree.pdf` — EMD using exact SID with degree fallback

**Key options:**
- `--emd-mode {none, degree, signature, both, sid_degree}` — which EMD variants to compute (default: sid_degree)
- `--degree-mode {weighted_total, weighted_out, unweighted_out, log_weighted_out, log_weighted_total}` — degree proxy for EMD (default: log_weighted_out)
- `--no-jac-v`, `--no-jac-e`, `--no-ged`, `--no-kld`, `--no-emd` — skip specific metrics
- `--max-xticks N` — control x-axis label density (default: 30)
- `--csv FILE` — optionally export metrics to CSV for use with kary_plot
- `--from-csv FILE` — load pre-computed metrics from CSV instead of recomputing

### k-ary Temporal Trends (smoothing & long-term trends)

Create smoothed and trend curves from a CSV table of binary metrics across multiple snapshots:

```bash
python kary_plot.py \
  --input results_binary/metrics.csv \
  --score-cols jac_v jac_e emd_sid_degree \
  --out-dir results_kary/
```

Applies moving-window smoothing and trend fitting to stabilize metric visualization. Useful after `Binary_plot.py` exports metrics to CSV.

**Generates:**
- One PDF per metric with three curves:
  - Raw metric (original)
  - Softened curve (short moving average, default k=4)
  - Trend curve (long moving average, default k=12)

**Key options:**
- `--score-cols COL1 COL2 ...` — which CSV columns to process (default: jac_v jac_e emd)
- `--k-soft N` — window length for short-term smoothing (default: 4)
- `--k-trend N` — window length for long-term trend (default: 12)
- `--label-col COL` — CSV column with timestep labels, e.g. dates (default: label)
- `--out-dir DIR` — output directory for PDFs (default: results_kary)
- `--out-table FILE` — save augmented CSV with k-ary columns (default: results_kary/kary_metrics.csv)
- `--max-xticks N` — control x-axis label density (default: 30)

### Plot from CSV (lightweight re-visualization)

Regenerate plots from existing metrics CSV files without re-running expensive computations:

```bash
python plot_CSV.py \
  --input results/metrics.csv \
  --out-dir results_updated/
```

Automatically detects known metric columns and generates the same PDFs the original scripts would create. Useful for quick plot style updates or label changes without recomputation.

**Key options:**
- `--label-col COL` — CSV column for x-axis labels (e.g., dates)
- `--max-xticks N` — control x-axis label density (default: 30)
- `--no-extra` — skip plotting unrecognized CSV columns (default: off, plots them generically)

---

## Data Filtering & Preprocessing

Prepare life sciences datasets and filter large N-Quads collections for targeted analysis.

### Filter N-Quads by Domain

Extract and merge life sciences data from weekly DyLDO snapshots by host/domain filtering:

```bash
python save_filtered_quads.py \
  --base /path/to/weekly/data \
  --out-dir filtered_data/ \
  --log-dir logs/ \
  --from-week 2012-01-01 \
  --to-week 2012-12-31 \
  --portals filtering/life_science_portals.txt \
  --datasets filtering/life_sciences_datasets.txt
```

**Behavior:**
- Processes weekly folders (`YYYY-MM-DD/`) under `--base`
- For each week, **merges** `data.nq.gz` + `data-seedlist.nq.gz` into ONE output file
- Filters to matching hosts and writes: `--out-dir/<WEEK>_life_science_filtered.nq.gz`
- Logs missing inputs to `logs/_missing.log` and corrupt files to `logs/_corrupt.log`
- Per-week logs written to `logs/<WEEK>_filter.log`
- Validates `.gz` integrity and skips corrupted weeks entirely

**Configuration files:**
- `--portals` — text file with portal definitions (`id | title | url`)
- `--datasets` — text file with dataset IDs to include

### Extract Life Sciences Datasets from LOD Cloud

Build curated lists of life sciences datasets from LOD Cloud metadata:

```bash
# Step 1: Build domain-specific dataset and host lists
python filtering/build_domain_files_from_lod_json.py \
  --lod-json lod-data.json \
  --domain "Life Sciences" \
  --out-datasets life_sciences_datasets.txt \
  --out-hosts life_science_hosts.txt

# Step 2 (optional): Filter to specific portals
python filtering/pick_portals.py
```

The first script extracts datasets matching a domain from LOD Cloud JSON metadata.
The second filters to BioPortal and Eagle-I datasets (modifiable via `PORTAL_PREFIXES`).

### Extract Hosts from RDF Data

List unique hosts/domains appearing in an N-Quads file:

```bash
python filtering/extract_hosts.py \
  --input data.nq.gz \
  --out hosts.txt
```

Outputs one host per line (e.g., `dbpedia.org`, `example.com`).
Useful for analyzing data provenance or building filtering rules.

### BioPortal Prefix Extraction

Utility to maintain BioPortal ontology prefixes:

```bash
python filtering/bioportal_prefixes.py
```

Reads `life_sciences_datasets.json`, filters by BioPortal/Eagle-I prefixes, and exports to JSON/CSV/TXT formats.
Used internally by the portal filtering pipeline.

---

## Troubleshooting

* **“No match found …” during parsing**
  The parser prints unmatched lines and a final counter at the end; check data cleanliness or relax assumptions. 

* **Corrupt `.gz` files**
  Validate with `gzip -t file.gz`. The batch script already checks and logs corrupt files and continues others. 

* **Huge outputs**
  Use `--only-id` to reduce JSON size; skip membership unless required. 

---

## Roadmap

* JSON summary diff/similarity tools
* Visualization export (GraphML / PNG)
* Batch processing of all DyLDO snapshots (hardened retry & resume)

---

## Contributing
- Open an issue describing the change or bug.
- For PRs: add a short description, reference an issue, and include a minimal test if logic changes.
- Keep README examples runnable; avoid introducing flags not present in the CLI.

---

## Authors & Acknowledgment

**Farnaz Telki** (HiWi project work, University of Ulm)
**Prof. Ansgar Scherp** — supervision
Based on FLUID Framework by Till Blume (2020): t-blume/fluid-framework

---

## License

TBD (e.g., MIT or Apache-2.0)

---

## Project Status

Active — ongoing HiWi project for Linked Data summarization and analysis.

