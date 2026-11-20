#!/usr/bin/env python3
"""
run_all.py
==========

Run the whole experiment several times, each run into its own folder, and
build the numbers of the thesis from all of them.

    python run_all.py                         # 6 runs of all six datasets
    python run_all.py --only bsbm1000         # one dataset
    python run_all.py --dry-run               # print the commands only
    python run_all.py --post-only             # only the steps after the runs

What it does
------------
For run k = 1 .. --runs, and for each dataset in turn:

  run_summary.py   all 4 models x (deterministic + 3 families x m in 1,5,10,25),
                   with --recollect, so that every model and configuration is
                   timed and measured on its own (--shared collects once per
                   model instead, about 7 times faster). Run 1 also exports
                   the summary packages; they are identical in every run.

Every run_summary.py call is a separate process, so no run inherits memory
from an earlier one, and the datasets run one after another, never in
parallel, so that the timings are not disturbed.

Run 1 is the warm-up (--warmup): its timings are not used, but its class
counts and all other measures are checked against the other runs like every
run's. After the runs, once per dataset:

  measures.py fidelity   quantisation error and non-empty bins (run 1)
  checks.py b0           m = 1 reproduces the deterministic partition (run 1)
  measures.py unary      the unary measures of the run-1 packages
  tables.py numbers      consistency between runs, runtime and memory over the
                         measured runs (median, min-max, mean, sd), every
                         result-dependent number of the thesis, LaTeX tables

Layout
------
<results>/manifest.json            settings, code version, all steps and times
<results>/run_<k>/<dataset>/       measures_<dataset>.json, binnings_used_*.json,
                                   run_<dataset>.log, stdout.log
<results>/run_1/<dataset>/packages/   the summary packages
<results>/run_1/<dataset>/bin_fidelity.json, b0_check.txt
<results>/run_1/unary/             the unary measures
<results>/tables/                  everything tables.py numbers writes

A step whose output already exists is skipped, so after an interruption the
same command continues where it stopped (--redo runs everything again).

Code version
------------
The commit, the tags pointing at it and whether tracked files were modified
are recorded in the manifest. If tracked files have uncommitted changes the
script stops, since the results could then not be traced to a version;
--allow-dirty overrides this.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

DATASET_KEYS = ["bsbm1000", "bsbm100m", "watdiv10m", "dbpedia39",
                "dyldo_2015-10-11", "btc150m"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str):
    try:
        return subprocess.check_output(["git", *args], cwd=HERE,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def code_version() -> dict:
    modified = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "tags": (git("tag", "--points-at", "HEAD") or "").split() or [],
        "describe": git("describe", "--tags", "--always", "--dirty"),
        # compiled Python files change whenever the code runs; they are not code
        "modified_tracked_files": [
            line for line in (modified.splitlines() if modified else [])
            if "__pycache__" not in line and not line.rstrip().endswith(".pyc")],
    }


class Manifest:
    """<results>/manifest.json, rewritten after every step."""

    def __init__(self, path, settings: dict):
        self.path = path
        self.doc = {"settings": settings, "steps": []}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh)
            self.doc["steps"] = old.get("steps", [])
            self.doc["earlier_settings"] = old.get("earlier_settings", []) + \
                [old.get("settings")]
        self.save()

    def add(self, step: dict) -> None:
        self.doc["steps"].append(step)
        self.save()

    def save(self) -> None:
        if not self.path:          # --dry-run writes nothing
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.doc, fh, indent=2)
        os.replace(tmp, self.path)


def run_step(cmd, log_path: str, manifest: Manifest, info: dict,
             dry: bool, capture_to: str | None = None) -> bool:
    """Run one command, writing its output to log_path; True on success."""
    line = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[{now()}] {info.get('step')} {info.get('dataset', '')} "
          f"run {info.get('run', '')}\n  $ {line}", flush=True)
    if dry:
        return True
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.perf_counter()
    started = now()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {started}\n$ {line}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        captured = []
        for out_line in proc.stdout:
            log.write(out_line)
            log.flush()
            if capture_to:
                captured.append(out_line)
        rc = proc.wait()
    if capture_to and rc == 0:
        with open(capture_to, "w", encoding="utf-8") as fh:
            fh.writelines(captured)
    seconds = time.perf_counter() - t0
    manifest.add(dict(info, command=cmd, started=started, finished=now(),
                      seconds=round(seconds, 1), returncode=rc, log=log_path))
    status = "ok" if rc == 0 else f"FAILED (exit {rc}), see {log_path}"
    print(f"  {status} in {seconds / 60:.1f} min", flush=True)
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="~/probabilistic_graphs/datasets/out",
                    help="folder with <dataset>/graph_weighted.wnt and vertex_types.tsv")
    ap.add_argument("--results", default="~/probabilistic_graphs/results_mu",
                    help="folder to write all runs to")
    ap.add_argument("--only", nargs="+", default=None, metavar="DATASET",
                    help=f"datasets to run (default all: {' '.join(DATASET_KEYS)})")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=1,
                    help="runs 1..WARMUP are warm-up runs, not used for timing")
    ap.add_argument("--model", nargs="+", default=["AC", "CC", "ACC", "SchemEX"])
    ap.add_argument("--m", type=int, nargs="+", default=[1, 5, 10, 25])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shared", action="store_true",
                    help="collect once per model instead of once per "
                         "configuration (no --recollect): about 7 times faster, "
                         "but the collect step's time and memory are then the "
                         "same in every configuration of a model")
    ap.add_argument("--redo", action="store_true",
                    help="run steps again even if their output exists")
    ap.add_argument("--post-only", action="store_true",
                    help="skip the runs; only fidelity, b0, unary and numbers")
    ap.add_argument("--no-post", action="store_true",
                    help="only the runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run even if tracked files have uncommitted changes")
    args = ap.parse_args()

    data_root = os.path.expanduser(args.data_root)
    results = os.path.expanduser(args.results)
    datasets = args.only or DATASET_KEYS
    py = sys.executable

    version = code_version()
    if version["modified_tracked_files"] and not args.allow_dirty:
        print("Tracked files have uncommitted changes:\n  " +
              "\n  ".join(version["modified_tracked_files"]) +
              "\nCommit them first, so the results can be traced to a version, "
              "or pass --allow-dirty.")
        return 2

    missing = [d for d in datasets
               if not os.path.exists(os.path.join(data_root, d, "graph_weighted.wnt"))]
    if missing:
        print(f"No graph_weighted.wnt under {data_root} for: {', '.join(missing)}")
        return 2

    if not args.dry_run:
        os.makedirs(results, exist_ok=True)
    settings = {
        "started": now(), "host": socket.gethostname(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "code": version, "data_root": data_root, "results": results,
        "datasets": datasets, "runs": args.runs, "warmup": args.warmup,
        "models": args.model, "m": args.m, "seed": args.seed,
        "recollect": not args.shared,
        "argv": sys.argv,
    }
    manifest = Manifest(None if args.dry_run else
                        os.path.join(results, "manifest.json"), settings)
    print(f"code {version['describe']} (commit {version['commit']}), "
          f"results in {results}")
    print(f"{len(datasets)} datasets x {args.runs} runs, run 1"
          f"{'' if args.warmup == 1 else f'..{args.warmup}'} warm-up")

    failures = []
    t_start = time.perf_counter()

    # ------------------------------------------------------------- the runs
    if not args.post_only:
        for k in range(1, args.runs + 1):
            t_run = time.perf_counter()
            for ds in datasets:
                out = os.path.join(results, f"run_{k}", ds)
                done = os.path.join(out, f"measures_{ds}.json")
                if os.path.exists(done) and not args.redo:
                    print(f"[skip] run {k} {ds}: {done} exists")
                    continue
                graph = os.path.join(data_root, ds, "graph_weighted.wnt")
                types = os.path.join(data_root, ds, "vertex_types.tsv")
                cmd = [py, os.path.join(HERE, "run_summary.py"),
                       "--graph", graph, "--types", types,
                       "--deterministic", "--family", "all",
                       "--m", *map(str, args.m), "--model", *args.model,
                       "--seed", str(args.seed), "--out", out]
                if not args.shared:
                    cmd.append("--recollect")
                if k == 1:
                    cmd += ["--save-summaries", os.path.join(out, "packages")]
                ok = run_step(cmd, os.path.join(out, "stdout.log"), manifest,
                              {"step": "run_summary", "run": k, "dataset": ds,
                               "warmup": k <= args.warmup}, args.dry_run)
                if not ok:
                    failures.append(f"run {k} {ds}")
            if k == 1 and args.runs > 1 and not args.dry_run:
                took = time.perf_counter() - t_run
                print(f"\n[estimate] run 1 took {took / 3600:.1f} h; the remaining "
                      f"{args.runs - 1} runs will take about "
                      f"{(args.runs - 1) * took / 3600:.1f} h", flush=True)

    # ------------------------------------------------------------- after the runs
    if not args.no_post:
        run1 = os.path.join(results, "run_1")
        for ds in datasets:
            out = os.path.join(run1, ds)
            measures = os.path.join(out, f"measures_{ds}.json")
            graph = os.path.join(data_root, ds, "graph_weighted.wnt")
            types = os.path.join(data_root, ds, "vertex_types.tsv")
            if not os.path.exists(measures) and not args.dry_run:
                print(f"[skip] post steps for {ds}: no run-1 measures")
                continue
            fid = os.path.join(out, "bin_fidelity.json")
            if args.redo or not os.path.exists(fid):
                if not run_step([py, os.path.join(HERE, "measures.py"), "fidelity",
                                 "--graph", graph, "--measures", measures,
                                 "--out", fid, "--no-per-bin"],
                                os.path.join(out, "stdout.log"), manifest,
                                {"step": "fidelity", "dataset": ds, "run": 1},
                                args.dry_run):
                    failures.append(f"fidelity {ds}")
            b0 = os.path.join(out, "b0_check.txt")
            if args.redo or not os.path.exists(b0):
                if not run_step([py, os.path.join(HERE, "checks.py"), "b0", graph, types],
                                os.path.join(out, "stdout.log"), manifest,
                                {"step": "b0", "dataset": ds, "run": 1},
                                args.dry_run, capture_to=b0):
                    failures.append(f"b0 {ds}")
        packages = [os.path.join(run1, ds, "packages") for ds in datasets
                    if os.path.isdir(os.path.join(run1, ds, "packages")) or args.dry_run]
        if packages:
            if not run_step([py, os.path.join(HERE, "measures.py"), "unary",
                             "--packages", *packages,
                             "--out", os.path.join(run1, "unary")]
                            + (["--force"] if args.redo else []),
                            os.path.join(run1, "unary", "stdout.log"), manifest,
                            {"step": "unary", "run": 1}, args.dry_run):
                failures.append("unary")
        if not run_step([py, os.path.join(HERE, "tables.py"), "numbers",
                         "--results", results, "--warmup", str(args.warmup),
                         "--datasets-root", data_root],
                        os.path.join(results, "tables", "stdout.log"), manifest,
                        {"step": "numbers"}, args.dry_run):
            failures.append("numbers")

    total = time.perf_counter() - t_start
    print(f"\n[{now()}] finished in {total / 3600:.1f} h")
    if failures:
        print("FAILED: " + ", ".join(failures) +
              "\nRun the same command again to retry only these.")
        return 1
    print(f"Results: {results}/tables/thesis_numbers.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
