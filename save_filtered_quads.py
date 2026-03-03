from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urlparse

from RDFparser import RDFParser

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PORTALS = REPO_ROOT / "filtering" / "life_science_portals.txt"
DEFAULT_DATASETS = REPO_ROOT / "filtering" / "life_sciences_datasets.txt"


def parse_ids_from_txt(path: Path) -> Set[str]:
    ids: Set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[0]:
            ids.add(parts[0])
    return ids


def week_from_path(p: Path) -> str:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", str(p))
    return m.group(1) if m else "unknown-week"


def host_from_term(term: Optional[str]) -> Optional[str]:
    if not term:
        return None
    t = term.strip()
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1]
    try:
        h = urlparse(t).netloc.lower()
        return h if h else None
    except Exception:
        return None


def ctx_is_life_science(g: Optional[str], dataset_ids: Set[str], portal_ids: Set[str]) -> bool:
    h = host_from_term(g) or ""
    g_low = (g or "").lower()

    if "bioportal" in h or "bioontology" in h:
        return True
    if "eagle-i" in g_low or "eagle-i" in h:
        return True
    if "bio2rdf" in g_low or "bio2rdf" in h:
        return True
    if "wiwiss.fu-berlin.de" in h:
        return True
    if "pubchem.ncbi.nlm.nih.gov" in h or "uniprot" in h:
        return True

    if any(d.startswith("bioportal-") for d in dataset_ids) and "bioportal" in g_low:
        return True
    if any(d.startswith("eagle-i-") for d in dataset_ids) and "eagle-i" in g_low:
        return True
    if any(d.startswith("bio2rdf-") for d in dataset_ids) and "bio2rdf" in g_low:
        return True

    return False


def _quad_to_line(s: str, p: str, o: str, g: Optional[str]) -> str:
    gg = g if g is not None else "None"
    return f"{s} {p} {o} {gg} ."


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(line: str, logf) -> None:
    msg = f"[{_now()}] {line}"
    print(msg)
    logf.write(msg + "\n")
    logf.flush()


def gzip_is_ok(path: Path) -> bool:
    """
    Fast integrity check using gzip -t, returns True if OK.
    """
    try:
        r = subprocess.run(
            ["gzip", "-t", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def _filter_one_input_into_open_stream(
    in_path: Path,
    fout,
    f_ignored,
    f_passed,
    *,
    dataset_ids: Set[str],
    portal_ids: Set[str],
    week: str,
    sample_n: int,
    counters: dict,
    bucket: str,  # "data" or "seed"
) -> None:
    parser = RDFParser(str(in_path))

    for s, p, o, g in parser.parse():
        counters[bucket]["total"] += 1
        counters["all"]["total"] += 1

        passed = False
        reason = ""

        if g is None:
            counters[bucket]["no_graph"] += 1
            counters["all"]["no_graph"] += 1
            passed = False
            reason = "no_graph"
        else:
            if ctx_is_life_science(g, dataset_ids, portal_ids):
                passed = True
            else:
                passed = False
                reason = "not_life_science"

        line = _quad_to_line(s, p, o, g)

        if passed:
            if counters["passed_sample_written"] < sample_n:
                f_passed.write(line + "\n")
                counters["passed_sample_written"] += 1

            fout.write(f"{s} {p} {o} {g} .\n")
            counters[bucket]["kept"] += 1
            counters["all"]["kept"] += 1
        else:
            if counters["ignored_sample_written"] < sample_n:
                f_ignored.write(f"[{reason}] {line}\n")
                counters["ignored_sample_written"] += 1

        if counters["all"]["total"] % 5_000_000 == 0:
            print(
                f"[{week}] progress read={counters['all']['total']:,} kept={counters['all']['kept']:,} "
                f"ignored_sample={counters['ignored_sample_written']}/{sample_n} "
                f"passed_sample={counters['passed_sample_written']}/{sample_n}"
            )


def process_week_folder(
    week_dir: Path,
    out_dir: Path,
    log_dir: Path,
    *,
    portals_path: Path,
    datasets_path: Path,
    sample_n: int,
) -> None:
    week = week_dir.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{week}_filter.log"

    t_start = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as logf:
        _log(f"week={week}", logf)
        _log(f"folder={week_dir}", logf)

        d1 = week_dir / "data.nq.gz"
        d2 = week_dir / "data-seedlist.nq.gz"

        if not d1.exists() or not d2.exists():
            _log("SKIP missing data.nq.gz or data-seedlist.nq.gz", logf)
            return

        _log(f"input_data={d1}", logf)
        _log(f"input_seed={d2}", logf)

        _log("check gzip integrity...", logf)
        ok1 = gzip_is_ok(d1)
        ok2 = gzip_is_ok(d2)
        _log(f"gzip_ok data={ok1} seed={ok2}", logf)

        if not ok1:
            _log(f"SKIP broken gzip: {d1}", logf)
            return
        if not ok2:
            _log(f"SKIP broken gzip: {d2}", logf)
            return

        if not portals_path.exists():
            _log(f"ERROR portals file not found: {portals_path}", logf)
            raise FileNotFoundError(f"Portals file not found: {portals_path}")
        if not datasets_path.exists():
            _log(f"ERROR datasets file not found: {datasets_path}", logf)
            raise FileNotFoundError(f"Datasets file not found: {datasets_path}")

        portal_ids = parse_ids_from_txt(portals_path)
        dataset_ids = parse_ids_from_txt(datasets_path)

        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{week}_lifescience_filtered.nq.gz"
        ignored_sample_path = out_dir / f"{week}_ignored_sample_{sample_n}.txt"
        passed_sample_path = out_dir / f"{week}_passed_sample_{sample_n}.txt"

        counters = {
            "data": {"total": 0, "kept": 0, "no_graph": 0},
            "seed": {"total": 0, "kept": 0, "no_graph": 0},
            "all": {"total": 0, "kept": 0, "no_graph": 0},
            "ignored_sample_written": 0,
            "passed_sample_written": 0,
        }

        _log(f"out={out_path}", logf)
        _log(f"samples_ignored={ignored_sample_path}", logf)
        _log(f"samples_passed={passed_sample_path}", logf)

        with (
            gzip.open(out_path, "wt", encoding="utf-8") as fout,
            ignored_sample_path.open("w", encoding="utf-8") as f_ignored,
            passed_sample_path.open("w", encoding="utf-8") as f_passed,
        ):
            _log("start filtering data.nq.gz", logf)
            _filter_one_input_into_open_stream(
                d1,
                fout,
                f_ignored,
                f_passed,
                dataset_ids=dataset_ids,
                portal_ids=portal_ids,
                week=week,
                sample_n=sample_n,
                counters=counters,
                bucket="data",
            )

            _log("start filtering data-seedlist.nq.gz", logf)
            _filter_one_input_into_open_stream(
                d2,
                fout,
                f_ignored,
                f_passed,
                dataset_ids=dataset_ids,
                portal_ids=portal_ids,
                week=week,
                sample_n=sample_n,
                counters=counters,
                bucket="seed",
            )

        t_end = time.perf_counter()
        secs = t_end - t_start

        _log("done", logf)
        _log(
            f"stats_data total={counters['data']['total']} kept={counters['data']['kept']} no_graph={counters['data']['no_graph']}",
            logf,
        )
        _log(
            f"stats_seed total={counters['seed']['total']} kept={counters['seed']['kept']} no_graph={counters['seed']['no_graph']}",
            logf,
        )
        _log(
            f"stats_all  total={counters['all']['total']} kept={counters['all']['kept']} no_graph={counters['all']['no_graph']}",
            logf,
        )
        ratio = (counters["all"]["kept"] / counters["all"]["total"]) if counters["all"]["total"] else 0.0
        _log(f"kept_ratio={ratio:.6f}", logf)
        _log(f"runtime_seconds={secs:.2f}", logf)


def parse_week_name(p: Path) -> Optional[str]:
    m = re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)
    return p.name if m else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter DyLDO weekly folders: read data.nq.gz and data-seedlist.nq.gz if both exist and are valid, "
                    "and write a single combined filtered .nq.gz per week with per-week logs."
    )
    ap.add_argument("--base", required=True, help="Base directory containing week folders YYYY-MM-DD")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-week combined gz files")
    ap.add_argument("--log-dir", required=True, help="Output directory for per-week log files")

    ap.add_argument("--portals", default=str(DEFAULT_PORTALS))
    ap.add_argument("--datasets", default=str(DEFAULT_DATASETS))
    ap.add_argument("--sample-n", type=int, default=200)

    ap.add_argument("--from-week", default=None, help="Optional start week YYYY-MM-DD (inclusive)")
    ap.add_argument("--to-week", default=None, help="Optional end week YYYY-MM-DD (inclusive)")

    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)

    portals_path = Path(args.portals)
    datasets_path = Path(args.datasets)

    if not base.exists():
        raise FileNotFoundError(f"Base folder not found: {base}")

    week_dirs = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        w = parse_week_name(p)
        if w is None:
            continue
        week_dirs.append(p)

    week_dirs.sort(key=lambda p: p.name)

    if args.from_week:
        week_dirs = [p for p in week_dirs if p.name >= args.from_week]
    if args.to_week:
        week_dirs = [p for p in week_dirs if p.name <= args.to_week]

    if not week_dirs:
        print("[info] No week folders found after filtering.")
        return

    print(f"[info] weeks={len(week_dirs)} base={base}")
    print(f"[info] out_dir={out_dir}")
    print(f"[info] log_dir={log_dir}")

    for wd in week_dirs:
        process_week_folder(
            wd,
            out_dir,
            log_dir,
            portals_path=portals_path,
            datasets_path=datasets_path,
            sample_n=args.sample_n,
        )


if __name__ == "__main__":
    main()