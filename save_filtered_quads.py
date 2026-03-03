from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Optional, Set, Tuple
from urllib.parse import urlparse
from datetime import datetime

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

    # keep these if you really need them
    if any(d.startswith("bioportal-") for d in dataset_ids) and "bioportal" in g_low:
        return True
    if any(d.startswith("eagle-i-") for d in dataset_ids) and "eagle-i" in g_low:
        return True
    if any(d.startswith("bio2rdf-") for d in dataset_ids) and "bio2rdf" in g_low:
        return True

    return False


def _is_valid_gzip(path: Path) -> Tuple[bool, str]:
    """
    Light gzip sanity check: try to read a small chunk.
    If file is badly corrupted, gzip.open will raise.
    """
    try:
        with gzip.open(path, "rb") as f:
            f.read(1024 * 64)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _week_dirs(base: Path) -> list[Path]:
    # week folders look like YYYY-MM-DD
    out = []
    for p in base.iterdir():
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name):
            out.append(p)
    out.sort(key=lambda x: x.name)
    return out


def _in_range(name: str, a: str, b: str) -> bool:
    return a <= name <= b


def filter_one_week(
    week_dir: Path,
    out_dir: Path,
    log_dir: Path,
    portal_ids: Set[str],
    dataset_ids: Set[str],
) -> None:
    week = week_dir.name
    d1 = week_dir / "data.nq.gz"
    d2 = week_dir / "data-seedlist.nq.gz"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{week}_life_science_filtered.nq.gz"
    log_path = log_dir / f"{week}_filter.log"

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    # start log fresh
    log_path.write_text("", encoding="utf-8")
    log(f"[START] week={week} time={datetime.now().isoformat(timespec='seconds')}")
    log(f"[PATH] week_dir={week_dir}")
    log(f"[PATH] data={d1}")
    log(f"[PATH] seed={d2}")
    log(f"[PATH] out={out_path}")

    if not d1.exists() or not d2.exists():
        log("[SKIP] missing one of the two inputs")
        return

    ok1, r1 = _is_valid_gzip(d1)
    ok2, r2 = _is_valid_gzip(d2)
    if not ok1 or not ok2:
        log(f"[SKIP] gzip check failed: data={ok1} ({r1}) seed={ok2} ({r2})")
        return

    total = 0
    kept = 0
    no_graph = 0
    bad_line = 0

    total1 = kept1 = 0
    total2 = kept2 = 0

    # write ONE output file per week, append both sources into it
    try:
        fout = gzip.open(out_path, "wt", encoding="utf-8")
    except Exception as e:
        log(f"[FAIL] cannot open output gzip: {type(e).__name__}: {e}")
        return

    try:
        # helper to process one file
        def process_file(in_path: Path) -> Tuple[int, int]:
            nonlocal total, kept, no_graph, bad_line
            local_total = 0
            local_kept = 0

            parser = RDFParser(str(in_path))
            for s, p, o, g in parser.parse():
                local_total += 1
                total += 1

                if g is None:
                    no_graph += 1
                    continue

                if ctx_is_life_science(g, dataset_ids, portal_ids):
                    # write N-Quads line
                    fout.write(f"{s} {p} {o} {g} .\n")
                    kept += 1
                    local_kept += 1

            return local_total, local_kept

        log("[RUN] filtering data.nq.gz")
        total1, kept1 = process_file(d1)
        log("[RUN] filtering data-seedlist.nq.gz")
        total2, kept2 = process_file(d2)

    except Exception as e:
        log(f"[FAIL] exception during filtering: {type(e).__name__}: {e}")
        # keep partial output (still useful for debugging)
    finally:
        fout.close()

    log("[DONE] totals")
    log(f"[STATS] data_total={total1} data_kept={kept1}")
    log(f"[STATS] seed_total={total2} seed_kept={kept2}")
    log(f"[STATS] total_read={total} total_kept={kept}")
    log(f"[STATS] no_graph={no_graph} bad_line={bad_line}")
    log(f"[END] time={datetime.now().isoformat(timespec='seconds')}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter DyLDO weekly data.nq.gz + data-seedlist.nq.gz into ONE gz output per week + log."
    )
    ap.add_argument("--base", required=True, help="Base folder containing week subfolders YYYY-MM-DD")
    ap.add_argument("--out-dir", required=True, help="Output directory for filtered week gz files")
    ap.add_argument("--log-dir", required=True, help="Output directory for per-week log files")
    ap.add_argument("--from-week", required=True, help="Start week YYYY-MM-DD (inclusive)")
    ap.add_argument("--to-week", required=True, help="End week YYYY-MM-DD (inclusive)")
    ap.add_argument("--portals", default=str(DEFAULT_PORTALS))
    ap.add_argument("--datasets", default=str(DEFAULT_DATASETS))

    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)

    portals_path = Path(args.portals)
    datasets_path = Path(args.datasets)

    if not base.exists():
        raise FileNotFoundError(f"Base folder not found: {base}")
    if not portals_path.exists():
        raise FileNotFoundError(f"Portals file not found: {portals_path}")
    if not datasets_path.exists():
        raise FileNotFoundError(f"Datasets file not found: {datasets_path}")

    portal_ids = parse_ids_from_txt(portals_path)
    dataset_ids = parse_ids_from_txt(datasets_path)

    weeks = _week_dirs(base)
    selected = [w for w in weeks if _in_range(w.name, args.from_week, args.to_week)]

    if not selected:
        print("[info] No week folders matched your range.")
        return

    print(f"[info] weeks selected: {len(selected)}")
    for wdir in selected:
        print(f"[info] processing week {wdir.name}")
        filter_one_week(wdir, out_dir, log_dir, portal_ids, dataset_ids)


if __name__ == "__main__":
    main()