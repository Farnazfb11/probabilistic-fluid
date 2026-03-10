from __future__ import annotations

import argparse
import gzip
import os
import re
from pathlib import Path
from typing import Optional, Set, Tuple
from urllib.parse import urlparse
from datetime import datetime

from RDFparser import RDFParser

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PORTALS = REPO_ROOT / "filtering" / "life_science_portals.txt"
DEFAULT_DATASETS = REPO_ROOT / "filtering" / "life_sciences_datasets.txt"


# ---------- helpers ----------

class CorruptInputError(RuntimeError):
    """Raised when an input .gz is corrupt (even if it looks ok at the beginning)."""


def parse_hosts_from_txt(path: Path) -> Set[str]:
    hosts: Set[str] = set()

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        url = parts[2]
        if not url or url.lower() == "none":
            continue

        try:
            h = urlparse(url).netloc.lower()
            if h:
                hosts.add(h)
        except Exception:
            continue

    return hosts


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


def ctx_is_life_science(g: Optional[str], allowed_hosts: Set[str]) -> bool:
    h = host_from_term(g)
    return (h in allowed_hosts) if h else False


def _is_valid_gzip_head(path: Path) -> Tuple[bool, str]:
    """
    Fast check: try to read a small chunk.
    This can miss corruption that occurs later in the file.
    """
    try:
        with gzip.open(path, "rb") as f:
            f.read(1024 * 64)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _week_dirs(base: Path) -> list[Path]:
    out = []
    for p in base.iterdir():
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name):
            out.append(p)
    out.sort(key=lambda x: x.name)
    return out


def _in_range(name: str, a: str, b: str) -> bool:
    return a <= name <= b


# ---------- main per-week worker ----------

def filter_one_week(
    week_dir: Path,
    out_dir: Path,
    log_dir: Path,
    allowed_hosts: Set[str],
) -> None:
    week = week_dir.name
    d1 = week_dir / "data.nq.gz"
    d2 = week_dir / "data-seedlist.nq.gz"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{week}_life_science_filtered.nq.gz"
    tmp_out_path = out_dir / f".{week}_life_science_filtered.tmp.nq.gz"
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

    # skip if output already exists (optional but useful)
    if out_path.exists() and out_path.stat().st_size > 0:
        msg = f"[SKIP] output already exists: {out_path}"
        log(msg)
        print(f"[skip] {week} already exists", flush=True)  # NEW: stdout summary
        log(f"[END] time={datetime.now().isoformat(timespec='seconds')}")
        return

    if not d1.exists() or not d2.exists():
        msg = "[SKIP] missing one of the two inputs"
        log(msg)
        print(f"[skip] {week} missing input(s)", flush=True)  # NEW
        log(f"[END] time={datetime.now().isoformat(timespec='seconds')}")
        return

    # fast head check (cheap)
    ok1, r1 = _is_valid_gzip_head(d1)
    ok2, r2 = _is_valid_gzip_head(d2)
    if not ok1 or not ok2:
        msg = f"[SKIP] gzip head-check failed: data={ok1} ({r1}) seed={ok2} ({r2})"
        log(msg)
        print(f"[skip] {week} gzip head-check failed", flush=True)  # NEW
        log(f"[END] time={datetime.now().isoformat(timespec='seconds')}")
        return

    total = 0
    kept = 0
    no_graph = 0
    bad_line = 0

    total1 = kept1 = 0
    total2 = kept2 = 0

    # Always write to temp first; only rename when fully successful
    try:
        fout = gzip.open(tmp_out_path, "wt", encoding="utf-8")
    except Exception as e:
        msg = f"[FAIL] cannot open output gzip: {type(e).__name__}: {e}"
        log(msg)
        print(f"[fail] {week} cannot open output", flush=True)  # NEW
        log(f"[END] time={datetime.now().isoformat(timespec='seconds')}")
        return

    def process_file(in_path: Path) -> Tuple[int, int]:
        """
        If gzip is corrupt, RDFParser.parse() will eventually trigger a gzip/zlib error.
        We catch it and raise CorruptInputError so the week is SKIPPED cleanly.
        """
        nonlocal total, kept, no_graph, bad_line
        local_total = 0
        local_kept = 0

        parser = RDFParser(str(in_path))

        try:
            for s, p, o, g in parser.parse():
                local_total += 1
                total += 1

                if g is None:
                    no_graph += 1
                    continue

                if ctx_is_life_science(g, allowed_hosts):
                    fout.write(f"{s} {p} {o} {g} .\n")
                    kept += 1
                    local_kept += 1

        except Exception as e:
            raise CorruptInputError(f"{in_path.name}: {type(e).__name__}: {e}") from e

        return local_total, local_kept

    status = "OK"
    skip_reason = ""

    try:
        log("[RUN] filtering data.nq.gz")
        total1, kept1 = process_file(d1)

        log("[RUN] filtering data-seedlist.nq.gz")
        total2, kept2 = process_file(d2)

    except CorruptInputError as e:
        status = "SKIP"
        skip_reason = str(e)
        log(f"[SKIP] input gzip became unreadable during streaming: {skip_reason}")
        print(f"[skip] {week} corrupt gzip during read", flush=True)  # NEW

    except Exception as e:
        status = "FAIL"
        log(f"[FAIL] unexpected exception during filtering: {type(e).__name__}: {e}")
        print(f"[fail] {week} unexpected error", flush=True)  # NEW

    finally:
        try:
            fout.close()
        except Exception:
            pass

    # If we skipped or failed, remove temp output so you never keep partial week files
    if status != "OK":
        try:
            if tmp_out_path.exists():
                tmp_out_path.unlink()
        except Exception:
            pass

        log("[DONE] totals")
        log(f"[STATS] data_total={total1} data_kept={kept1}")
        log(f"[STATS] seed_total={total2} seed_kept={kept2}")
        log(f"[STATS] total_read={total} total_kept={kept}")
        log(f"[STATS] no_graph={no_graph} bad_line={bad_line}")
        log(f"[END] time={datetime.now().isoformat(timespec='seconds')}")
        return

    # OK case: rename temp -> final output
    try:
        os.replace(tmp_out_path, out_path)
        log(f"[OK] wrote output: {out_path}")
        print(f"[ok] {week} kept={kept} read={total}", flush=True)  # NEW: stdout summary
    except Exception as e:
        log(f"[FAIL] could not rename temp output: {type(e).__name__}: {e}")
        print(f"[fail] {week} rename output failed", flush=True)  # NEW
        try:
            if tmp_out_path.exists():
                tmp_out_path.unlink()
        except Exception:
            pass

    log("[DONE] totals")
    log(f"[STATS] data_total={total1} data_kept={kept1}")
    log(f"[STATS] seed_total={total2} seed_kept={kept2}")
    log(f"[STATS] total_read={total} total_kept={kept}")
    log(f"[STATS] no_graph={no_graph} bad_line={bad_line}")
    log(f"[END] time={datetime.now().isoformat(timespec='seconds')}")


# ---------- CLI ----------

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

    portal_hosts = parse_hosts_from_txt(portals_path)
    dataset_hosts = parse_hosts_from_txt(datasets_path)
    allowed_hosts = portal_hosts | dataset_hosts

    # optional manual additions
    allowed_hosts |= {
        "bioportal.bioontology.org",
        "data.bioontology.org",
    }

    weeks = _week_dirs(base)
    selected = [w for w in weeks if _in_range(w.name, args.from_week, args.to_week)]

    if not selected:
        print("[info] No week folders matched your range.")
        return

    print(f"[info] weeks selected: {len(selected)}", flush=True)
    for wdir in selected:
        print(f"[info] processing week {wdir.name}", flush=True)
        filter_one_week(wdir, out_dir, log_dir, allowed_hosts)


if __name__ == "__main__":
    main()