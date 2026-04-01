from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _lower_set(values: Iterable[Any]) -> Set[str]:
    out: Set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip().lower()
        if s:
            out.add(s)
    return out


def _extract_domains(rec: Dict[str, Any]) -> Set[str]:
    # LOD Cloud metadata has historically used "domain".
    # Be tolerant and also check a few near variants.
    vals: List[Any] = []
    for key in ("domain", "domains", "subcloud", "subclouds", "category", "categories"):
        if key in rec:
            vals.extend(_as_list(rec.get(key)))
    return _lower_set(vals)


def _candidate_urls(rec: Dict[str, Any]) -> List[str]:
    urls: List[str] = []

    # Keep the most useful human-facing / dataset-facing URLs first.
    for key in ("website", "example", "sparql", "full_download", "other_download", "identifier"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip() and val.strip().lower() != "none":
            urls.append(val.strip())

    # Some fields may be nested lists or dicts
    links = rec.get("links")
    if isinstance(links, list):
        for item in links:
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        urls.append(v.strip())

    # Deduplicate, preserve order
    seen: Set[str] = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _pick_best_url(rec: Dict[str, Any]) -> Optional[str]:
    urls = _candidate_urls(rec)
    for u in urls:
        try:
            parsed = urlparse(u)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return u
        except Exception:
            pass
    return None


def _host_of(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        h = urlparse(url).netloc.lower()
        return h if h else None
    except Exception:
        return None


def _iter_records(data: Any) -> Iterable[Dict[str, Any]]:
    # Be tolerant to possible shapes:
    # 1) list of dataset dicts
    # 2) {"datasets": [...]}
    # 3) dict keyed by dataset id -> dataset dict
    if isinstance(data, list):
        for rec in data:
            if isinstance(rec, dict):
                yield rec
        return

    if isinstance(data, dict):
        if isinstance(data.get("datasets"), list):
            for rec in data["datasets"]:
                if isinstance(rec, dict):
                    yield rec
            return

        # keyed dict
        for k, v in data.items():
            if isinstance(v, dict):
                rec = dict(v)
                rec.setdefault("_id", k)
                yield rec


def build_domain_files(
    lod_json: Path,
    domain_name: str,
    out_datasets: Path,
    out_hosts: Path,
) -> Tuple[int, int]:
    domain_target = domain_name.strip().lower()

    with lod_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_lines: List[str] = []
    hosts: Set[str] = set()

    for rec in _iter_records(data):
        domains = _extract_domains(rec)
        if domain_target not in domains:
            continue

        ds_id = str(rec.get("_id") or rec.get("identifier") or "").strip()
        title = str(rec.get("title") or ds_id or "").strip()
        url = _pick_best_url(rec)

        # Keep line format consistent with your existing files
        dataset_lines.append(f"{ds_id} | {title} | {url if url else 'None'}")

        host = _host_of(url)
        if host:
            hosts.add(host)

    out_datasets.parent.mkdir(parents=True, exist_ok=True)
    out_hosts.parent.mkdir(parents=True, exist_ok=True)

    out_datasets.write_text("\n".join(dataset_lines) + ("\n" if dataset_lines else ""), encoding="utf-8")
    out_hosts.write_text("\n".join(sorted(hosts)) + ("\n" if hosts else ""), encoding="utf-8")

    return len(dataset_lines), len(hosts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build domain-specific dataset and host files from LOD Cloud lod-data.json")
    ap.add_argument("--lod-json", required=True, help="Path to downloaded lod-data.json")
    ap.add_argument("--domain", required=True, help="Domain name, e.g. 'Life Sciences' or 'Media'")
    ap.add_argument("--out-datasets", required=True, help="Output txt file in 'id | title | url' format")
    ap.add_argument("--out-hosts", required=True, help="Output txt file with one host per line")
    args = ap.parse_args()

    n_ds, n_hosts = build_domain_files(
        lod_json=Path(args.lod_json),
        domain_name=args.domain,
        out_datasets=Path(args.out_datasets),
        out_hosts=Path(args.out_hosts),
    )

    print(f"[ok] datasets={n_ds} hosts={n_hosts}")
    print(f"[ok] wrote datasets -> {args.out_datasets}")
    print(f"[ok] wrote hosts    -> {args.out_hosts}")


if __name__ == "__main__":
    main()