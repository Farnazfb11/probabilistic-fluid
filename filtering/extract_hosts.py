from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Iterable, Set


def iter_urls(value: Any) -> Iterable[str]:
    """
    Walk any JSON-like structure and yield strings that look like URLs.
    """
    if value is None:
        return
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("http://") or s.startswith("https://"):
            yield s
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from iter_urls(v)
        return
    if isinstance(value, list):
        for v in value:
            yield from iter_urls(v)
        return


def host_from_url(url: str) -> str | None:
    try:
        p = urlparse(url)
        host = (p.hostname or "").strip().lower()
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return None


def main() -> None:
    # Change this if your file has a different name
    in_path = Path("life_science_datasets.json")

    if not in_path.exists():
        raise FileNotFoundError(
            f"Could not find {in_path}. Run your dataset filtering script first, or fix the path."
        )

    data = json.loads(in_path.read_text(encoding="utf-8"))

    # Your file might be:
    # 1) a list of dataset dicts
    # 2) a dict with a key like "datasets"
    if isinstance(data, dict):
        if "datasets" in data and isinstance(data["datasets"], list):
            datasets = data["datasets"]
        elif "life_science_datasets" in data and isinstance(data["life_science_datasets"], list):
            datasets = data["life_science_datasets"]
        else:
            # fallback: if it is lod-data.json style dict with entries, take values
            datasets = list(data.values())
    elif isinstance(data, list):
        datasets = data
    else:
        raise ValueError("Unsupported JSON structure in life_science_datasets.json")

    hosts: Set[str] = set()

    for ds in datasets:
        # scan all URL-like strings inside the dataset object
        for url in iter_urls(ds):
            h = host_from_url(url)
            if h:
                hosts.add(h)

    out_txt = Path("life_science_hosts.txt")
    out_json = Path("life_science_hosts.json")

    sorted_hosts = sorted(hosts)

    out_txt.write_text("\n".join(sorted_hosts) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(sorted_hosts, indent=2), encoding="utf-8")

    print(f"Datasets scanned: {len(datasets)}")
    print(f"Unique hosts found: {len(sorted_hosts)}")
    print(f"Saved: {out_txt}")
    print(f"Saved: {out_json}")
    print("\nFirst 20 hosts:")
    for h in sorted_hosts[:20]:
        print(" ", h)


if __name__ == "__main__":
    main()
