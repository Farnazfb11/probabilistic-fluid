from __future__ import annotations
import os
import json
from pathlib import Path
import requests

API_BASE = "https://data.bioontology.org"

def get_apikey() -> str:
    key = os.environ.get("BIOPORTAL_APIKEY")
    if not key:
        raise SystemExit("Set BIOPORTAL_APIKEY first.")
    return key

def main():
    apikey = get_apikey()

    r = requests.get(
        f"{API_BASE}/ontologies",
        params={"apikey": apikey, "pagesize": 1000}
    )
    r.raise_for_status()
    ontologies = r.json()

    acronyms = sorted([
        o.get("acronym")
        for o in ontologies
        if o.get("acronym")
    ])

    Path("bioportal_acronyms.json").write_text(
        json.dumps(acronyms, indent=2),
        encoding="utf-8"
    )

    Path("bioportal_acronyms.txt").write_text(
        "\n".join(acronyms),
        encoding="utf-8"
    )

    print(f"Ontologies found: {len(acronyms)}")
    print("Saved bioportal_acronyms.json and .txt")

if __name__ == "__main__":
    main()
