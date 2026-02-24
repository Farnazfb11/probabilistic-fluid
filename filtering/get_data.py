import json
from pathlib import Path

data = json.loads(Path("lod-data.json").read_text())

if isinstance(data, dict):
    first_key = next(iter(data))
    print("Example dataset id:", first_key)
    print("Fields:", list(data[first_key].keys()))
elif isinstance(data, list):
    print("Fields:", list(data[0].keys()))
