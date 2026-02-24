import json
import csv

PORTAL_PREFIXES = [
    "bioportal-",
    "eagle-i-",
]

def main():
    with open("life_sciences_datasets.json", "r", encoding="utf-8") as f:
        ds = json.load(f)

    portals = []
    for x in ds:
        _id = (x.get("id") or "").lower()
        if any(_id.startswith(p) for p in PORTAL_PREFIXES):
            portals.append(x)

    # Save JSON
    with open("life_science_portals.json", "w", encoding="utf-8") as f:
        json.dump(portals, f, indent=2, ensure_ascii=False)

    # Save CSV
    with open("life_science_portals.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "homepage"])
        for x in portals:
            writer.writerow([
                x.get("id"),
                x.get("title"),
                x.get("homepage"),
            ])

    # Save TXT
    with open("life_science_portals.txt", "w", encoding="utf-8") as f:
        for x in portals:
            f.write(f"{x.get('id')} | {x.get('title')} | {x.get('homepage')}\n")

    print("Portals found:", len(portals))
    print("Saved life_science_portals.json / .csv / .txt")

if __name__ == "__main__":
    main()
