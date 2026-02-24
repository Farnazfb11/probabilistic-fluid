import json
import csv

def main():
    with open("lod-data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    life_science = []

    for dataset_id, dataset in data.items():
        domain = (dataset.get("domain") or "").strip().lower()
        if domain == "life_sciences":
            entry = {
                "id": dataset_id,
                "title": dataset.get("title"),
                "website": dataset.get("website"),
                "sparql": dataset.get("sparql"),
                "triples": dataset.get("triples"),
            }
            life_science.append(entry)

    print(f"\nFound {len(life_science)} life sciences datasets.\n")

    # Print first 20
    for ds in life_science[:20]:
        print(f"{ds['id']} | {ds['title']} | {ds['website']}")

    # ---------- SAVE JSON ----------
    with open("life_sciences_datasets.json", "w", encoding="utf-8") as f:
        json.dump(life_science, f, indent=2)

    # ---------- SAVE CSV ----------
    with open("life_sciences_datasets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "title", "website", "sparql", "triples"]
        )
        writer.writeheader()
        writer.writerows(life_science)

    # ---------- SAVE TXT ----------
    with open("life_sciences_datasets.txt", "w", encoding="utf-8") as f:
        for ds in life_science:
            f.write(f"{ds['id']} | {ds['title']} | {ds['website']}\n")

    print("\nSaved files in current folder.")


if __name__ == "__main__":
    main()
