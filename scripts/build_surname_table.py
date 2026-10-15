"""Build static/surname_race.json from the US Census 2010 surnames file.

Source: https://www2.census.gov/topics/genealogy/2010surnames/names.zip
(Download once to /tmp/surnames.zip, then run: python scripts/build_surname_table.py)

Output maps NORMALIZED surname parts to Census race percentages, limited to
surnames appearing in this project's roster. Values are statistical estimates
about US bearers of a surname, NOT facts about any individual student.
"""
import csv
import json
import os
import re
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = [
    ("white", "pctwhite"),
    ("black", "pctblack"),
    ("asian", "pctapi"),
    ("native", "pctaian"),
    ("multi", "pct2prace"),
    ("hispanic", "pcthispanic"),
]


def norm(s):
    return re.sub(r"[^A-Z]", "", (s or "").upper())


def parts_of(last):
    return [p for p in re.split(r"[^A-Za-z]+", last or "") if p]


def pct(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def main():
    import sys
    zip_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/surnames.zip"
    roster = json.load(open(os.path.join(BASE, "roster_rows.json"), encoding="utf-8"))
    wanted = set()
    for r in roster:
        for part in parts_of(r.get("last", "")):
            if len(part) >= 2:
                wanted.add(norm(part))

    z = zipfile.ZipFile(zip_path)
    csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
    table = {}
    with z.open(csv_name) as fh:
        text = fh.read().decode("utf-8", "ignore").splitlines()
    for row in csv.DictReader(text):
        name = (row.get("name") or "").strip().upper()
        if name in wanted and name not in table:
            table[name] = {label: pct(row.get(col)) for label, col in COLS}

    out_path = os.path.join(BASE, "static", "surname_race.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "US Census Bureau, 2010 surnames (percentages of US bearers per surname)",
                "note": "Statistical estimates only, not facts about any individual.",
                "surnames": table,
            },
            f,
        )

    matched_parts = len(table)
    students_covered = 0
    for r in roster:
        ps = [norm(p) for p in parts_of(r.get("last", "")) if len(p) >= 2]
        if any(p in table for p in ps):
            students_covered += 1
    print(f"surname parts in table: {matched_parts} / {len(wanted)}")
    print(f"students with at least one known part: {students_covered} / {len(roster)}")
    missing = sorted(wanted - set(table))[:20]
    print(f"unmatched sample: {missing}")


if __name__ == "__main__":
    main()
