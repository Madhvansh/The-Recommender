#!/usr/bin/env python3
"""Download raw Amazon Reviews interaction files.

Fetches the per-category 5-core / ratings files from the Amazon Reviews 2023
collection (McAuley Lab) into ``data/raw/``.  These files are not committed; run
this once before training on real data.

Examples
--------
    python scripts/download_data.py --category beauty
    python scripts/download_data.py --category toys --out data/raw

If your environment blocks outbound network access, train on a ``synthetic*``
dataset instead (no download required).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Amazon Reviews 2023 ratings-only CSVs (item,user,rating,timestamp).
BASE = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories"
CATEGORIES = {
    "beauty": "All_Beauty.csv",
    "toys": "Toys_and_Games.csv",
    "sports": "Sports_and_Outdoors.csv",
    "games": "Video_Games.csv",
    "office": "Office_Products.csv",
}


def download(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}\n        -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done >> 20} / {total >> 20} MiB ({pct:5.1f}%)", end="")
    print("\ndone.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", choices=sorted(CATEGORIES), default="beauty")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    fname = CATEGORIES[args.category]
    dest = Path(args.out) / f"amazon-{args.category}.csv"
    url = f"{BASE}/{fname}"
    try:
        download(url, dest)
    except Exception as exc:  # network blocked, 404, etc.
        print(f"\nERROR: download failed ({exc}).", file=sys.stderr)
        print(
            "If outbound network is blocked, train on a 'synthetic' dataset "
            "instead (see configs/).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
