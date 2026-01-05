#!/usr/bin/env python3
"""
Danbooru Tag CSV to JSON Converter

Downloads the latest Danbooru tag list from DraconicDragon's archive
and converts it to JSON format optimized for tag autocomplete.

Usage:
    python scripts/convert_tags.py [--min-count MIN] [--output OUTPUT]

Arguments:
    --min-count MIN    Minimum post count threshold (default: 100)
    --output OUTPUT    Output JSON file path (default: data/tags.json)
"""

import csv
import json
import urllib.request
from pathlib import Path
import argparse


# Source CSV URL
CSV_URL = "https://raw.githubusercontent.com/DraconicDragon/dbr-e621-lists-archive/main/tag-lists/danbooru/danbooru_2026-01-01_pt20-ia-dd.csv"

# Category mapping (Danbooru category codes)
CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta"
}


def download_csv(url: str) -> list[str]:
    """Download CSV file from URL and return lines"""
    print(f"Downloading CSV from {url}...")
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')
    print(f"Downloaded {len(content)} bytes")
    return content.strip().split('\n')


def parse_csv_line(line: str) -> dict | None:
    """Parse a CSV line into tag data"""
    reader = csv.reader([line])
    row = next(reader)

    if len(row) < 3:
        return None

    tag_name = row[0].strip()
    category = int(row[1])
    count = int(row[2])
    aliases_str = row[3] if len(row) > 3 else ""

    # Parse aliases (comma-separated, may be quoted)
    aliases = []
    if aliases_str:
        aliases = [a.strip().strip('"') for a in aliases_str.split(',') if a.strip()]

    return {
        "label": tag_name,
        "value": tag_name,
        "count": count,
        "type": CATEGORY_NAMES.get(category, "general"),
        "category": category,
        "aliases": aliases
    }


def convert_csv_to_json(csv_lines: list[str], min_count: int = 100) -> list[dict]:
    """Convert CSV lines to JSON tag list"""
    tags = []
    skipped = 0

    print(f"Parsing CSV (min count: {min_count})...")

    for line in csv_lines:
        if not line.strip():
            continue

        tag_data = parse_csv_line(line)
        if not tag_data:
            continue

        # Filter by minimum post count
        if tag_data["count"] < min_count:
            skipped += 1
            continue

        tags.append(tag_data)

    print(f"Parsed {len(tags)} tags (skipped {skipped} with count < {min_count})")
    return tags


def save_json(tags: list[dict], output_path: Path):
    """Save tags to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tags, f, ensure_ascii=False, separators=(',', ':'))

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"Saved {len(tags)} tags ({file_size:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert Danbooru tag CSV to JSON")
    parser.add_argument("--min-count", type=int, default=100,
                       help="Minimum post count threshold (default: 100)")
    parser.add_argument("--output", type=str, default="data/tags.json",
                       help="Output JSON file path (default: data/tags.json)")

    args = parser.parse_args()

    # Download CSV
    csv_lines = download_csv(CSV_URL)

    # Convert to JSON
    tags = convert_csv_to_json(csv_lines, min_count=args.min_count)

    # Save to file
    output_path = Path(args.output)
    save_json(tags, output_path)

    print("\n✅ Conversion complete!")
    print(f"📁 Output: {output_path}")
    print(f"📊 Total tags: {len(tags)}")
    print(f"\n💡 Add this line to .gitignore if not already present:")
    print(f"   data/tags.json")


if __name__ == "__main__":
    main()
