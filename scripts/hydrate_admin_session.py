"""Hydrate a [REDACTED] admin session JSON with actual item text from a local items file.

Usage:
    python scripts/hydrate_admin_session.py \
        --admin_session admin_sessions/prod_run_01_external_rating.json \
        --items data/bfi2_items.json \
        --output admin_sessions/local/prod_run_01_hydrated.json

The items file maps item IDs to their text. The admin session's [REDACTED]
values are replaced with the corresponding text from the items file.

Items files and hydrated outputs live in gitignored directories (data/,
admin_sessions/local/) to avoid publishing copyrighted instrument text.
"""

import argparse
import json
import os
import sys


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def hydrate(admin_session: dict, items: dict) -> dict:
    """Replace [REDACTED] item values in admin_session with text from items."""
    result = json.loads(json.dumps(admin_session))  # deep copy
    items_map = items["items"]

    for measure_id, measure in result.get("measures", {}).items():
        measure_items = measure.get("items", {})
        replaced = 0
        skipped = 0
        for item_id, item_text in measure_items.items():
            if item_text == "[REDACTED]":
                if item_id in items_map:
                    measure_items[item_id] = items_map[item_id]
                    replaced += 1
                else:
                    skipped += 1
        print(f"  {measure_id}: {replaced} items hydrated, {skipped} not found in items file")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--admin_session", required=True,
                        help="Path to admin session JSON with [REDACTED] items")
    parser.add_argument("--items", required=True,
                        help="Path to items JSON (e.g., data/bfi2_items.json)")
    parser.add_argument("--output", required=True,
                        help="Path to write hydrated admin session JSON")
    args = parser.parse_args()

    admin_session = load_json(args.admin_session)
    items = load_json(args.items)

    print(f"Hydrating {args.admin_session} with items from {args.items}...")
    result = hydrate(admin_session, items)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote hydrated session to {args.output}")


if __name__ == "__main__":
    main()
