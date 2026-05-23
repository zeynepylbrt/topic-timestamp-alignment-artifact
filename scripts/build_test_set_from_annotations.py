#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build eval.py-compatible test_set.json from released annotation JSONL."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/annotations/topic_timestamp_annotations.jsonl")
    parser.add_argument("--output", default="data/test_set.json")
    parser.add_argument("--source-field", default="recording_alias", choices=["recording_alias", "source_id"])
    args = parser.parse_args()

    rows = []
    for line in Path(args.annotations).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        source = obj[args.source_field]
        if args.source_field == "source_id":
            source = f"{source}.json"
        rows.append({"source": source, "heading": obj["topic_heading"], "timestamp": obj["timestamp"]})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
