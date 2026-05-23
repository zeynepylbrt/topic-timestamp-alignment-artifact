#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Remove transcript excerpts and other text-bearing fields from result JSONs."""

import argparse
import json
from pathlib import Path

TEXT_FIELDS_TO_DROP = {"selected_excerpt", "context", "prompt", "retrieved_context"}
OPTIONAL_FIELDS_TO_DROP = {"model_output"}  # keep disabled by default only if you need exact raw outputs internally


def sanitize(input_path, output_path, drop_model_output=False):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    cleaned = {k: v for k, v in data.items() if k != "detailed_results"}
    cleaned_details = []
    for row in data.get("detailed_results", []):
        new = {k: v for k, v in row.items() if k not in TEXT_FIELDS_TO_DROP}
        if drop_model_output:
            new = {k: v for k, v in new.items() if k not in OPTIONAL_FIELDS_TO_DROP}
        cleaned_details.append(new)
    cleaned["detailed_results"] = cleaned_details
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--drop-model-output", action="store_true")
    args = parser.parse_args()
    sanitize(args.input, args.output, args.drop_model_output)


if __name__ == "__main__":
    main()
