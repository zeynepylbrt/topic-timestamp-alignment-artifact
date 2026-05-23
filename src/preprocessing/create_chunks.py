#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create transcript chunks for local reconstruction.

This script operates on local transcript JSON files and writes text-containing
chunk files. The generated chunks are not part of the public artifact because
they may contain transcript text.
"""

import argparse
import json
import os
from pathlib import Path

import nltk
import tiktoken
from nltk.tokenize import sent_tokenize


def ts_to_sec(ts):
    parts = str(ts).strip().replace("::", ":").split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


def sec_to_ts(seconds):
    seconds = float(seconds)
    return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"


def create_time_window_chunks(transcript, window_sec=90, overlap_sec=45):
    if overlap_sec >= window_sec:
        raise ValueError("overlap_sec must be smaller than window_sec")
    if not transcript:
        return []
    transcript = sorted(transcript, key=lambda x: float(x.get("start", 0.0)))
    max_end = max(float(seg.get("end", seg.get("start", 0.0))) for seg in transcript)
    step = window_sec - overlap_sec
    chunks = []
    t = 0.0
    chunk_id = 1
    while t <= max_end:
        t0, t1 = t, t + window_sec
        lines = []
        for seg in transcript:
            start = float(seg.get("start", 0.0))
            if t0 <= start < t1:
                text = (seg.get("text") or "").strip()
                if text:
                    speaker = seg.get("speaker", "UNKNOWN")
                    lines.append(f"[{sec_to_ts(start)}] {speaker}: {text}")
        if lines:
            chunks.append({
                "chunk_id": chunk_id,
                "content": "\n".join(lines),
                "start_time": sec_to_ts(t0),
                "end_time": sec_to_ts(min(t1, max_end)),
                "window_sec": window_sec,
                "overlap_sec": overlap_sec,
            })
            chunk_id += 1
        t += step
    return chunks


def create_fixed_length_chunks(full_text, topic_structure, chunk_size=200, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(full_text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_text = enc.decode(tokens[start:end])
        topics_in_chunk = [
            {"heading": t["heading"], "timestamp": t["timestamp"]}
            for t in topic_structure
            if t.get("heading") in chunk_text or t.get("timestamp") in chunk_text
        ]
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "content": chunk_text,
            "topics_in_chunk": topics_in_chunk,
        })
        if end >= len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def create_semantic_chunks(transcript, topic_structure, min_sentences=3, max_sentences=5):
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    sentences = []
    for seg in transcript:
        for sent in sent_tokenize((seg.get("text") or "").strip()):
            if sent.strip():
                sentences.append({
                    "text": sent.strip(),
                    "timestamp": sec_to_ts(seg.get("start", 0.0)),
                    "speaker": seg.get("speaker", "UNKNOWN"),
                    "original_start": float(seg.get("start", 0.0)),
                })
    chunks = []
    current = []
    for sent in sentences:
        current.append(sent)
        if len(current) >= max_sentences:
            chunk_text = "\n".join(f"[{s['timestamp']}] {s['speaker']}: {s['text']}" for s in current)
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "content": chunk_text,
                "start_time": current[0]["timestamp"],
                "end_time": current[-1]["timestamp"],
                "min_sentences": min_sentences,
                "max_sentences": max_sentences,
            })
            current = []
    if current:
        chunk_text = "\n".join(f"[{s['timestamp']}] {s['speaker']}: {s['text']}" for s in current)
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "content": chunk_text,
            "start_time": current[0]["timestamp"],
            "end_time": current[-1]["timestamp"],
            "min_sentences": min_sentences,
            "max_sentences": max_sentences,
        })
    return chunks


def build_topic_text(transcript, topic_structure):
    full_text = []
    for i, topic in enumerate(topic_structure):
        start_sec = ts_to_sec(topic["timestamp"])
        end_sec = ts_to_sec(topic_structure[i + 1]["timestamp"]) if i + 1 < len(topic_structure) else float("inf")
        lines = []
        for seg in transcript:
            if start_sec <= float(seg.get("start", 0.0)) < end_sec:
                lines.append(f"[{sec_to_ts(seg.get('start', 0.0))}] {seg.get('speaker', 'UNKNOWN')}: {(seg.get('text') or '').strip()}")
        full_text.append(f"[TOPIC: {topic['heading']}]\n" + "\n".join(lines))
    return "\n\n".join(full_text)


def write_chunks(output_dir, filename, strategy, topic_structure, chunks, params):
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "source_file": filename,
        "topic_structure": topic_structure,
        "chunking_strategy": strategy,
        **params,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }
    suffix = {"time_window": "_chunks_timewindow.json", "fixed_length": "_chunks.json", "semantic": "_chunks_semantic.json"}[strategy]
    out_path = output_dir / filename.replace(".json", suffix)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Create local text chunks from transcript JSON files.")
    parser.add_argument("--input-dir", default="data/annotations/manuel_timestamped_data")
    parser.add_argument("--output-root", default="data/chunked")
    parser.add_argument("--window-sec", type=int, default=90)
    parser.add_argument("--overlap-sec", type=int, default=45)
    parser.add_argument("--fixed-chunk-size", type=int, default=200)
    parser.add_argument("--fixed-overlap", type=int, default=50)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    for path in sorted(input_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        topic_structure = data.get("topic_structure", [])
        transcript = data.get("transcript", [])
        if not topic_structure or not transcript:
            print(f"Skipping {path.name}: missing topic_structure or transcript")
            continue
        time_chunks = create_time_window_chunks(transcript, args.window_sec, args.overlap_sec)
        write_chunks(output_root / "time_window", path.name, "time_window", topic_structure, time_chunks, {"window_sec": args.window_sec, "overlap_sec": args.overlap_sec})

        full_text = build_topic_text(transcript, topic_structure)
        fixed_chunks = create_fixed_length_chunks(full_text, topic_structure, args.fixed_chunk_size, args.fixed_overlap)
        write_chunks(output_root / "fixed_length", path.name, "fixed_length", topic_structure, fixed_chunks, {"chunk_size": args.fixed_chunk_size, "chunk_overlap": args.fixed_overlap})

        semantic_chunks = create_semantic_chunks(transcript, topic_structure)
        write_chunks(output_root / "semantic", path.name, "semantic", topic_structure, semantic_chunks, {})
        print(f"Processed {path.name}: time={len(time_chunks)}, fixed={len(fixed_chunks)}, semantic={len(semantic_chunks)}")


if __name__ == "__main__":
    main()
