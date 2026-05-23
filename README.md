# Heading-to-Timestamp Alignment in Long-Duration Public Speech and Municipal Meetings Recordings 

This repository contains an anonymized EMNLP artifact for **heading-to-timestamp alignment**: given a natural-language topic heading, the system returns the timestamp where that topic starts in a long transcript.

The benchmark focuses on **long-form public speech recordings**, primarily municipal and government meetings such as city council meetings, town halls, budget hearings, workshops, committee sessions, and public forums. A small subset of long-form public speech recordings is included to increase variation in topic structure and speaking style.

The artifact is released as an **annotation layer** plus reproducibility code. It does **not** redistribute third-party media, full transcripts, or text-containing chunks.

---

## 1. Task overview

Input:

```text
Council Debate and Motion on Emergency Resolution
```

Output:

```text
00:42:18
```

The input heading may be exact, normalized, or query-like. The system retrieves candidate transcript chunks and asks an LLM to identify the candidate timestamp corresponding to the start of the requested topic. The main evaluated strategy uses **time-window chunking**, **hybrid retrieval**, and **constrained Chunk-ID prediction**.

---

## 2. Problem motivation

Finding a topic start time in long recordings is difficult because:

- recordings can be one to three hours long;
- ASR transcripts may be noisy or inconsistent;
- repeated terms such as "budget", "public comment", or "resolution" appear many times;
- topic headings may be paraphrased or normalized;
- retrieval can accidentally pull chunks from the wrong recording if source filtering is not enforced;
- free-form timestamp generation can hallucinate plausible but wrong timestamps.

The project evaluates whether a lightweight RAG pipeline can improve temporal grounding without redistributing the underlying media or transcripts.

---

## 3. What is released

```text
data/
  manifest/source_manifest_release.csv
  annotations/topic_timestamp_annotations.json
  annotations/annotation_provenance_legend.md
  splits/test.json
  test_set.json

src/
  preprocessing/create_chunks.py
  rag/setup_rag.py
  evaluation/eval.py
  evaluation/significance_tests.py

scripts/
  build_test_set_from_annotations.py
  sanitize_results.py

results/
  summaries/

prompts/
configs/
```

The release includes:

- source manifest with public source URLs where available;
- heading--timestamp annotation labels;
- source-safe aliases for local evaluation;
- a test split file;
- prompt templates;
- retrieval, indexing, evaluation, and significance-test scripts;
- sanitized result summaries;
- configuration files and requirements.

---

## 4. What is not released

This artifact does **not** include:

- raw videos;
- raw audio;
- converted WAV files;
- full ASR transcripts;
- text-containing chunks;
- ChromaDB vector database files;
- `.env` files or API keys;
- raw result JSONs containing transcript excerpts.

This artifact does not include raw media, full transcripts, text-containing chunks, or automated third-party media downloaders. Users are responsible for ensuring that any use of external source material complies with the applicable licenses, platform terms, and copyright rules.

---

## 5. Evaluation dataset files

### `data/annotations/topic_timestamp_annotations.json`

This is the main released annotation file. It contains **one JSON object per heading--timestamp query**.

Each row includes:

```json
{
  "query_id": "q_000001",
  "source_id": "src_0002",
  "recording_alias": "src_0002.json",
  "topic_heading": "Call to Order, Roll Call, and Setting Hearing Procedures",
  "timestamp": "00:06:21",
  "timestamp_sec": 381,
  "split": "test",
  "heading_origin": "mixed_public_seed_or_manual_normalized",
  "timestamp_origin": "mixed_public_seed_or_manual_verified",
  "verification_status": "manually_verified_or_author_verified",
  "release_status": "annotation_only_no_media_no_transcript_text"
}
```

This file was derived from the internal evaluation labels by replacing local/raw filenames with stable source aliases and adding timestamp seconds, split metadata, provenance fields, and release-status fields. It is the most important dataset file in the artifact.


### `data/test_set.json`

This is a minimal compatibility file used by `src/evaluation/eval.py`. It contains only:

```json
{
  "source": "src_0002.json",
  "heading": "Call to Order, Roll Call, and Setting Hearing Procedures",
  "timestamp": "00:06:21"
}
```

It can be regenerated from the JSONL annotation file:

```bash
python scripts/build_test_set_from_annotations.py \
  --annotations data/annotations/topic_timestamp_annotations.jsonl \
  --output data/test_set.json
```

### `data/annotations/annotation_provenance_legend.md`

This file explains the meaning of fields such as `heading_origin`, `timestamp_origin`, `verification_status`, and `release_status`.

The provenance fields are intentionally coarse-grained. They document that headings may have been manually written, normalized from public descriptions, or initialized from public timeline/agenda information and then verified by the authors.

### `data/manifest/source_manifest_release.csv`

This file maps stable `source_id` values to public source metadata such as source URL, platform/provenance, release decisions, and notes.

Some rows may still be marked as requiring source verification before final public release. These placeholders are explicit so that unresolved source metadata is not silently presented as verified.

---

## 6. Why use `source_id` and `recording_alias`?

The released annotation files use stable aliases such as:

```text
src_0002
src_0002.json
```

rather than the original local filenames.

This is intentional:

1. It removes local preprocessing traces and downloader-related filename artifacts.
2. It gives each recording a stable, anonymous ID.
3. It lets the annotation file link cleanly to `source_manifest_release.csv`.
4. It keeps evaluation reproducible as long as locally reconstructed transcript/chunk files use the same aliases.

These IDs are **not meant to hide the public source**. The source manifest provides public source URLs where available. The aliases are used to avoid exposing local file names and to make the evaluation format stable.

If users reconstruct transcripts locally, transcript JSON files should be named according to `recording_alias` so that `eval.py` can match test labels to ChromaDB chunk metadata.

---

## 7. Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```


`ffmpeg` is a system dependency for audio conversion:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

Copy `.env.example` if needed:

```bash
cp .env.example .env
```

Do not commit `.env`.

---

## 8. How to run

The public artifact does not include transcript text or text chunks. The following commands are for users who have locally reconstructed transcript JSON files using the released aliases.

### Step 1: Create chunks

For backward compatibility with the original project workflow, the release includes a top-level `main.py` wrapper. The following two commands are equivalent:

```bash
python main.py
```

or

```bash
python src/preprocessing/create_chunks.py \
  --input-dir data/annotations/manuel_timestamped_data \
  --output-root data/chunked \
  --window-sec 90 \
  --overlap-sec 45
```

This creates:

```text
data/chunked/time_window/
data/chunked/fixed_length/
data/chunked/semantic/
```

These outputs contain transcript text and are therefore ignored by `.gitignore`.

### Step 2: Build ChromaDB index

```bash
python src/rag/setup_rag.py \
  --chunk-dir data/chunked/time_window \
  --db-dir data/chroma_db \
  --collection-name meetings \
  --model-name BAAI/bge-large-en-v1.5 \
  --force
```

### Step 3: Run Mistral Chunk-ID evaluation

```bash
python src/evaluation/eval.py \
  --db-dir data/chroma_db \
  --chunks-dir data/chunked/time_window \
  --collection-name meetings \
  --embed-model BAAI/bge-large-en-v1.5 \
  --model-type ollama \
  --llm-model mistral:7b-instruct \
  --retrieval-mode hybrid_bm25 \
  --output-mode chunk_id \
  --disable-stage2-fallback \
  --test-set data/test_set.json \
  --output results/eval_time_90_45_mistral_hybrid_bm25_rrf_chunk_id_strict.json
```

### Step 4: Run OpenAI/GPT evaluation

```bash
python src/evaluation/eval.py \
  --db-dir data/chroma_db \
  --chunks-dir data/chunked/time_window \
  --collection-name meetings \
  --embed-model BAAI/bge-large-en-v1.5 \
  --model-type openai \
  --llm-model gpt-5.4-2026-03-05 \
  --retrieval-mode hybrid_bm25 \
  --output-mode chunk_id \
  --disable-stage2-fallback \
  --test-set data/test_set.json \
  --output results/eval_time_90_45_gpt54_hybrid_bm25_rrf_chunk_id_strict.json
```

The cleaned `eval.py` shows per-file and per-query progress by default. Use `--quiet` to suppress progress output and `--verbose` for additional debugging. Transcript excerpts are not saved unless `--save-excerpts` is explicitly passed.

---

## 9. Result summaries

Sanitized result summaries are provided under:

```text
results/summaries/
```

These summaries preserve aggregate metrics but remove transcript excerpts and retrieved context text.

Main 420-query setting:

| Setting | Model | Retrieval / Output | R@1 | R@3 | R@5 | Exact@30s | Avg. MAE |
|---|---|---|---:|---:|---:|---:|---:|
| Baseline | Mistral-7B | Dense + simple timestamp | 0.164 | 0.281 | 0.319 | 0.288 | 836.98 |
| Baseline | GPT-5.1 | Dense + simple timestamp | 0.164 | 0.281 | 0.319 | 0.286 | 1115.06 |
| Baseline | GPT-5.2 | Dense + simple timestamp | 0.164 | 0.281 | 0.319 | 0.286 | 1080.75 |
| Baseline | GPT-5.4 | Dense + simple timestamp | 0.164 | 0.281 | 0.319 | 0.293 | 1118.31 |
| Ours | Mistral-7B | 90/45 + Hybrid BM25/RRF + Chunk-ID | 0.288 | 0.424 | 0.500 | 0.300 | 760.97 |
| Ours | GPT-5.4 | 90/45 + Hybrid BM25/RRF + Chunk-ID | 0.288 | 0.424 | 0.500 | 0.343 | 743.36 |

Extended 600-query setting:

| Setting | Model | R@1 | R@3 | R@5 | Exact@30s | Avg. MAE |
|---|---|---:|---:|---:|---:|---:|
| Extended | Mistral-7B + Hybrid BM25/RRF + Chunk-ID | 0.357 | 0.493 | 0.563 | 0.373 | 632.01 |

---

## 10. Metrics

- **Recall@K:** whether a relevant/gold-supporting chunk appears among the top-K retrieved candidates.
- **Exact@30s:** fraction of predictions within the given tolerance of the gold timestamp.
- **Avg. MAE:** mean absolute timestamp error in seconds.
- **Answered count (Parsed Output):** number of cases where the model produced a parsable prediction.

---

## Final source-manifest status

The released source manifest contains 200 source rows and all rows include a public source URL. Raw media, full transcripts, and text-containing chunks are not redistributed.
