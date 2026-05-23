"""Evaluation runner for heading-to-timestamp alignment.

The script supports dense and hybrid retrieval modes, candidate-ID constrained
prediction, direct timestamp prediction, and free-form timestamp baselines.
It expects locally reconstructed text chunks and a benchmark annotation file.
"""

import os
import sys
import json
import requests
import chromadb
import re
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from datetime import datetime
from collections import defaultdict, Counter
import math
from dotenv import load_dotenv
load_dotenv()

EARLY_HINTS = ["call to order","pledge","roll call","opening","welcome","invocation"]

def is_early_heading(heading: str) -> bool:
    h = (heading or "").lower()
    return any(x in h for x in EARLY_HINTS)

def build_query_variants(heading: str) -> list[str]:
    base = [
        f"meeting topic {heading}",
        f"{heading} meeting section",
        f"topic: {heading}",
        f"heading: {heading}",
        heading,
    ]
    if is_early_heading(heading):
        base = [
            f"{heading} at the beginning of the meeting",
            f"meeting start {heading}",
            f"opening section {heading}",
            "beginning of the meeting",
            "meeting starts",
        ] + base
    return base

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


TS_LINE_RE = re.compile(r'^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$')

KEYWORDS = [
    "call to order", "roll call", "pledge", "invocation", "agenda", "minutes",
    "item", "motion", "welcome", "opening", "adjourn", "public comment"
]
STOPWORDS = set("""
a an the and or of to in on for with vs into from by at as is are was were be been being
meeting council commission committee board item agenda minutes approval adoption call order roll
""".split())

def heading_keywords(heading: str, max_kw=6):
    h = (heading or "").lower()
    tokens = re.findall(r"[a-z0-9]+", h)
    toks = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_kw:
            break
    return out


def extract_lines(context: str):
    lines = []
    for raw in (context or "").splitlines():
        m = TS_LINE_RE.match(raw.strip())
        if not m:
            continue
        ts = m.group(1)
        text = m.group(2)
        lines.append((ts, text))
    return lines



def build_candidates(context: str, heading: str, max_candidates=15):
    lines = extract_lines(context)
    hk = heading_keywords(heading)

    candidates = []
    for ts, text in lines[:10]:
        candidates.append((ts, text))

    for ts, text in lines:
        low = text.lower()
        if any(k in low for k in hk) or any(k in low for k in KEYWORDS):
            candidates.append((ts, text))
        if len(candidates) >= max_candidates * 3:
            break

    seen = set()
    out = []
    for ts, text in candidates:
        if ts in seen:
            continue
        seen.add(ts)
        out.append((ts, text))
        if len(out) >= max_candidates:
            break
    return out


# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

DEFAULT_CHROMA_DIR = 'data/chroma_db'
DEFAULT_CHUNKS_DIR = 'data/chunked/fixed_length'
DEFAULT_OLLAMA_URL = 'http://localhost:11434/api/generate'
DEFAULT_LLM_MODEL = 'mistral:7b-instruct'
DEFAULT_EMBED_MODEL = 'BAAI/bge-large-en-v1.5'
K_VALUES = [1, 3, 5]
EXACT_TOLERANCES_SEC = [2, 10, 30]

CHROMA_DIR = None
CHUNKS_DIR = None
COLLECTION_NAME = None
OLLAMA_URL = None
LLM_MODEL = None
embed_model = None
client = None
collection = None
BM25_CACHE = {}
TEST_SET_PATH = 'data/test_set.json'



def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with different embedding models')
    parser.add_argument('--db-dir', type=str, default=DEFAULT_CHROMA_DIR)
    parser.add_argument('--chunks-dir', type=str, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument('--collection-name', type=str, default='meetings')
    parser.add_argument('--embed-model', type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument('--use-metadata', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--ollama-url', type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument('--llm-model', type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument('--test-limit', type=int, default=None)
    parser.add_argument('--test-set', type=str, default='data/test_set.json', help='Path to evaluation test set JSON')
    parser.add_argument('--save-excerpts', action='store_true', help='Save selected transcript excerpts in detailed_results. Off by default for artifact release.')
    parser.add_argument('--verbose', action='store_true', help='Print additional debug messages.')
    parser.add_argument('--quiet', action='store_true', help='Disable per-file and per-query progress output.')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--model-type', type=str, default='ollama', choices=['ollama', 'openai'])
    parser.add_argument('--openai-api-key', type=str, default=None)
    parser.add_argument(
    '--retrieval-mode',
    type=str,
    default='dense_only',
    choices=['dense_only', 'hybrid_bm25'],
    help='Retrieval mode for non-baseline runs.'
)
    parser.add_argument(
    '--output-mode',
    type=str,
    default='chunk_id',
    choices=['chunk_id', 'direct_timestamp', 'freeform_timestamp'],
     help='Output mode: chunk_id, direct_timestamp, or freeform_timestamp.'
)

    parser.add_argument(
        '--disable-stage2-fallback',
        action='store_true',
        help='Disable deterministic fallback extraction for strict output-strategy ablation.'
    )
    parser.add_argument(
        '--include-procedural',
        action='store_true',
        help='Include procedural headings instead of filtering them out.'
)


    return parser.parse_args()


SKIP_HINTS = [
    'call to order', 'roll call', 'pledge', 'invocation', 'opening remark',
    'welcome', 'adjourn', 'closing remark', 'pre-meeting', 'introduction and invocation'
]

def load_test_cases_by_file(include_procedural=False):
    with open(TEST_SET_PATH, encoding='utf-8') as f:
        test_set = json.load(f)

    file_cases = defaultdict(list)
    for tc in test_set:
        h = tc['heading'].lower()
        if (not include_procedural) and any(hint in h for hint in SKIP_HINTS):
            continue
        source = tc['source']
        file_cases[source].append({
            "question": f"I have the following topic: '{tc['heading']}'. Give me the exact timestamp.",
            "expected_timestamp": tc['timestamp'],
            "expected_heading": tc['heading'],
            "source": source
        })
    return dict(file_cases)


def ts_to_sec(ts):
    if not ts:
        return None
    ts = ts.strip().replace('::', ':')
    p = ts.split(":")
    try:
        if len(p) == 3:
            return int(p[0])*3600 + int(p[1])*60 + int(float(p[2]))
        elif len(p) == 2:
            return int(p[0])*60 + int(float(p[1]))
    except:
        return None


def extract_timestamp(text):
    if not text:
        return None
    match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', text)
    if match:
        ts = match.group(1)
        if ts.count(':') == 1:
            ts = f"00:{ts}"
        return ts
    match = re.search(r'(\d{1,2}):(\d{1,2}):(\d{1,2})', text)
    if match:
        h, m, s = match.groups()
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    return None


def build_prompt(context, question, expected_heading):
    return f"""You are given a transcript chunk from a meeting. Each line starts with a timestamp in brackets [HH:MM:SS] followed by the speaker and text. Topics are marked with [TOPIC: ...] lines.

Context:
{context}

Question: {question}

INSTRUCTIONS (follow exactly):
1. Find the line that starts with "[TOPIC: {expected_heading}]" in the context
2. Look at the VERY NEXT line that has a timestamp (it will start with [HH:MM:SS])
3. That timestamp is the exact start time of this topic
4. Output ONLY that timestamp in HH:MM:SS format
5. Do NOT add any extra text, explanations, or formatting

Example:
If context has:
[TOPIC: Pledge of Allegiance]
[00:01:42] SPEAKER_01: Please rise...
You should output: 00:01:42

Answer:"""


def build_prompt_candidates(heading: str, candidates):
    heading_lower = (heading or "").lower()
    is_early = any(h in heading_lower for h in EARLY_HINTS)

    cand_lines = []
    for i, (ts, text) in enumerate(candidates, 1):
        cid = f"C{i:02d}"
        short = text.strip()
        if len(short) > 180:
            short = short[:180] + "..."
        # Real pipeline: timestamp is visible as candidate metadata,
        # but the model must output only the candidate ID.
        cand_lines.append(f"{cid} | {ts} | {short}")

    cand_block = "\n".join(cand_lines)

    early_note = ""
    if is_early:
        early_note = (
            "Note: This heading is likely an EARLY procedural section. "
            "Prefer candidates near the beginning of the candidate list if the excerpts are similarly relevant.\n"
        )

    return f"""You are matching a user-provided meeting heading to where it starts in the transcript.

Heading: "{heading}"

{early_note}Choose the SINGLE best candidate that marks the START of this heading/section.

Candidates (ID | timestamp | excerpt):
{cand_block}

Rules:
- Output ONLY the candidate ID, for example C03.
- Do NOT output a timestamp.
- Do NOT invent a new ID.
- Do NOT add explanations or extra text.
- Prefer the candidate whose excerpt best matches the heading.

"""

def build_prompt_direct_timestamp_from_candidates(heading: str, candidates):
    heading_lower = (heading or "").lower()
    is_early = any(h in heading_lower for h in EARLY_HINTS)

    cand_lines = []
    for i, (ts, text) in enumerate(candidates, 1):
        cid = f"C{i:02d}"
        short = text.strip()
        if len(short) > 180:
            short = short[:180] + "..."
        # Keep the same candidate table format across output modes for fair comparison.
        cand_lines.append(f"{cid} | {ts} | {short}")

    cand_block = "\n".join(cand_lines)

    early_note = ""
    if is_early:
        early_note = (
            "Note: This heading is likely an EARLY procedural section. "
            "Prefer candidates near the beginning of the candidate list if the excerpts are similarly relevant.\n"
        )

    return f"""You are matching a user-provided meeting heading to where it starts in the transcript.

Heading: "{heading}"

{early_note}Choose the SINGLE best candidate that marks the START of this heading/section.

Candidates (ID | timestamp | excerpt):
{cand_block}

Rules:
- Output ONLY the timestamp of the best candidate in HH:MM:SS format.
- Do NOT output a candidate ID.
- Do NOT add explanations or extra text.
- Prefer the candidate whose excerpt best matches the heading.

Answer:"""


#Rules:
#- Output ONLY the candidate ID (e.g., C03).
#- Do NOT output a timestamp.
#Prefer candidates whose excerpt contains words from the heading.

def build_prompt_freeform_timestamp(context, question, expected_heading):
    return f"""You are given retrieved transcript chunks from a meeting.
Each transcript line starts with a timestamp in brackets [HH:MM:SS].

Context:
{context}

Question: {question}

Task:
Find the start timestamp of the topic: "{expected_heading}".

Rules:
- Output ONLY one timestamp in HH:MM:SS format.
- Do NOT output a candidate ID.
- Do NOT add explanations or extra text.
- Use a timestamp that appears in the transcript context.
- If the exact topic marker is not present, choose the earliest timestamped line where the topic is explicitly introduced or substantively addressed.

Answer:"""

def extract_candidate_id(text: str):
    if not text:
        return None
    m = re.search(r'\bC(\d{2})\b', text.strip().upper())
    return f"C{m.group(1)}" if m else None


# ---------------------------------------------------------------------------
# [IMP-2] BM25-style keyword re-ranker
# ---------------------------------------------------------------------------

def keyword_score(doc_text: str, heading: str) -> float:
    """
    Lightweight TF-style score: count heading keyword hits in doc text.
    Returns a score in [0, 1] (normalized by number of keywords).
    No extra dependencies — pure Python.
    """
    hk = heading_keywords(heading, max_kw=8)
    if not hk:
        return 0.0
    low = (doc_text or "").lower()
    hits = sum(1 for k in hk if k in low)
    return hits / len(hk)


def rerank_results(rag_results, heading: str, is_early: bool) -> dict:
    """
    [IMP-2] + [IMP-3]  Re-rank retrieved chunks using:
      - keyword overlap with heading (BM25-proxy)
      - temporal prior for early headings (prefer lower start_sec)

    Returns a new rag_results dict with re-ordered entries.
    """
    docs   = rag_results["documents"][0]
    metas  = rag_results["metadatas"][0]
    ids    = rag_results["ids"][0]
    dists  = rag_results["distances"][0]

    if not docs:
        return rag_results

    # Normalise semantic distance → similarity in [0,1]
    raw_dists = [safe_float(d, default=1.0) for d in dists]
    max_d = max(raw_dists) if raw_dists else 1.0
    sem_sims = [1.0 - (d / (max_d + 1e-9)) for d in raw_dists]

    scores = []
    for i, (doc, meta, sem) in enumerate(zip(docs, metas, sem_sims)):
        kw   = keyword_score(doc, heading)
        # [IMP-1] also score against heading-augmented text embedded in chunk
        # (chunks built with [TOPIC: ...] markers get extra keyword hits for free)

        # [IMP-3] temporal prior: favour chunks near meeting start for early headings
        start_sec = safe_float(meta.get("start_sec"), default=None)
        temporal_bonus = 0.0
        if is_early and start_sec is not None:
            # Decay: 1.0 at t=0, 0.0 at t=300s (5 min mark)
            temporal_bonus = max(0.0, 1.0 - start_sec / 300.0) * 0.15

        # Weighted combination  (tunable)
        combined = 0.55 * sem + 0.30 * kw + temporal_bonus
        scores.append((i, combined))

    scores.sort(key=lambda x: x[1], reverse=True)
    order = [i for i, _ in scores]

    return {
        "ids":        [[ids[i]   for i in order]],
        "documents":  [[docs[i]  for i in order]],
        "metadatas":  [[metas[i] for i in order]],
        "distances":  [[dists[i] for i in order]],
    }

# Context builder: keep the top retrieved chunks and order them chronologically.
def build_context(rag_results, use_metadata, cluster_size=3, max_chunks=10, top_k=None):
    docs = rag_results["documents"][0]
    metas = rag_results["metadatas"][0]
    
    # Keep only the top retrieved chunks.
    top_n = min(max_chunks, len(docs))
    pairs = list(zip(docs[:top_n], metas[:top_n]))

    # Helper for chronological ordering.
    def get_start(meta):
        return float(meta.get("start_sec", 0.0))

    # Preserve all selected chunks, ordered by start time.
    pairs.sort(key=lambda dm: get_start(dm[1]))

    # Join chunks with a clear separator.
    parts = []
    for doc, meta in pairs:
        parts.append(doc)
        
    return "\n\n---\n\n".join(parts)



def pinpoint_timestamp_in_context(context: str, heading: str) -> str | None:
    """
    Stage-2: given the context string already found by retrieval,
    scan for the best timestamp line WITHOUT an LLM call.

    Priority order:
      1. Line immediately after a [TOPIC: <heading>] marker
      2. First line whose text contains a heading keyword hit
      3. First timestamp line in the context (fallback)
    """
    lines = context.splitlines()
    hk    = heading_keywords(heading, max_kw=8)

    # Priority 1: TOPIC marker
    for idx, line in enumerate(lines):
        if re.search(rf'\[TOPIC:.*{re.escape(heading[:20])}', line, re.IGNORECASE):
            # Look forward for first [HH:MM:SS] line
            for nxt in lines[idx + 1 : idx + 6]:
                m = TS_LINE_RE.match(nxt.strip())
                if m:
                    return m.group(1)

    # Priority 2: keyword hit in transcript line
    for line in lines:
        m = TS_LINE_RE.match(line.strip())
        if not m:
            continue
        low = m.group(2).lower()
        if any(k in low for k in hk):
            return m.group(1)

    # Priority 3: first timestamp line
    for line in lines:
        m = TS_LINE_RE.match(line.strip())
        if m:
            return m.group(1)

    return None


def precision_at_k(rag_results, expected_heading, expected_source, expected_sec, k):
    for meta in rag_results['metadatas'][0][:k]:
        if meta.get('source_file', '') != expected_source:
            continue
        topics_raw = meta.get('topics', '[]')
        try:
            topics = json.loads(topics_raw) if isinstance(topics_raw, str) else (topics_raw or [])
        except Exception:
            topics = []
        if topics:
            for t in topics:
                heading = (t.get('heading', '') or '')
                if expected_heading.lower() in heading.lower() or heading.lower() in expected_heading.lower():
                    return 1.0
        else:
            if expected_sec is None:
                continue
            start_sec = meta.get('start_sec')
            end_sec   = meta.get('end_sec')
            if start_sec is None or end_sec is None:
                continue
            try:
                if float(start_sec) <= float(expected_sec) <= float(end_sec):
                    return 1.0
            except Exception:
                continue
    return 0.0


def bm25_tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25 sparse retrieval.
    Keeps alphanumeric terms and removes very common stopwords.
    """
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [
        t for t in tokens
        if len(t) >= 2 and t not in STOPWORDS
    ]


def get_bm25_index_for_source(source: str):
    """
    Build and cache a BM25 index for one source_file.
    This avoids rebuilding the sparse index for every query from the same meeting.
    """
    global collection, BM25_CACHE

    if source in BM25_CACHE:
        return BM25_CACHE[source]

    all_chunks = collection.get(
        where={"source_file": source},
        include=["documents", "metadatas"]
    )

    ids = all_chunks.get("ids", [])
    docs = all_chunks.get("documents", [])
    metas = all_chunks.get("metadatas", [])

    tokenized_docs = [bm25_tokenize(doc) for doc in docs]
    doc_lengths = [len(toks) for toks in tokenized_docs]
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0

    df = Counter()
    for toks in tokenized_docs:
        df.update(set(toks))

    n_docs = len(tokenized_docs)
    idf = {}
    for term, freq in df.items():
        # Standard BM25-style IDF with +1 smoothing
        idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    BM25_CACHE[source] = {
        "ids": ids,
        "documents": docs,
        "metadatas": metas,
        "tokenized_docs": tokenized_docs,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
    }

    return BM25_CACHE[source]


def bm25_search(query: str, source: str, top_k: int = 20, k1: float = 1.5, b: float = 0.75):
    """
    True BM25 sparse retrieval over chunks from the given source_file.
    Returns results in a Chroma-like format.
    """
    index = get_bm25_index_for_source(source)

    ids = index["ids"]
    docs = index["documents"]
    metas = index["metadatas"]
    tokenized_docs = index["tokenized_docs"]
    doc_lengths = index["doc_lengths"]
    avgdl = index["avgdl"]
    idf = index["idf"]

    query_terms = bm25_tokenize(query)
    if not query_terms or not docs:
        return None

    scores = []

    for i, toks in enumerate(tokenized_docs):
        tf = Counter(toks)
        dl = doc_lengths[i] if i < len(doc_lengths) else 0

        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue

            freq = tf[term]
            term_idf = idf.get(term, 0.0)

            denom = freq + k1 * (1.0 - b + b * (dl / (avgdl + 1e-9)))
            score += term_idf * ((freq * (k1 + 1.0)) / (denom + 1e-9))

        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]

    return {
        "ids": [[ids[i] for i, _ in top]],
        "documents": [[docs[i] for i, _ in top]],
        "metadatas": [[metas[i] for i, _ in top]],
        # Chroma distances are lower-is-better; BM25 score is higher-is-better.
        # We store negative score only for compatibility.
        "distances": [[-score for _, score in top]],
    }


def retrieve_hybrid_bm25_rrf(query_heading: str, source: str, top_k: int = 20, rrf_k: int = 60):
    """
    Hybrid retrieval:
      1. Dense retrieval over multiple query variants
      2. BM25 sparse retrieval over the same source transcript
      3. RRF fusion across all ranking lists

    This function does NOT use topic metadata and therefore avoids heading leakage.
    """
    global embed_model, collection

    acc = {}

    def add_ranked_results(res, weight: float = 1.0):
        if res is None or not res.get("ids") or not res["ids"][0]:
            return

        ids = res["ids"][0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for rank, doc_id in enumerate(ids, start=1):
            score = weight * (1.0 / (rrf_k + rank))

            doc = docs[rank - 1] if rank - 1 < len(docs) else None
            meta = metas[rank - 1] if rank - 1 < len(metas) else None
            dist = dists[rank - 1] if rank - 1 < len(dists) else None

            if doc_id not in acc:
                acc[doc_id] = [score, doc, meta, dist]
            else:
                acc[doc_id][0] += score

                # Keep the best available document/meta representation
                if acc[doc_id][1] is None and doc is not None:
                    acc[doc_id][1] = doc
                if acc[doc_id][2] is None and meta is not None:
                    acc[doc_id][2] = meta
                if dist is not None and (acc[doc_id][3] is None or dist < acc[doc_id][3]):
                    acc[doc_id][3] = dist

    # Dense retrieval over query variants
    for qf in build_query_variants(query_heading):
        emb = embed_model.encode([qf]).tolist()
        dense_res = collection.query(
            query_embeddings=emb,
            n_results=top_k,
            where={"source_file": source}
        )
        add_ranked_results(dense_res, weight=1.0)

    # BM25 sparse retrieval
    bm25_res = bm25_search(query_heading, source, top_k=top_k)
    add_ranked_results(bm25_res, weight=1.0)

    if not acc:
        return None

    ranked = sorted(acc.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]

    return {
        "ids": [[x[0] for x in ranked]],
        "documents": [[x[1][1] for x in ranked]],
        "metadatas": [[x[1][2] for x in ranked]],
        "distances": [[x[1][3] for x in ranked]],
    }

# Dense-only retrieval used for the ablation setting.
def retrieve_dense_only(query_heading, source, top_k=5):
    """Run source-filtered dense retrieval without BM25/RRF or query variants."""
    global embed_model, collection

    # Encode only the original heading.
    emb = embed_model.encode([query_heading]).tolist()

    # Query ChromaDB directly without RRF.
    res = collection.query(
        query_embeddings=emb,
        n_results=top_k,
        where={"source_file": source}
    )

    if not res.get("ids") or not res["ids"][0]:
        return None

    # Return the same structure as the hybrid retriever.
    return {
        "ids":       res["ids"],
        "documents": res["documents"],
        "metadatas": res["metadatas"],
        "distances": res["distances"]
    }


def calculate_mrr(rag_results, expected_heading, expected_source, expected_sec):
    for rank, meta in enumerate(rag_results['metadatas'][0], 1):
        if meta.get('source_file', '') != expected_source:
            continue
        topics_raw = meta.get('topics', '[]')
        try:
            topics = json.loads(topics_raw) if isinstance(topics_raw, str) else (topics_raw or [])
        except Exception:
            topics = []
        if topics:
            for t in topics:
                if expected_heading.lower() in (t.get('heading', '') or '').lower():
                    return 1.0 / rank
        else:
            if expected_sec is None:
                continue
            start_sec = meta.get('start_sec')
            end_sec   = meta.get('end_sec')
            if start_sec is None or end_sec is None:
                continue
            try:
                if float(start_sec) <= float(expected_sec) <= float(end_sec):
                    return 1.0 / rank
            except Exception:
                continue
    return 0.0

# Main evaluation loop.

def run_eval(use_metadata=True, test_limit=None, args=None):
    is_baseline = args.baseline if args else False

    global CHROMA_DIR, CHUNKS_DIR, COLLECTION_NAME, LLM_MODEL, OLLAMA_URL
    global embed_model, client, collection

    condition = "WITH_METADATA" if use_metadata else "NO_METADATA"
    if is_baseline:
        mode_name = "BASELINE_DENSE_SIMPLE"
    elif args.retrieval_mode == "hybrid_bm25":
        mode_name = "HYBRID_BM25_RRF"
    else:
        mode_name = "DENSE_ONLY"
    if not is_baseline:
        mode_name = f"{mode_name}_{args.output_mode.upper()}"    


    print(f"\n{'='*60}")
    print(f"CONDITION: {condition} — {mode_name}")
    print(f"{'='*60}")

    is_time_window = "time_window" in (args.chunks_dir or "")

    file_cases = load_test_cases_by_file(
        include_procedural=args.include_procedural
)

    all_predictions = []
    precision_scores = {k: [] for k in K_VALUES}
    mrr_scores = []
    file_stats = {}
    exact_hits = {t: 0 for t in EXACT_TOLERANCES_SEC}

    total_tests_run = 0
    total_test_cases = 0

    for source, cases in file_cases.items():
        if not cases:
            continue

        if not args.quiet:
            print(f"\n File: {source}", flush=True)

        if test_limit:
            remaining = test_limit - total_tests_run
            if remaining <= 0:
                break
            test_cases = cases[:min(len(cases), remaining)]
        else:
            test_cases = cases[:3]

        if not test_cases:
            continue

        total_tests_run += len(test_cases)
        file_correct = 0

        for i, tc in enumerate(test_cases):
            stage2_ts = None
            predicted = None
            answer = None
            selected_id = None
            selected_excerpt = None
            candidates = []
            timestamp_in_candidates = None


            if not args.quiet:
                print(f"\n  [{i+1}/{len(test_cases)}] {tc['expected_timestamp']} | {tc['expected_heading'][:50]}...", flush=True)

            total_test_cases += 1
            expected_sec = ts_to_sec(tc['expected_timestamp'])

            # ------------------------------------------------------------
            # Retrieval + Prompt construction
            # ------------------------------------------------------------
            if is_baseline:
                # True baseline:
                # source-filtered dense retrieval + simple timestamp prompt
                rag_results = retrieve_baseline(tc['expected_heading'], source, top_k=5)

                if rag_results is None or not rag_results.get("ids") or not rag_results["ids"][0]:
                    if not args.quiet:
                        print("    No retrieved chunk found for this source.", flush=True)
                    all_predictions.append({
                        "source": source,
                        "expected": tc['expected_timestamp'],
                        "predicted": None,
                        "exact_match": False,
                        "mae_sec": None,
                        "model_output": None
                    })
                    continue

                context = build_context(
                    rag_results,
                    use_metadata=False,
                    cluster_size=3,
                    max_chunks=5,
                )

                # No candidate prompt, no stage-2 fallback
                stage2_ts = None
                prompt = build_prompt_baseline(context, tc['question'])

            else:
                # Non-baseline runs:
                # dense_only  = dense retrieval + candidate selection
                # hybrid_bm25 = dense + BM25 RRF retrieval + candidate selection

                if args.retrieval_mode == "hybrid_bm25":
                    rag_results = retrieve_hybrid_bm25_rrf(
                        tc['expected_heading'],
                        source,
                        top_k=20
                    )

                    # Optional lexical + temporal re-ranking after Dense+BM25 RRF
                    is_early = is_early_heading(tc['expected_heading'])
                    rag_results = rerank_results(
                        rag_results,
                        tc['expected_heading'],
                        is_early
                    ) if rag_results is not None else None

                else:
                    rag_results = retrieve_dense_only(
                        tc['expected_heading'],
                        source,
                        top_k=5
                    )

                if rag_results is None or not rag_results.get("ids") or not rag_results["ids"][0]:
                    if not args.quiet:
                        print("    No retrieved chunk found for this source.", flush=True)
                    all_predictions.append({
                        "source": source,
                        "expected": tc['expected_timestamp'],
                        "predicted": None,
                        "exact_match": False,
                        "mae_sec": None,
                        "model_output": None
                    })
                    continue

                context = build_context(
                    rag_results,
                    use_metadata=False,
                    cluster_size=3,
                    max_chunks=5,
                )

                stage2_ts = pinpoint_timestamp_in_context(context, tc['expected_heading'])

                if is_time_window:
                    if args.output_mode == "chunk_id":
                        candidates = build_candidates(context, tc['expected_heading'])
                        prompt = build_prompt_candidates(tc['expected_heading'], candidates)

                    elif args.output_mode == "direct_timestamp":
                        candidates = build_candidates(context, tc['expected_heading'])
                        prompt = build_prompt_direct_timestamp_from_candidates(
                            tc['expected_heading'],
                            candidates
                        )

                    elif args.output_mode == "freeform_timestamp":
                        candidates = None
                        prompt = build_prompt_freeform_timestamp(
                            context,
                            tc['question'],
                            tc['expected_heading']
                        )

                    else:
                        raise ValueError(f"Unknown output_mode: {args.output_mode}")
                else:
                    prompt = build_prompt(context, tc['question'], tc['expected_heading'])
            # ------------------------------------------------------------
            # Retrieval metrics
            # ------------------------------------------------------------
            for k in K_VALUES:
                score = precision_at_k(
                    rag_results,
                    tc['expected_heading'],
                    source,
                    expected_sec,
                    k
                )
                precision_scores[k].append(score)

            mrr = calculate_mrr(
                rag_results,
                tc['expected_heading'],
                source,
                expected_sec
            )
            mrr_scores.append(mrr)

            # ------------------------------------------------------------
            # LLM call + prediction parsing
            # ------------------------------------------------------------
            try:
                if args.model_type == 'openai':
                    answer = generate_with_openai(prompt, model=LLM_MODEL, verbose=args.verbose)
                else:
                    response = requests.post(
                        OLLAMA_URL,
                        json={
                            "model": LLM_MODEL,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "max_tokens": 50
                            }
                        },
                        timeout=300
                    )
                    answer = response.json()['response'].strip()

                # Baseline: simple prompt, no candidate IDs
                if is_baseline:
                    predicted = extract_timestamp(answer)

                # Dense-only ablation with candidate selection
                elif is_time_window:
                    if args.output_mode == "chunk_id":
                        selected_id = extract_candidate_id(answer)

                        if selected_id is not None:
                            idx = int(selected_id[1:]) - 1

                            if candidates is not None and 0 <= idx < len(candidates):
                                predicted = candidates[idx][0]
                                selected_excerpt = candidates[idx][1]
                            else:
                                predicted = None
                        else:
                            predicted = None

                    elif args.output_mode == "direct_timestamp":
                        predicted = extract_timestamp(answer)

                        # For analysis only: check whether the model copied a candidate timestamp
                        candidate_timestamps = {ts for ts, _ in candidates}
                        if predicted is not None and predicted not in candidate_timestamps:
                            if args.verbose:
                                print(f"    [direct_timestamp] timestamp not in candidates: {predicted}")

                    elif args.output_mode == "freeform_timestamp":
                        predicted = extract_timestamp(answer)

                        # For analysis only: check whether the model used a timestamp from the retrieved context
                        context_timestamps = {ts for ts, _ in extract_lines(context)}
                        if predicted is not None and predicted not in context_timestamps:
                            if args.verbose:
                                print(f"    [freeform_timestamp] timestamp not in retrieved context: {predicted}")

                    else:
                        raise ValueError(f"Unknown output_mode: {args.output_mode}")


                # Stage-2 fallback is not allowed for true baseline
                if (
                    (not is_baseline)
                    and (not args.disable_stage2_fallback)
                    and predicted is None
                    and stage2_ts is not None
                ):
                    predicted = stage2_ts

            except Exception as e:
                if not args.quiet:
                    print(f"    Error: {e}", flush=True)
                predicted = None
                answer = None

            timestamp_in_candidates = None
            timestamp_in_context = None

            if is_time_window and predicted is not None:
                context_timestamps = {ts for ts, _ in extract_lines(context)}
                timestamp_in_context = predicted in context_timestamps

                if candidates:
                    timestamp_in_candidates = predicted in {ts for ts, _ in candidates}
            # ------------------------------------------------------------
            # Timestamp metrics
            # ------------------------------------------------------------
            
            
            predicted_sec = ts_to_sec(predicted)

            if predicted_sec is not None and expected_sec is not None:
                for t in EXACT_TOLERANCES_SEC:
                    if abs(predicted_sec - expected_sec) <= t:
                        exact_hits[t] += 1

            exact = (
                abs(predicted_sec - expected_sec) <= 2
                if predicted_sec is not None and expected_sec is not None
                else False
            )

            mae = (
                abs(predicted_sec - expected_sec)
                if predicted_sec is not None and expected_sec is not None
                else None
            )

            if exact:
                file_correct += 1

            prediction_record = {
                "source": source,
                "expected": tc['expected_timestamp'],
                "predicted": predicted,
                "exact_match": exact,
                "mae_sec": mae,
                "model_output": answer,
                "selected_id": selected_id,
                "output_mode": args.output_mode,
                "timestamp_in_candidates": timestamp_in_candidates,
                "timestamp_in_context": timestamp_in_context,
            }
            if args.save_excerpts:
                prediction_record["selected_excerpt"] = selected_excerpt
            all_predictions.append(prediction_record)

            if not args.quiet:
                print(f"    Prediction: {predicted} | Exact: {exact} | MAE: {mae if mae is not None else 'N/A'}s", flush=True)

            if args.verbose and answer is not None:
                short_answer = str(answer).replace("\n", " ")[:160]
                print(f"    Model output: {short_answer}", flush=True)

        file_stats[source] = {
            "total": len(test_cases),
            "correct": file_correct,
            "accuracy": file_correct / len(test_cases) if test_cases else 0
        }

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT — {condition} — {mode_name}")
    print(f"{'='*60}")

    print("\n Precision@K (Retrieval - Source Filtered):")
    for k in K_VALUES:
        avg = np.mean(precision_scores[k]) if precision_scores[k] else 0
        print(f"  Precision@{k}: {avg:.3f}")

    mrr_avg = float(np.mean(mrr_scores)) if mrr_scores else None
    print(f"\n MRR: {mrr_avg:.3f}" if mrr_avg is not None else "\n MRR: N/A")

    mae_vals = [
        p["mae_sec"]
        for p in all_predictions
        if p.get("mae_sec") is not None
    ]

    avg_mae = float(np.mean(mae_vals)) if mae_vals else None
    n = len(all_predictions)

    exact_rates = {
        f"exact@{t}s": exact_hits[t] / n if n else 0.0
        for t in EXACT_TOLERANCES_SEC
    }

    print("\n Exact Match (tolerance):")
    for t in EXACT_TOLERANCES_SEC:
        print(f"  Exact@{t}s: {exact_rates[f'exact@{t}s']:.1%}")

    print("\n Final Results:")
    for t in EXACT_TOLERANCES_SEC:
        print(f"  Exact@{t}s:   {exact_hits[t]}/{n} ({exact_rates[f'exact@{t}s']:.1%})")

    print(f"  Avg MAE:     {avg_mae:.1f}s" if avg_mae is not None else "  Avg MAE:     N/A")
    print(f"  Answered:    {len(mae_vals)}/{n}")

    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "condition": condition,
        "retrieval_mode": mode_name,
        "model": LLM_MODEL,
        "embed_model": embed_model.__class__.__name__,
        "total_test_cases": n,
        "precision_at_k": {
            f"P@{k}": round(float(np.mean(v)), 3)
            for k, v in precision_scores.items()
            if v
        },
        "mrr": round(mrr_avg, 3) if mrr_avg is not None else None,
        "avg_mae_sec": avg_mae,
        "answered_count": len(mae_vals),
        "exact_match_rates": exact_rates,
        "exact_match_counts": {
            f"exact@{t}s": exact_hits[t]
            for t in EXACT_TOLERANCES_SEC
        },
        "file_stats": file_stats,
        "detailed_results": all_predictions,
    }


def retrieve_baseline(query_heading,source, top_k=5):
    global embed_model, collection

    emb = embed_model.encode([query_heading]).tolist()

    results = collection.query(
        query_embeddings=emb,
        n_results=top_k,
        where={"source_file": source}
    )

    return results

def build_prompt_baseline(context, question):
    return f"""Context: {context}

Question: {question}

Answer with only the timestamp:"""


def generate_with_openai(prompt, model="gpt-4o-mini", verbose=False):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=200
    )
    if verbose:
        print(f"    FINISH REASON: {response.choices[0].finish_reason}")
    return response.choices[0].message.content.strip()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"EVAL STARTING")
    print(f"{'='*60}")
    print(f"ChromaDB: {args.db_dir}")
    print(f"Collection: {args.collection_name}")
    print(f"Embedding model: {args.embed_model}")
    print(f"LLM model: {args.llm_model}")
    print(f"Metadata usage: {args.use_metadata}")

    global CHROMA_DIR, CHUNKS_DIR, COLLECTION_NAME, OLLAMA_URL, LLM_MODEL, TEST_SET_PATH
    global embed_model, client, collection

    CHROMA_DIR       = args.db_dir
    CHUNKS_DIR       = args.chunks_dir
    COLLECTION_NAME  = args.collection_name
    OLLAMA_URL       = args.ollama_url
    LLM_MODEL        = args.llm_model
    TEST_SET_PATH    = args.test_set

    print(f"\nLoading embedding model: {args.embed_model}")
    embed_model = SentenceTransformer(args.embed_model)
    print(f"Embedding dimension: {embed_model.get_sentence_embedding_dimension()}")

    print(f"\nConnecting to ChromaDB: {CHROMA_DIR}")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Collection found: {COLLECTION_NAME}")

    results = run_eval(use_metadata=args.use_metadata, test_limit=args.test_limit, args=args)

    if args.output:
        output_file = args.output
    else:
        meta_str    = "with_metadata" if args.use_metadata else "no_metadata"
        model_name  = args.embed_model.split('/')[-1].replace('-', '_')
        output_file = f"eval_{model_name}_{meta_str}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nOutputs saved to: {output_file}")
    print(f"\n{'='*60}")
    print(f"EVAL Completed")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    main()