#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a ChromaDB index from locally reconstructed chunk JSON files.

This script expects chunk files that contain transcript text. These chunk files are
not part of the public artifact and should be created locally from lawfully
obtained transcripts/media.
"""

import os
import sys
import json
import argparse
import chromadb
from sentence_transformers import SentenceTransformer

def ts_to_sec(ts: str) -> int:
    """
    Convert 'HH:MM:SS' (or 'MM:SS') timestamp string to integer seconds.
    Safe against None/empty values.
    """
    if not ts:
        return 0
    ts = str(ts).strip().replace("::", ":")
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(float(s))
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(float(s))
    except Exception:
        return 0
    return 0


# Add project root 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Default embedding model 
DEFAULT_EMBED_MODEL = 'BAAI/bge-large-en-v1.5'

def main():
    parser = argparse.ArgumentParser(description='Setup ChromaDB with chunks')
    parser.add_argument('--chunk-dir', type=str, required=True, 
                       help='Directory containing chunk JSON files')
    parser.add_argument('--db-dir', type=str, default='data/chroma_db',
                       help='ChromaDB directory (default: data/chroma_db)')
    parser.add_argument('--collection-name', type=str, default='meetings',
                       help='Collection name (default: meetings)')
    parser.add_argument('--model-name', type=str, default=DEFAULT_EMBED_MODEL,
                       help=f'Embedding model name (default: {DEFAULT_EMBED_MODEL})')
    parser.add_argument('--force', action='store_true',
                       help='Force recreate even if exists')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("SETUP RAG STARTING")
    print(f"{'='*60}")
    print(f"Chunk: {args.chunk_dir}")
    print(f"ChromaDB: {args.db_dir}")
    print(f"Collection name: {args.collection_name}")
    print(f"Embedding model: {args.model_name}")
    
    # Create ChromaDB client 
    client = chromadb.PersistentClient(path=args.db_dir)
    
    # (if exists) Delete the old collection ( with --force ile)
    try:
        existing = client.list_collections()
        for col in existing:
            if col.name == args.collection_name:
                if args.force:
                    client.delete_collection(args.collection_name)
                    print(f"Deleted existing collection: {args.collection_name}")
                else:
                    print(f"Collection already exists: {args.collection_name}")
                    print(f" Use --force to overwrite")
                    return
    except:
        pass
    
    # Create new collection 
    collection = client.create_collection(name=args.collection_name)
    print(f"Collection created: {args.collection_name}")
    
    # Embedding model
    print(f"\nLoading embedding model : {args.model_name}")
    model = SentenceTransformer(args.model_name)
    print(f"Model loaded (size: {model.get_sentence_embedding_dimension()})")
    
    total_chunks = 0
    total_files = 0
    
    # Process all chunk files 
    for filename in os.listdir(args.chunk_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(args.chunk_dir, filename)
        print(f"\n Processing: {filename}")
        
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        
        source = data.get('source_file') or data.get('original_file') or data.get('original') or filename
        chunks = data.get('chunks', [])
        
        if not chunks:
            print(f"Warning: {filename} not found chunk")
            continue
        
        docs, ids, metas = [], [], []
        for chunk in chunks:
            chunk_id = f"{source}_chunk_{chunk['chunk_id']}"
            docs.append(chunk['content'])
            ids.append(chunk_id)
            metas.append({
                "source_file": source,
                "chunk_id": chunk.get("chunk_id"),
                "start_sec": ts_to_sec(chunk.get("start_time", "00:00:00")),
                "end_sec": ts_to_sec(chunk.get("end_time", "00:00:00")),
                "topics": json.dumps(chunk.get("topics_in_chunk", []))
            })
        
        print(f"   {len(docs)} chunks: creating embeddings...")
        
        # Create embeddings  (with progress bar)
        embeddings = model.encode(docs, show_progress_bar=True).tolist()
        
        # Add into ChromaDB
        collection.add(
            documents=docs, 
            embeddings=embeddings, 
            ids=ids, 
            metadatas=metas
        )
        
        print(f" {len(chunks)} chunks uploaded")
        total_chunks += len(chunks)
        total_files += 1
    
    print(f"\n{'='*60}")
    print(f"Statistics")
    print(f"{'='*60}")
    print(f"Processed files: {total_files}")
    print(f"Total chunks : {total_chunks}")
    print(f"Collection: {args.collection_name}")
    print(f"Embedding model: {args.model_name}")
    print(f"ChromaDB: {args.db_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()