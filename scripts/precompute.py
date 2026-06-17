"""
Offline Candidate Pre-computation Pipeline.

Loads candidate profiles, extracts metadata, flags anomalies (honeypots),
encodes profiles to semantic vectors, and builds a FAISS index.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json
import pickle
import time
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from src.config.config import (
    CANDIDATES_JSONL_PATH,
    FAISS_INDEX_PATH,
    METADATA_CACHE_PATH,
    EMBEDDINGS_PATH,
    HONEYPOT_LOOKUP_PATH,
    SENTENCE_TRANSFORMER_MODEL,
    EMBEDDING_DIM
)
from src.preprocessing.data_models import CandidateRecord
from src.preprocessing.preprocessor import ProfilePreprocessor
from src.engines.honeypot_detector import HoneypotDetector
from src.utils.logger import get_logger

__all__ = ["precompute_pipeline"]

logger = get_logger("precompute")

def precompute_pipeline(candidates_path: Path = None) -> None:
    """Offline pre-computation pipeline.
    
    Loads the candidate pool, extracts metadata caches, flags logical honeypots,
    generates semantic profile text embeddings using SentenceTransformers, and constructs
    a normalized FAISS IndexFlatIP for cosine similarity search.
    
    Args:
        candidates_path (Path, optional): Overriding candidates file path.
    """
    start_time = time.time()
    logger.info("=== Starting Offline Candidate Pre-computation Pipeline ===")
    
    input_path = candidates_path if candidates_path else CANDIDATES_JSONL_PATH
    
    if not input_path.exists():
        raise FileNotFoundError(f"Candidate dataset not found at {input_path}")
    
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading SentenceTransformer model: {SENTENCE_TRANSFORMER_MODEL}...")
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cpu")
    
    logger.info(f"Reading candidates from {input_path}...")
    
    profile_texts: List[str] = []
    metadata_cache: List[Dict[str, Any]] = []
    honeypots_lookup: Dict[str, List[str]] = {}
    
    total_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            data = json.loads(line)
            record = CandidateRecord(**data)
            cid = record.candidate_id
            
            is_trap, reasons = HoneypotDetector.is_honeypot(record)
            if is_trap:
                honeypots_lookup[cid] = reasons
                
            meta = ProfilePreprocessor.extract_metadata_cache(record)
            metadata_cache.append(meta)
            
            profile_text = ProfilePreprocessor.build_text_representation(record)
            profile_texts.append(profile_text)
            
            total_count += 1
            if total_count % 10000 == 0:
                logger.info(f"  Loaded and parsed {total_count} profiles...")
                
    logger.info(f"Completed loading dataset. Total Candidates: {total_count}")
    logger.info(f"Flagged {len(honeypots_lookup)} logical honeypots ({len(honeypots_lookup)/total_count*100:.2f}% of dataset).")

    logger.info(f"Generating semantic embeddings for {total_count} candidates...")
    logger.info("This runs on CPU and takes about 12-15 minutes on a standard developer machine...")
    
    batch_size = 256
    
    embeddings = model.encode(
        profile_texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    embeddings = embeddings.astype(np.float32)
    logger.info(f"Finished generating embeddings. Shape: {embeddings.shape}")

    logger.info(f"Saving embeddings array to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings)

    logger.info(f"Saving metadata cache list to {METADATA_CACHE_PATH}...")
    with open(METADATA_CACHE_PATH, "wb") as cache_file:
        pickle.dump(metadata_cache, cache_file)

    logger.info(f"Saving honeypots lookup map to {HONEYPOT_LOOKUP_PATH}...")
    with open(HONEYPOT_LOOKUP_PATH, "w", encoding="utf-8") as hp_file:
        json.dump(honeypots_lookup, hp_file, indent=2)

    logger.info("Building FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    
    logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    duration = time.time() - start_time
    logger.info("=== Pre-computation Pipeline Completed Successfully ===")
    logger.info(f"Total time elapsed: {duration/60:.2f} minutes")
    logger.info(f"FAISS Index Location: {FAISS_INDEX_PATH}")
    logger.info(f"Metadata Cache Location: {METADATA_CACHE_PATH}")
    logger.info(f"Numpy Embeddings Location: {EMBEDDINGS_PATH}")
    logger.info(f"Honeypot Lookup Location: {HONEYPOT_LOOKUP_PATH}")

if __name__ == "__main__":
    import sys
    path_arg = None
    if len(sys.argv) > 1:
        path_arg = Path(sys.argv[1])
    precompute_pipeline(path_arg)
