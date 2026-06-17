import os
import sys
from pathlib import Path

# Disable TF to prevent keras/protobuf cross-version conflicts
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import pickle
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from src.config import FAISS_INDEX_PATH, METADATA_CACHE_PATH
from src.preprocessing import CandidateRecord, JDRequirementGraph, ScoreBreakdown
from src.engines import JDGraphParser
from src.ranking import CandidateRanker
from src.explainability import CandidateExplainer
from src.ranking import run_ranking_pipeline
from src.utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Redrob Intelligent Candidate Discovery & Ranking System",
    description="Production-grade search and ranking API using FAISS, SentenceTransformers, and Career Intelligence.",
    version="1.0.0"
)

# Global in-memory cache loaded once on startup
METADATA_CACHE: List[Dict[str, Any]] = []

@app.on_event("startup")
def load_startup_resources():
    """Loads FAISS index and metadata cache into memory on startup."""
    global METADATA_CACHE
    if not METADATA_CACHE_PATH.exists():
        logger.warning(f"Metadata cache not found at {METADATA_CACHE_PATH}. API will return errors until precompute runs.")
        return
        
    try:
        with open(METADATA_CACHE_PATH, "rb") as f:
            METADATA_CACHE = pickle.load(f)
        logger.info(f"Startup: Loaded {len(METADATA_CACHE)} candidate profiles into memory.")
    except Exception as e:
        logger.error(f"Error loading metadata cache: {e}")

class RankRequest(BaseModel):
    job_description: str
    limit: Optional[int] = 100

class RankResponseItem(BaseModel):
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    profile: Dict[str, Any]
    breakdown: ScoreBreakdown

@app.post("/rank", response_model=List[RankResponseItem])
def rank_candidates(request: RankRequest = Body(...)):
    """
    Ranks the candidate pool against the submitted Job Description text.
    Returns the top N candidates with detailed score breakdowns and reasoning.
    """
    if not METADATA_CACHE:
        raise HTTPException(status_code=503, detail="Metadata cache is not loaded yet on server startup.")
        
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")

    try:
        # We run the ranking pipeline using the override text
        # To avoid writing to the default submission file during API calls,
        # we do the ranking steps in-memory
        jd_text = request.job_description
        jd_graph = JDGraphParser.parse_jd(jd_text)
        
        # Load index
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from src.config import SENTENCE_TRANSFORMER_MODEL
        
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cpu")
        
        # Encode JD
        jd_vector = model.encode([jd_text], convert_to_numpy=True, normalize_embeddings=True)
        jd_vector = jd_vector.astype(np.float32)
        
        # Retrieve top 5,000 matches
        k = min(5000, len(METADATA_CACHE))
        distances, indices = index.search(jd_vector, k)
        
        similarities = distances[0]
        candidate_indices = indices[0]
        
        # Load records and run scorer
        retrieved_records = []
        retrieved_similarities = []
        for i, cache_idx in enumerate(candidate_indices):
            if cache_idx == -1:
                continue
            sim = similarities[i]
            meta = METADATA_CACHE[cache_idx]
            record = CandidateRecord.model_validate(meta)
            retrieved_records.append(record)
            retrieved_similarities.append(sim)
            
        # Rank candidates
        ranked_results = CandidateRanker.rank_candidates(retrieved_records, retrieved_similarities, jd_graph)
        
        # Format response
        response = []
        for rank_pos, (record, score, breakdown) in enumerate(ranked_results[:request.limit], 1):
            reasoning = CandidateExplainer.generate_reasoning(record, breakdown)
            response.append(RankResponseItem(
                candidate_id=record.candidate_id,
                rank=rank_pos,
                score=round(score, 4),
                reasoning=reasoning,
                profile=record.profile.model_dump(),
                breakdown=breakdown
            ))
            
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking pipeline error: {str(e)}")

@app.get("/candidates/{candidate_id}", response_model=Dict[str, Any])
def get_candidate(candidate_id: str):
    """Retrieves full profile data for a specific candidate ID from the cache."""
    if not METADATA_CACHE:
        raise HTTPException(status_code=503, detail="Metadata cache is not loaded yet.")
        
    for meta in METADATA_CACHE:
        if meta["candidate_id"] == candidate_id:
            return meta
            
    raise HTTPException(status_code=404, detail=f"Candidate with ID {candidate_id} not found.")

@app.get("/health")
def health_check():
    """System health check endpoint."""
    index_loaded = FAISS_INDEX_PATH.exists()
    cache_loaded = len(METADATA_CACHE) > 0
    return {
        "status": "healthy" if (index_loaded and cache_loaded) else "degraded",
        "faiss_index_exists": index_loaded,
        "metadata_cache_loaded": cache_loaded,
        "total_cached_candidates": len(METADATA_CACHE),
        "os": sys.platform,
        "python_version": sys.version
    }
