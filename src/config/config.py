import os
from pathlib import Path
from typing import Dict, List, Set

__all__ = [
    "WORKSPACE_DIR",
    "SRC_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "CANDIDATES_JSONL_PATH",
    "SAMPLE_CANDIDATES_PATH",
    "JOB_DESCRIPTION_DOCX_PATH",
    "FAISS_INDEX_PATH",
    "METADATA_CACHE_PATH",
    "EMBEDDINGS_PATH",
    "HONEYPOT_LOOKUP_PATH",
    "SENTENCE_TRANSFORMER_MODEL",
    "EMBEDDING_DIM",
    "WEIGHT_TECH_FIT",
    "WEIGHT_CAREER_FIT",
    "WEIGHT_RECRUITABILITY",
    "WEIGHT_TRUST",
    "TECH_FIT_SEMANTIC_WEIGHT",
    "TECH_FIT_INTEL_WEIGHT",
    "CAREER_FIT_GRAPH_WEIGHT",
    "CAREER_FIT_QUALITY_WEIGHT",
    "CAREER_FIT_ALIGNMENT_WEIGHT",
    "IDEAL_EXPERIENCE_YEARS",
    "EXPERIENCE_SIGMA",
    "EXP_ALIGN_SIGMA_LEFT",
    "EXP_ALIGN_SIGMA_RIGHT",
    "INTEL_BASE_SCORE",
    "INTEL_TITLE_BOOST",
    "INTEL_ACTION_VERB_BOOST",
    "INTEL_METRIC_BOOST",
    "INTEL_DIVISOR",
    "STUFFING_TIER_1_RATIO",
    "STUFFING_TIER_1_PENALTY",
    "STUFFING_TIER_2_RATIO",
    "STUFFING_TIER_2_PENALTY",
    "STUFFING_MAX_PENALTY",
    "COMPANY_TIERS",
    "DEFAULT_COMPANY_POINTS",
    "IT_SERVICES_COMPANIES",
    "CORE_CONCEPTS_KEYWORDS",
    "ACTION_VERBS",
    "PREFERRED_LOCATIONS"
]

# Base Directories
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = WORKSPACE_DIR / "src"
DATA_DIR = WORKSPACE_DIR / "data"
MODELS_DIR = WORKSPACE_DIR / "models"
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"

# Data Files
CANDIDATES_JSONL_PATH = DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES_PATH = DATA_DIR / "sample" / "sample_candidates.json"
JOB_DESCRIPTION_DOCX_PATH = DATA_DIR / "job_description.docx"

# Generated Cache Files (For Online Phase Speedup)
FAISS_INDEX_PATH = MODELS_DIR / "candidate_index.faiss"
METADATA_CACHE_PATH = MODELS_DIR / "metadata_cache.pkl"
EMBEDDINGS_PATH = MODELS_DIR / "candidate_embeddings.npy"
HONEYPOT_LOOKUP_PATH = MODELS_DIR / "honeypots_lookup.json"

# Model Parameters
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Candidate Scoring Outer Weights
WEIGHT_TECH_FIT = 0.40
WEIGHT_CAREER_FIT = 0.30
WEIGHT_RECRUITABILITY = 0.20
WEIGHT_TRUST = 0.10

# Sub-score Inner Weights
TECH_FIT_SEMANTIC_WEIGHT = 0.30
TECH_FIT_INTEL_WEIGHT = 0.70

CAREER_FIT_GRAPH_WEIGHT = 0.60
CAREER_FIT_QUALITY_WEIGHT = 0.25
CAREER_FIT_ALIGNMENT_WEIGHT = 0.15

# Career Quality parameters
IDEAL_EXPERIENCE_YEARS = 7.0
EXPERIENCE_SIGMA = 1.5

# Experience Alignment curve sigmas
EXP_ALIGN_SIGMA_LEFT = 1.75
EXP_ALIGN_SIGMA_RIGHT = 2.45

# Career Intelligence Engine Weights
INTEL_BASE_SCORE = 4.0
INTEL_TITLE_BOOST = 3.0
INTEL_ACTION_VERB_BOOST = 2.0
INTEL_METRIC_BOOST = 2.0
INTEL_DIVISOR = 50.0

# Stuffing Penalty Tiers
STUFFING_TIER_1_RATIO = 0.25
STUFFING_TIER_1_PENALTY = 0.05
STUFFING_TIER_2_RATIO = 0.60
STUFFING_TIER_2_PENALTY = 0.10
STUFFING_MAX_PENALTY = 0.15

# Company Classification & Tiers (Points out of 100)
COMPANY_TIERS: Dict[str, float] = {
    # Product & High-Growth Tech (Tier 1/2) - 100 points
    "Hooli": 100.0,
    "Pied Piper": 100.0,
    "Initech": 100.0,
    "Swiggy": 100.0,
    "Razorpay": 100.0,
    "CRED": 100.0,
    "Zomato": 100.0,
    "Flipkart": 100.0,
    "Zoho": 100.0,
    "Ola": 100.0,
    "InMobi": 100.0,
    "Nykaa": 100.0,
    "Meesho": 100.0,
    "Vedantu": 100.0,
    "BYJU'S": 100.0,
    "Wayne Enterprises": 100.0,
    "Stark Industries": 100.0,
    
    # IT Services & Consulting (Tier 3) - 20 points
    "Infosys": 20.0,
    "Wipro": 20.0,
    "TCS": 20.0,
    "Accenture": 20.0,
    "Capgemini": 20.0,
    "Cognizant": 20.0,
    "HCL": 20.0,
    "Mindtree": 20.0,
    "Tech Mahindra": 20.0,
    "Mphasis": 20.0,
    
    # Other / Fictional Traditional Companies - 50 points
    "Acme Corp": 50.0,
    "Globex Inc": 50.0,
    "Dunder Mifflin": 50.0,
}

# Default points for unknown companies
DEFAULT_COMPANY_POINTS = 50.0

# IT Services companies that are heavily penalized if candidate's entire career is spent there
IT_SERVICES_COMPANIES: Set[str] = {
    "Infosys", "Wipro", "TCS", "Accenture", "Capgemini", 
    "Cognizant", "HCL", "Mindtree", "Tech Mahindra", "Mphasis"
}

# Core Search/ML Keywords for the Career Intelligence Engine
CORE_CONCEPTS_KEYWORDS: Dict[str, List[str]] = {
    "retrieval": [
        "retrieval", "vector search", "dense retrieval", "bm25", "hybrid search", 
        "information retrieval", "vector database", "faiss", "milvus", "qdrant", 
        "pinecone", "elasticsearch", "opensearch", "solr", "lucene"
    ],
    "ranking": [
        "ranking", "learning to rank", "ltr", "ndcg", "mrr", "map", "re-ranking", 
        "re-rank", "cross-encoder", "relevance score"
    ],
    "recommendation": [
        "recommendation", "rec-sys", "collaborative filtering", "personalization", 
        "matrix factorization", "recommender"
    ],
    "semantic_search": [
        "semantic search", "semantic match", "neural search", "dense passage retrieval", "dpr"
    ],
    "embeddings": [
        "embedding", "sentence transformer", "bge", "cohere embedding", "word2vec", 
        "sentence-transformer"
    ],
    "experimentation": [
        "experimentation", "ab test", "a/b test", "multi-armed bandit", "hypothesis test"
    ],
    "ab_testing": [
        "a/b testing", "split test", "online experiment"
    ],
    "production_ml": [
        "production", "monitoring", "latency", "scale", "docker", "kubernetes", 
        "mlops", "inference optimization", "model drift", "drift detection"
    ]
}

# Strong recruiter-preferred action verbs
ACTION_VERBS: Set[str] = {
    "built", "shipped", "designed", "implemented", "optimized", "scaled", 
    "deployed", "architected", "engineered", "created", "developed", "led",
    "improved", "reduced", "increased", "launched"
}

# Target cities/locations preferred by the JD
PREFERRED_LOCATIONS: Set[str] = {
    "pune", "noida", "hyderabad", "mumbai", "delhi ncr", "noida-preferred", "pune-preferred", "india"
}
