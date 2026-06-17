# System Architecture

The Redrob Candidate Discovery & Ranking System is split into a **Pre-computation Phase** (offline) and a **Search & Ranking Phase** (online) to achieve high throughput and low-latency recruiter search over large candidate datasets.

```mermaid
graph TD
    subgraph Pre-computation Phase (Offline)
        Candidates[Data/candidates.jsonl] --> Precompute[scripts/precompute.py]
        Precompute --> FAISS_Index[models/candidate_index.faiss]
        Precompute --> Meta_Cache[models/metadata_cache.pkl]
        Precompute --> Honeypots[models/honeypots_lookup.json]
    end

    subgraph Search & Ranking Phase (Online)
        JD[Job Description DOCX / Input] --> JDGraph[src/engines/jd_graph.py]
        JDGraph --> ParseGraph[JDRequirementGraph]
        
        JD --> Encoder[SentenceTransformer Model]
        Encoder --> JDVector[JD Vector]
        
        JDVector --> Search[FAISS Semantic Search]
        FAISS_Index --> Search
        
        Search --> Retrieve[Top 5,000 Matches]
        Meta_Cache --> Retrieve
        
        Retrieve --> Ranker[src/ranking/ranker.py]
        ParseGraph --> Ranker
        Honeypots --> Ranker
        
        Ranker --> FinalRank[Top 100 Candidates]
        FinalRank --> Explainer[src/explainability/explainer.py]
        FinalRank --> Output[outputs/submission.csv]
    end
```

## Module Directory Structure

```
.
├── Data/                      # Datasets (raw profiles, schemas, templates)
├── docs/                      # Architectural & design documentation
├── models/                    # Binary model artifacts and offline caches
├── outputs/                   # Exported CSV ranking submissions
├── src/                       # Main source code package
│   ├── config/                # Centralized hyperparameters & configurations
│   ├── preprocessing/         # Pydantic schemas and profile encoders
│   ├── engines/               # Recruiter alignment engines
│   │   ├── honeypot_detector.py
│   │   ├── jd_graph.py
│   │   ├── career_intel.py
│   │   └── experience_alignment.py
│   ├── ranking/               # Candidate ranker & pipeline orchestrator
│   ├── explainability/        # Recruiter fact-based natural language generator
│   └── utils/                 # Logging & shared utilities
├── tests/                     # Automated unit tests
├── rank.py                    # Core CLI runner
└── requirements.txt           # Package dependencies
```

## Phase Overview

### 1. Offline Pre-computation Phase
Runs periodically to generate static vector representations of all candidate profiles.
- Parses raw candidate records into validated Pydantic structures.
- Evaluates logical consistency checks via `HoneypotDetector` and builds a rapid-lookup disqualification map.
- Serializes candidate identifiers, locations, availability parameters, and recent timeline histories into a serialized `metadata_cache.pkl` cache for fast online lookup.
- Concatenates parsed profiles into dense text sentences, generates 384-dimensional embeddings using `SentenceTransformer (all-MiniLM-L6-v2)`, and writes them to a `FAISS` index structure.

### 2. Online Search & Ranking Phase
Executes in real-time when a recruiter submits a job description:
- **JD Parsing**: Converts job description text into a structured semantic requirement graph (matching constraints like experience bounds, locations, and skills).
- **Semantic Retrieval**: Encodes the job description text and performs a cosine-similarity vector search against the FAISS index to extract the top 5,000 candidate profiles.
- **Multi-Factor Scoring**: Evaluates candidate matches along five independent axes (Semantic Fit, Graph Constraints, Experience Curve, Technical Career Intelligence, Platform Recruitability).
- **Honeypot Disqualification**: Filters out candidates pre-flagged with fraudulent profile inconsistencies.
- **Ordering & Tie-Breaking**: Sorts matching candidates by final score (descending) and breaks ties deterministically by `candidate_id` ascending.
- **Explainability**: Synthesizes profile timeline events and score breakdowns into a personalized, anonymized recruiter reasoning statement.
