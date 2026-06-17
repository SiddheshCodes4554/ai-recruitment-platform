# Redrob Candidate Discovery & Ranking System

[![Build & Test Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, recruiter-centric candidate ranking and retrieval system designed to discover and rank candidates against a job description. The system scales to evaluate a pool of 100,000 candidates under 35 seconds, prioritizing logical trust, experience alignment, and technical career intelligence.

---

## 📖 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Multi-Stage Recruiter Scoring](#-multi-stage-recruiter-scoring)
3. [Project Directory Layout](#-project-directory-layout)
4. [Installation & Setup](#-installation--setup)
5. [Usage & CLI Reference](#-usage--cli-reference)
6. [Testing & Verification](#-testing--verification)
7. [Detailed Documentation](#-detailed-documentation)

---

## ⚙️ System Architecture

The system utilizes a split **Offline Pre-computation** and **Online Real-time Search** architecture to meet strict low-latency requirements.

```
                                 [candidates.jsonl]
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                 ▼
               [Honeypot Detector]              [Profile Text Builder]
                        │                                 │
                        ▼                                 ▼
             [honeypots_lookup.json]            [SentenceTransformer]
                                                          │
                                                          ▼
             [metadata_cache.pkl] ◄────────────── [FAISS Index]
                      │                                   │
                      └─────────────────┬─────────────────┘
                                        │ (Baked into Docker Image)
   ─────────────────────────────────────┼─────────────────────────────────────
                                        ▼ (Online Phase)
  [Job Description] ──► [JD Graph] ──► [FAISS Cosine Search] (Top 5,000)
                                        │
                                        ▼
                                [Scoring Engine]
                       (Tech Fit + Career Fit + Recruitability)
                                        │
                                        ▼
                            [Honeypot Safety Lock]
                                        │
                                        ▼
                           [Deterministic Sorting]
                                        │
                                        ▼
                           [Reasoning Generator] ──► [submission.csv]
```

---

## 🎯 Multi-Stage Recruiter Scoring

The final composite ranking score evaluates candidates along multiple dimensions to replicate realistic recruiter preferences:

\[\text{RawScore} = 0.40 \cdot \text{TechnicalFit} + 0.30 \cdot \text{CareerFit} + 0.20 \cdot \text{Recruitability} + 0.10 \cdot \text{Availability}\]

\[\text{FinalScore} = \text{RawScore} \cdot \text{BehavioralModifier}\]

### 1. Score Dimensions
- **Technical Fit (Weight: 40%)**: Blend of FAISS semantic similarity and a deterministic requirement graph match.
- **Career Fit (Weight: 30%)**: Evaluates candidate career quality (tenure stability and company tiers) and Technical Career Intelligence (verified action verbs, quantitative metrics, and progressive search engineering bonuses).
- **Recruitability (Weight: 20%)**: Evaluates active platform signals (open-to-work flag, signup date, profile views, and willingness to relocate).
- **Availability (Weight: 10%)**: Stated notice period alignment (full score for sub-30 days notice).

### 2. Double-Sigmoid Experience Alignment
The system incorporates a smooth double-sigmoid experience alignment score centered around the JD's preferred range. Candidates in the ideal range (\(6.0 - 8.0\) years) receive a maximum score of \(100\), while those outside are smoothly penalized without harsh step-function drop-offs.

---

## 📂 Project Directory Layout

```
.
├── Data/                           # Raw datasets (candidates, templates, and specifications)
├── docs/                           # Architectural & methodology documentation
│   ├── architecture.md             # Systems architecture and workflow overview
│   ├── ranking-methodology.md      # Detailed composite scoring and Experience curves
│   ├── career-intelligence.md      # Career Intelligence scoring and keyword stuffing
│   ├── trust-engine.md             # Anomaly/Honeypot detection rules
│   └── deployment.md               # API, UI, and Docker deployment manuals
├── models/                         # Serialized offline models and FAISS indexes
├── outputs/                        # Output CSV ranking files
├── scripts/                        # Precomputation and submission validation tools
├── src/                            # Primary package source files
│   ├── config/                     # Centralized hyperparameters & paths
│   ├── preprocessing/              # Pydantic validation schemas & preprocessors
│   ├── engines/                    # Core recruiter logic engines (Honeypot, JD Graph, CI)
│   ├── ranking/                    # Scorer ranker & pipeline orchestrator
│   ├── explainability/             # Recruiter fact-based natural language generator
│   ├── utils/                      # Standard python logging config
│   ├── api/                        # FastAPI search & rank service endpoints
│   └── ui/                         # Streamlit dashboard server
├── tests/                          # Comprehensive unit test suite
├── rank.py                         # Root-level CLI pipeline execution script
├── requirements.txt                # System dependency packages
└── pyproject.toml                  # Python package configuration
```

---

## 🚀 Installation & Setup

### Local Installation
1. Clone the repository and navigate into the workspace.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install project dependencies and development packages:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

### Docker Deployment (FastAPI + Streamlit)
Launch the microservice stack (FastAPI backend on port `8000`, Streamlit Recruiter Dashboard on port `8501`):
```bash
docker-compose up --build -d
```

---

## 🛠️ Usage & CLI Reference

### Running the Candidate Ranking
Run the end-to-end ranking pipeline over the candidate database to generate the final `submission.csv`:
```bash
python rank.py --candidates Data/candidates.jsonl --out outputs/submission.csv
```

### Running the Offline Pre-computation
Generate vector embeddings and metadata caches for new candidates:
```bash
python scripts/precompute.py Data/candidates.jsonl
```

---

## 🧪 Testing & Verification

Run the full automated test suite containing validation schemas, scoring curves, and explainability mocks:
```bash
python -m pytest
```

---

## 📚 Detailed Documentation

For deep dives into the underlying mathematical formulas, anti-fraud triggers, and system configuration, review the documents inside the [docs/](file:///d:/Projects/Redrob%20Hackathon/docs/) directory:
- [System Architecture](file:///d:/Projects/Redrob%20Hackathon/docs/architecture.md)
- [Ranking Methodology & Experience Curves](file:///d:/Projects/Redrob%20Hackathon/docs/ranking-methodology.md)
- [Career Intelligence & Stuffing Penalties](file:///d:/Projects/Redrob%20Hackathon/docs/career-intelligence.md)
- [Trust Engine & Honeypot Detection](file:///d:/Projects/Redrob%20Hackathon/docs/trust-engine.md)
- [Deployment & Run Guide](file:///d:/Projects/Redrob%20Hackathon/docs/deployment.md)
- [Hackathon Executive Summary](file:///d:/Projects/Redrob%20Hackathon/EXECUTIVE_SUMMARY.md)
