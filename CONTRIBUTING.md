# Contributing to Redrob Candidate Ranker

Thank you for your interest in contributing! This document details the guidelines and workflows for developing and contributing improvements to the Redrob Intelligent Candidate Discovery & Ranking System.

## Codebase Architecture

The project is structured logically as follows:
- `src/config/`: Centralized parameters, thresholds, model settings, and paths.
- `src/preprocessing/`: Candidate data schema definitions (Pydantic models) and text representation preprocessors.
- `src/engines/`: Recruiter-centric evaluation engines:
  - `honeypot_detector.py`: Detects profile inconsistencies (e.g. claimed vs. actual experience).
  - `jd_graph.py`: Parses job descriptions into requirement graphs.
  - `career_intel.py`: Scores candidate career evidence, including keyword stuffing checks and search engineering bonuses.
  - `experience_alignment.py`: Evaluates experience length curves against JD bounds.
- `src/ranking/`: The ranker and execution pipeline, driving retrieved candidate matches.
- `src/explainability/`: Explainability model producing fact-based, natural justifications.
- `tests/`: Automated unit tests verifying logical engines and the ranker.

## Development Setup

1. **Clone the Repository** and open it in your development workspace.
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Development Guidelines

- **Code Quality**: Ensure all code is cleanly type-hinted and uses Google-style docstrings.
- **Scoring Integrity**: The candidate ranking scoring logic is strictly calibrated. Do not modify scoring weights, sigmas, or curve parameters without verified regression testing and alignment.
- **Imports**: Always use clean, absolute package imports (e.g., `from src.config import ...`).

## Running Tests

Before submitting any code changes, ensure all tests pass:
```bash
python -m pytest
```

## Submitting Pull Requests

1. Fork the repository and create your branch from `main`.
2. Add comprehensive unit tests for any new features or bug fixes.
3. Write clear commit messages detailing *what* and *why* changes were made.
4. Open a Pull Request referencing the related issue.
