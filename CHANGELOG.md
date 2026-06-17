# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-17

### Added
- Created `ExperienceAlignmentScore` evaluating candidate YoE curves against job description preferred ranges.
- Created `SearchEngineeringScore` rewarding candidates showing evidence of search, retrieval, indexing, metrics, and production deployments.
- Implemented `CandidateExplainer` generating personalized, anonymized, natural 1-2 sentence recruiter explanations.
- Added comprehensive unit testing suite (`test_honeypots.py`, `test_ranker.py`, `test_engines.py`) validating scoring, matching, and explainability.

### Changed
- Refactored project directory structure into clean packages (`config`, `preprocessing`, `engines`, `ranking`, `explainability`, `utils`).
- Replaced direct print statements with Python standard logging (`src/utils/logger.py`).
- Centralized all magic numbers, sigmas, weights, and thresholds in `src/config/config.py`.
- Type-hinted and documented all modules with Google-style docstrings.

### Fixed
- Fixed Career Intelligence under-scoring for search candidates.
- Adjusted keyword stuffing penalties to prevent over-penalizing legitimate skills.
- Corrected score compression to increase score separation between ranks.
