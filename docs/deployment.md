# Deployment & Run Guide

The Redrob Candidate Discovery & Ranking System is fully package-configured and supports local CLI execution, Docker deployments, FastAPI backend endpoints, and Streamlit frontend dashboards.

---

## 1. CLI Usage

Run the candidate ranking directly from your terminal:

```bash
python rank.py --candidates Data/candidates.jsonl --out outputs/submission.csv
```

- `--candidates`: Path to the candidate profile dataset in JSON Lines format.
- `--out`: Path where the ranked top 100 CSV will be written.

---

## 2. Running Local Servers

### A. FastAPI Backend
The FastAPI server provides production-grade endpoints for search, ranking, and candidate detail retrieval.

Start the API server:
```bash
uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload
```

- Interactive swagger documentation is available at: `http://localhost:8000/docs`
- Endpoint `/rank` accepts a JSON body containing a job description and return limit, and responds with ranked candidate scores and natural justifications.

### B. Streamlit Recruiter Dashboard
The dashboard provides a premium UI for visualizing requirement graphs, scoring progress bars, keyword stuffing alerts, and detailed recruiter explanations.

Start the Streamlit application:
```bash
streamlit run src/ui/app.py
```
Access the application at `http://localhost:8501`.

---

## 3. Docker Deployment

The application is dockerized using a multi-phase configuration to run in isolated containerized environments.

### Build and Run with Docker Compose
To launch both the FastAPI backend and Streamlit dashboard:

```bash
docker-compose up --build
```

- FastAPI Server: `http://localhost:8000`
- Streamlit UI Dashboard: `http://localhost:8501`

### Running Offline Precompute inside Docker
To rebuild the FAISS index and cache artifacts inside a Docker container:
```bash
docker build -t redrob-precompute -f Dockerfile .
docker run -v $(pwd)/models:/app/models redrob-precompute python scripts/precompute.py Data/candidates.jsonl
```
This mounts the local `models/` directory so the built index and metadata caches are saved back to the host file system.
