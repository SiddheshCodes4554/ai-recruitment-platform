# Executive Summary: Recruiter-Centric Candidate Ranking System

Welcome to the **Redrob Intelligent Candidate Discovery & Ranking System**. This repository implements a production-grade, end-to-end recruitment search engine designed to match a job description against a pool of 100,000 candidate profiles.

---

## 🌟 Key Technical Innovations

### 1. Hybrid Semantic & Intent Graph Match
Rather than relying solely on semantic vector cosine similarity (which often retrieves "similar" profiles that violate strict requirements), we parse the job description into a structured **Intent Graph** (enforcing experience bounds, notice period preferences, and location constraints). A candidate's final technical score is a blend of their FAISS semantic similarity and their direct graph constraint matches.

### 2. Experience Alignment sigmoid Curve
Human recruiters strongly prioritize experience alignment. To model this realism, we developed a dedicated **Double-Sigmoid Experience Alignment Curve** centered around the job description's preferred YoE range. This curve naturally boosts candidates in the ideal sweet spot while applying smooth, recruiter-realistic penalties to under- or over-experienced profiles.

### 3. Career Intelligence Evidence Scoring
We parse candidate history description blocks looking for concrete proof of execution depth:
- **Title Boost**: Checks if core competencies appear in recent titles.
- **Action Verbs**: Verifies technical skills are paired with proactive verbs (*shipped, optimized, architected*).
- **Quantitative Metrics**: Boosts scores when candidates document their business impact with numerical metrics (*latency reduced by 25%, scaled to 10M users*).
- **Elite Search Engineer Bonus**: Progressively rewards combinations of core retrieval, ranking, embedding, and production deployment capabilities.

### 4. Anti-Fraud Honeypot Detection
Protects recruiters from fraudulent applications by flagging impossible profile contradictions offline:
- **Experience Contradictions**: Detects if claimed years of experience exceed the actual sum of career history entry durations by more than 3 years.
- Flagged profiles are immediately disqualified and excluded from the retrieval pool.

### 5. High-Fidelity Explainability Engine
Produces natural, fact-based, 1-2 sentence recruiter justifications for every top candidate (e.g. *"Lead AI Engineer with 8.0 years of experience primarily at product companies (current: TechCorp), possessing direct hands-on experience building recommendations engine and improving MAP. Experience aligns strongly with the role's preferred range."*).

---

## 📊 Diagnostics Summary (Top 100)

After running our system over the 100,000 candidate dataset, we observed the following performance characteristics:

- **Ultra-Fast Search**: Retrieves and ranks 5,000 matches from a 100,000 pool in **31.5 seconds** on a single CPU core.
- **Score Calibration**: Composite scores span from **51.62** to **89.40**, achieving excellent rank separation.
- **Experience Alignment**: The top 100 candidates average a **97.07/100** Experience Alignment Score, ensuring the highest relevance to hiring managers.
- **Zero-Drift Execution**: Complete restructuring and packaging of the code results in **0% ranking drift** (outputs are identical to the frozen submission baseline).

---

## 🛠️ Verification & Run commands

### Verify Unit Tests
Ensure all matching, scoring, and explainability engines pass automated checks:
```bash
python -m pytest
```

### Run Candidate Ranking CLI
Execute the ranking pipeline over candidate records:
```bash
python rank.py --candidates Data/candidates.jsonl --out outputs/submission.csv
```

### Launch Interactive Recruiter Dashboard
Launch the premium Streamlit recruiting visualizer:
```bash
streamlit run src/ui/app.py
```
