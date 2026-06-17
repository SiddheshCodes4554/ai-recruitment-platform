# Career Intelligence Engine

The Career Intelligence Engine extracts recruiter-realistic evidence of candidate technical expertise, project achievements, and execution depth from raw profile history text blocks.

## Evidence Evaluation

Unlike simple keyword matching, the engine parses work experience timelines looking for contextual indications of quality:

1. **Title Alignment**: Boosts score if matching terms appear in recent job titles or headlines (indicating leadership or primary ownership).
2. **Action Verbs**: Verifies the presence of action verbs (e.g. *designed, shipped, optimized, architected, migrated*) surrounding technical terms.
3. **Quantitative Metrics**: Rewards descriptions containing numerical metrics (e.g. *improved latency by 20%, scaled to 10M users, reduced cost by 30%*), signifying real-world business impact.

---

## Keyword Stuffing Penalties

Candidates frequently list highly sought-after technologies in their skills list without actually using them in their work histories. To address this, the engine detects and penalizes keyword stuffing.

### Logic Flow
1. **Define Target Technologies**: Filters the candidate's skills list to check only search-relevant keywords (e.g. *Elasticsearch, FAISS, PyTorch, HNSW*).
2. **Scan Work Timeline**: Searches for the skill term in the candidate's headline, summary, and career history description blocks.
3. **Calculate Stuffing Ratio**:
   \[\text{StuffingRatio} = \frac{\text{StuffedSkillsCount}}{\text{TotalSearchSkillsCount}}\]
4. **Apply Penalty**:
   - **No stuffed skills**: \(0\%\) penalty.
   - **Stuffing Ratio** \(\le 30\%\): \(5\%\) penalty.
   - **Stuffing Ratio** \(\le 60\%\): \(10\%\) penalty.
   - **Stuffing Ratio** \(> 60\%\): \(15\%\) penalty.

This multi-tiered threshold penalizes severe stuffing while tolerating minor omissions, preventing unfair disqualifications of strong candidates who fail to document minor details.

---

## Search Engineering Bonuses

To surface elite candidates with deep expertise in search engineering, the engine applies a progressive bonus when multiple core capabilities appear together in the candidate's history descriptions:

- **Core Search Engineering Capabilities**:
  - **Retrieval**: DPR, BM25, Information Retrieval.
  - **Ranking**: Learning to Rank, MRR, NDCG, MAP.
  - **Recommendation**: Rec-sys, Collaborative Filtering.
  - **Search Infra**: Elasticsearch, OpenSearch.
  - **Embeddings**: Sentence-transformers, BERT.
  - **Vector DB**: FAISS, Qdrant, Pinecone.
  - **Production ML**: MLOps, drift detection, scale.

### Progressive Bonus Rules
- **Retrieval + Ranking**: \(+5.0\) points.
- **Retrieval + Ranking + Production ML**: \(+10.0\) points.
- **Retrieval + Ranking + Embeddings + Evaluation Metrics**: \(+15.0\) points (Elite Search Engineer bonus).

The progressive bonuses are capped at a maximum of **15 points** and are directly added to the search engineering sub-score before combination with the final Career Intelligence score.
