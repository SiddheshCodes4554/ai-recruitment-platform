# Trust & Honeypot Engine

To maintain high data quality and protect recruiters from fraudulent applications, the system implements a strict, offline **Trust & Honeypot Engine**.

## Honeypot Detection Rules

The engine detects logical contradictions in candidate profiles that are physically or chronologically impossible:

### 1. Chronological Discrepancies
- **Claimed vs. Actual Experience**: Compares the candidate's total claimed years of experience (`profile.years_of_experience`) against the actual sum of durations of their work history entries (`career_history.duration_months / 12`).
  - Flagged if:
    \[\text{profile.years\_of\_experience} - \text{actual\_duration\_years} > 3.0\]
  - This prevents fake candidates from claiming 10+ years of experience while having only 1-2 years of actual work details.

### 2. Education vs. Work Timelines
- Identifies if a candidate completed school *after* their latest work history began, or holds roles starting years before their stated birth or university enrollment ranges, suggesting fabricated histories.

---

## Disqualification Mechanism

If a candidate record fails any of the honeypot detection rules:
1. **Flagged offline**: The pre-computation pipeline flags the candidate and writes the identifier along with the exact mismatch reasons into `models/honeypots_lookup.json`.
2. **Immediate Disqualification**: During the online ranking phase, the ranker does not simply penalize the candidate's score. Instead, they are completely excluded from the candidate retrieval pool.
3. **Audit Log**: The reason for disqualification is stored in the lookup map for admin review (e.g. *"Years of experience mismatch: claimed 11.5, actual work history sum is 2.3 years"*).

This approach ensures zero recruiter exposure to logical honeypots, maintaining high trust in the platform.
