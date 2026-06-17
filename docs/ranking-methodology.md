# Ranking Methodology

The candidate discovery system ranks matching candidates by evaluating five core axes that represent recruiter-realistic alignment.

## Composite Scoring Formula

The final candidate ranking score is a composite combination of four base scores, adjusted by a behavioral modifier:

\[\text{RawScore} = W_t \cdot \text{TechnicalFit} + W_c \cdot \text{CareerFit} + W_r \cdot \text{Recruitability} + W_a \cdot \text{Availability}\]

\[\text{FinalScore} = \text{RawScore} \cdot \text{BehavioralModifier}\]

### 1. Weights
Weights are centralized in `src/config/config.py` and are strictly calibrated to reflect standard recruiting priorities:
- **Technical Fit (Weight: 40%)**: Combination of Semantic similarity and Requirement graph match.
  - \(\text{TechnicalFit} = 0.4 \cdot \text{SemanticScore} + 0.6 \cdot \text{GraphMatchScore}\)
- **Career Fit (Weight: 30%)**: Combination of Career Quality and Technical Evidence.
  - \(\text{CareerFit} = 0.5 \cdot \text{CareerQualityScore} + 0.5 \cdot \text{CareerIntelligenceScore}\)
- **Recruitability (Weight: 20%)**: Evaluates candidate active status, profile views, and willingness to relocate.
- **Availability (Weight: 10%)**: Evaluates notice period alignment.
  - \(100.0\) for sub-30 day notice, scaling down to \(10.0\) for long notice periods.

### 2. Behavioral Modifier (Scale: 0.95 - 1.05)
A multiplier derived from platform engagement signals:
- **Signup Date Decay**: Boosts newer signups while retaining baseline scores for older profiles.
- **Recruiter Engagement Rate**: Multiplier based on recruiter response rates and response times.
- **Willingness to Relocate**: Bonus if the candidate matches preferred locations or is willing to relocate.

---

## Experience Alignment Score

Recruiters strongly consider experience length alignment. The dedicated `ExperienceAlignmentScore` ensures candidates with years of experience (YoE) outside the preferred bounds are naturally deprioritized, while keeping those within the ideal range at the top.

```
       100 |                 * * * * *
           |               *           *
           |             *               *
        50 |           *                   *
           |         *                       *
           |       *                           *
         0 +-----*-------------------------------*----->
                 3       5       6   8   9      12
                         <-- Preferred Range -->
                                 <- Ideal ->
```

### Mathematical Formulation
The score curve uses a smooth double-sigmoid distribution centered around the target range bounds:

- **Inside the Ideal Range** (\(6.0 \le \text{YoE} \le 8.0\)):
  - \(\text{Score} = 100.0\)
- **Below Ideal Range** (\(\text{YoE} < 6.0\)):
  - \(\text{Score} = 100.0 \cdot \exp\left(-\frac{(6.0 - \text{YoE})^2}{2 \cdot \sigma_{\text{low}}^2}\right)\)
  - Where \(\sigma_{\text{low}} = 1.6\)
- **Above Ideal Range** (\(\text{YoE} > 8.0\)):
  - \(\text{Score} = 100.0 \cdot \exp\left(-\frac{(\text{YoE} - 8.0)^2}{2 \cdot \sigma_{\text{high}}^2}\right)\)
  - Where \(\sigma_{\text{high}} = 2.4\)

This asymmetrical curve applies a stricter penalty to under-experienced candidates compared to over-experienced candidates, aligning with typical hiring manager behavior.

---

## Tie-Breaking Logic
When two candidates receive identical composite scores, the system resolves the tie deterministically:
1. Sort by `score` descending.
2. If scores are equal, sort by `candidate_id` ascending (lexicographically).
This ensures reproducible, zero-drift results across execution runs.
