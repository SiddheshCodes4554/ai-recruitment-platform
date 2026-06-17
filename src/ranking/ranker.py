"""
Candidate Ranker Module.

Aggregates semantic similarity, JD graph matching, career intelligence, career quality,
and platforms signals into a final composite recruiter-centric score.
"""

import math
from datetime import datetime
from typing import Dict, List, Tuple, Any
from src.preprocessing.data_models import CandidateRecord, JDRequirementGraph, ScoreBreakdown, CandidateRankResult
from src.config import (
    COMPANY_TIERS,
    DEFAULT_COMPANY_POINTS,
    IDEAL_EXPERIENCE_YEARS,
    EXPERIENCE_SIGMA,
    WEIGHT_TECH_FIT,
    WEIGHT_CAREER_FIT,
    WEIGHT_RECRUITABILITY,
    WEIGHT_TRUST,
    TECH_FIT_SEMANTIC_WEIGHT,
    TECH_FIT_INTEL_WEIGHT,
    CAREER_FIT_GRAPH_WEIGHT,
    CAREER_FIT_QUALITY_WEIGHT,
    CAREER_FIT_ALIGNMENT_WEIGHT
)
from src.engines.honeypot_detector import HoneypotDetector
from src.engines.jd_graph import JDGraphParser
from src.engines.career_intel import CareerIntelligenceEngine
from src.engines.experience_alignment import ExperienceAlignmentEngine

__all__ = ["CandidateRanker"]

class CandidateRanker:
    """
    Candidate Discovery & Ranking Engine.
    Aggregates semantic search similarity, structured JD constraint matching,
    evidence-based career intelligence, career history quality, and behavioral engagement
    signals into a final recruiter-centric score.
    """

    REFERENCE_DATE = datetime(2026, 6, 17)

    @classmethod
    def calculate_career_quality_score(cls, record: CandidateRecord) -> float:
        """Computes the candidate's career trajectory quality based on company tiers and tenure.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            
        Returns:
            float: Career quality score (0-100).
        """
        # 1. Company Tier Score (S_company)
        total_months = 0
        company_points_sum = 0.0
        
        for job in record.career_history:
            dur = job.duration_months
            comp = job.company
            
            points = COMPANY_TIERS.get(comp, DEFAULT_COMPANY_POINTS)
            company_points_sum += points * dur
            total_months += dur
            
        s_company = company_points_sum / total_months if total_months > 0 else DEFAULT_COMPANY_POINTS

        # 2. Tenure Stability Score (S_tenure)
        num_jobs = len(record.career_history)
        avg_tenure = total_months / num_jobs if num_jobs > 0 else 0.0
        
        if avg_tenure >= 36.0:
            s_tenure = 100.0
        elif avg_tenure >= 24.0:
            s_tenure = 80.0
        elif avg_tenure >= 18.0:
            s_tenure = 60.0
        else:
            s_tenure = 20.0

        # 3. Experience Curve Score (S_exp_curve)
        yoe = record.profile.years_of_experience
        s_exp_curve = math.exp(-((yoe - IDEAL_EXPERIENCE_YEARS) ** 2) / (2 * (EXPERIENCE_SIGMA ** 2))) * 100.0

        # Combine components
        s_quality = (0.50 * s_company) + (0.30 * s_exp_curve) + (0.20 * s_tenure)
        return round(s_quality, 2)

    @classmethod
    def calculate_company_score(cls, record: CandidateRecord) -> float:
        """Computes the candidate's time-weighted average company tier score.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            
        Returns:
            float: Time-weighted company score (0-100).
        """
        total_months = 0
        company_points_sum = 0.0
        
        for job in record.career_history:
            dur = job.duration_months
            comp = job.company
            points = COMPANY_TIERS.get(comp, DEFAULT_COMPANY_POINTS)
            company_points_sum += points * dur
            total_months += dur
            
        return company_points_sum / total_months if total_months > 0 else DEFAULT_COMPANY_POINTS

    @classmethod
    def calculate_recruitability_score(cls, record: CandidateRecord) -> float:
        """Evaluates the candidate's platform activity and availability signals.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            
        Returns:
            float: Recruitability score (0-100).
        """
        signals = record.redrob_signals
        
        # 1. Recruiter Response Rate
        s_response = signals.recruiter_response_rate * 100.0

        # 2. Recruiter Response Time Score (lower is better)
        resp_time = signals.avg_response_time_hours
        if resp_time <= 12.0:
            s_time = 100.0
        elif resp_time <= 24.0:
            s_time = 85.0
        elif resp_time <= 48.0:
            s_time = 60.0
        elif resp_time <= 120.0:
            s_time = 30.0
        else:
            s_time = 10.0

        # 3. Open To Work Flag
        s_open = 100.0 if signals.open_to_work_flag else 50.0

        # 4. Interview Completion Rate
        s_interview = signals.interview_completion_rate * 100.0

        # 5. Last Active recency score
        last_act_str = signals.last_active_date
        try:
            last_act_dt = datetime.strptime(last_act_str, "%Y-%m-%d")
            days_inactive = (cls.REFERENCE_DATE - last_act_dt).days
        except ValueError:
            days_inactive = 180
            
        if days_inactive <= 30:
            s_active = 100.0
        elif days_inactive <= 90:
            s_active = 70.0
        elif days_inactive <= 180:
            s_active = 30.0
        else:
            s_active = 10.0

        # Weighted combination of recruitability signals
        s_recruitability = (
            0.30 * s_response +
            0.20 * s_time +
            0.20 * s_open +
            0.15 * s_interview +
            0.15 * s_active
        )
        return round(s_recruitability, 2)

    @classmethod
    def score_candidate(
        cls, 
        record: CandidateRecord, 
        semantic_similarity: float, 
        jd_graph: JDRequirementGraph
    ) -> Tuple[float, ScoreBreakdown, List[str]]:
        """Computes the complete, multi-stage composite score for a candidate.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            semantic_similarity (float): The semantic match cosine similarity score.
            jd_graph (JDRequirementGraph): The parsed job requirements.
            
        Returns:
            Tuple[float, ScoreBreakdown, List[str]]: The final score, detailed breakdown,
            and honeypot consistency warnings.
        """
        # 1. Honeypot Consistency Check (Trust Score)
        is_trap, honeypot_reasons = HoneypotDetector.is_honeypot(record)
        trust_score, _, _ = HoneypotDetector.evaluate(record)
        
        # 2. Semantic Embedding Score (Scale cosine from [0, 1] to [0, 100])
        semantic_score = max(0.0, min(100.0, semantic_similarity * 100.0))

        # 3. JD Graph Match Score (Domain, tech alignments and constraint matches)
        graph_match_score, _ = JDGraphParser.calculate_match_score(record, jd_graph)

        # 4. Career Intelligence Score (Contextual evidence of shipping ML/search)
        intel_result = CareerIntelligenceEngine.evaluate(record)
        career_intel_score = intel_result.career_intelligence_score

        # 5. Career Quality Score
        career_quality_score = cls.calculate_career_quality_score(record)

        # 5.5. Experience Alignment Score
        exp_score = ExperienceAlignmentEngine.calculate_score(
            yoe=record.profile.years_of_experience,
            min_exp=jd_graph.constraints.min_exp,
            max_exp=jd_graph.constraints.max_exp,
            ideal_min=jd_graph.constraints.ideal_min_exp,
            ideal_max=jd_graph.constraints.ideal_max_exp
        )

        # 6. Recruitability Score (Behavioral signals)
        recruitability_score = cls.calculate_recruitability_score(record)

        # AGGREGATE CORE METRICS (Each in range 0 - 100)
        
        # Technical Fit: lower semantic weight, higher career evidence weight
        tech_fit_score = (TECH_FIT_SEMANTIC_WEIGHT * semantic_score) + (TECH_FIT_INTEL_WEIGHT * career_intel_score)
        
        # Career Fit: stronger JD graph alignment, lower career quality weight, add experience alignment
        career_fit_score = (CAREER_FIT_GRAPH_WEIGHT * graph_match_score) + (CAREER_FIT_QUALITY_WEIGHT * career_quality_score) + (CAREER_FIT_ALIGNMENT_WEIGHT * exp_score)
        
        # Recruitability: Activity and responsiveness
        recruitability_final = recruitability_score
        
        # Trust: Consistency scoring
        trust_final = trust_score

        # CALCULATE COMPOSITE FINAL SCORE
        final_score = (
            WEIGHT_TECH_FIT * tech_fit_score +
            WEIGHT_CAREER_FIT * career_fit_score +
            WEIGHT_RECRUITABILITY * recruitability_final +
            WEIGHT_TRUST * trust_final
        )

        # 7. Synergy Bonus Logic
        synergy_bonus = 0.0
        
        # Check presence of 5 core technical capabilities (score >= 60.0)
        has_retrieval = intel_result.retrieval_score >= 60.0
        has_ranking = intel_result.ranking_score >= 60.0
        has_rec = intel_result.recommendation_score >= 60.0
        has_semantic = (
            intel_result.embeddings_score >= 60.0 or
            intel_result.vector_db_score >= 60.0 or
            intel_result.search_infra_score >= 60.0
        )
        has_prod_ml = intel_result.production_ml_score >= 60.0
        
        core_caps_count = sum([has_retrieval, has_ranking, has_rec, has_semantic, has_prod_ml])
        
        # Technical capabilities synergy bonus:
        if core_caps_count == 3:
            synergy_bonus += 5.0
        elif core_caps_count == 4:
            synergy_bonus += 10.0
        elif core_caps_count >= 5:
            synergy_bonus += 15.0
            
        # Product company boost: +5.0 if s_company >= 70.0 and at least 2 core capabilities
        s_company = cls.calculate_company_score(record)
        if s_company >= 70.0 and core_caps_count >= 2:
            synergy_bonus += 5.0
            
        # Recruiter engagement boost: +3.0 if recruitability_score >= 75.0 and at least 2 core capabilities
        if recruitability_score >= 75.0 and core_caps_count >= 2:
            synergy_bonus += 3.0
            
        # Cap synergy bonus at 25.0 points
        synergy_bonus = min(25.0, synergy_bonus)
        
        # Apply Synergy Bonus and cap final score at 100.0
        final_score = min(100.0, final_score + synergy_bonus)

        # FILTER RULE: If candidate is a flagged honeypot, force final score to 0.0
        if is_trap:
            final_score = 0.0

        breakdown = ScoreBreakdown(
            semantic_score=round(semantic_score, 2),
            graph_match_score=round(graph_match_score, 2),
            career_intel_score=round(career_intel_score, 2),
            career_quality_score=round(career_quality_score, 2),
            behavioral_modifier=round(recruitability_score, 2),
            final_score=round(final_score, 4)
        )

        return round(final_score, 4), breakdown, honeypot_reasons

    @classmethod
    def rank_candidates(
        cls, 
        candidates_list: List[CandidateRecord], 
        similarities: List[float], 
        jd_graph: JDRequirementGraph
    ) -> List[Tuple[CandidateRecord, float, ScoreBreakdown]]:
        """Scores all candidates, filters honeypots, and ranks them.
        
        Args:
            candidates_list (List[CandidateRecord]): The candidate list.
            similarities (List[float]): Cosine similarities from FAISS.
            jd_graph (JDRequirementGraph): The parsed job requirements.
            
        Returns:
            List[Tuple[CandidateRecord, float, ScoreBreakdown]]: Sorted rank list.
        """
        valid_candidates = []
        honeypot_candidates = []
        
        for idx, record in enumerate(candidates_list):
            sim = similarities[idx]
            score, breakdown, reasons = cls.score_candidate(record, sim, jd_graph)
            
            if score > 0.0:
                valid_candidates.append((record, score, breakdown))
            else:
                honeypot_candidates.append((record, 0.0, breakdown))

        # Deterministic sorting
        valid_candidates.sort(key=lambda x: (-x[1], x[0].candidate_id))
        honeypot_candidates.sort(key=lambda x: (x[0].candidate_id))
        
        return valid_candidates + honeypot_candidates
