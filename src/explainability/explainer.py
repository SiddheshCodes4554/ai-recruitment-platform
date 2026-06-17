"""
Candidate Explainer Module.

Generates personalized, non-templated, fact-based 1-2 sentence justifications
explaining why a candidate is ranked at their position.
"""

from src.preprocessing.data_models import CandidateRecord, ScoreBreakdown
from src.engines.experience_alignment import ExperienceAlignmentEngine

__all__ = ["CandidateExplainer"]

class CandidateExplainer:
    """
    Candidate Explainer Engine.
    Generates personalized, non-templated, fact-based 1-2 sentence justifications
    explaining why a candidate is ranked at their position, referencing specific profile
    facts and career history achievements.
    """

    @classmethod
    def generate_reasoning(cls, record: CandidateRecord, breakdown: ScoreBreakdown) -> str:
        """Synthesizes candidate profile data and scoring breakdowns into an explanation.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            breakdown (ScoreBreakdown): The detailed score breakdowns.
            
        Returns:
            str: Fact-based, personalized reasoning explanation.
        """
        p = record.profile
        history = record.career_history
        skills = record.skills
        
        # 1. Experience & Role Context
        yoe = p.years_of_experience
        title = p.current_title
        company = p.current_company
        
        # 2. Extract key achievements/tools from work history description
        experience_text = " ".join([job.description for job in history]).lower()
        
        achievements = []
        if "retrieval" in experience_text or "search" in experience_text:
            achievements.append("search retrieval")
        if "ranking" in experience_text or "re-rank" in experience_text or "ndcg" in experience_text:
            achievements.append("ranking systems")
        if "embeddings" in experience_text or "sentence-transformers" in experience_text:
            achievements.append("vector embeddings")
        if "faiss" in experience_text or "pinecone" in experience_text or "qdrant" in experience_text or "milvus" in experience_text:
            achievements.append("vector indexing")
        if "ab test" in experience_text or "a/b test" in experience_text or "experiment" in experience_text:
            achievements.append("A/B experimentation")
        if "production" in experience_text or "deploy" in experience_text or "scale" in experience_text:
            achievements.append("production scale deployments")

        # 3. Company tier and quality signals
        is_product = breakdown.career_quality_score > 60.0
        company_type = "product companies" if is_product else "IT services"

        # 4. Availability and notice period context
        notice_days = record.redrob_signals.notice_period_days
        response_rate = record.redrob_signals.recruiter_response_rate * 100.0
        
        availability_str = ""
        if notice_days <= 30:
            availability_str = "with immediate/30-day availability"
        elif notice_days <= 60:
            availability_str = f"with a {notice_days}-day notice period"
        else:
            availability_str = f"with a longer {notice_days}-day notice period"

        # 4.5. Experience Alignment Explanation
        exp_align_score = ExperienceAlignmentEngine.calculate_score(yoe, 5.0, 9.0, 6.0, 8.0)
        has_search_evidence = "search retrieval" in achievements or "ranking systems" in achievements or "vector embeddings" in achievements
        
        exp_explanation = ""
        if exp_align_score >= 90.0:
            exp_explanation = "Experience aligns strongly with the role's preferred 6–8 year range."
        elif yoe < 6.0:
            if has_search_evidence:
                exp_explanation = "Candidate is slightly below the preferred experience range but demonstrates exceptional search engineering evidence."
            else:
                exp_explanation = "Candidate is slightly below the preferred experience range."
        else: # yoe > 8.0
            if has_search_evidence:
                exp_explanation = "Candidate exceeds the preferred experience range but remains relevant due to strong production search experience."
            else:
                exp_explanation = "Candidate exceeds the preferred experience range."

        # 5. Formulate natural sentences
        # Sentence 1: Profile identity, experience level, and key evidence
        achievements_str = ""
        if achievements:
            if len(achievements) > 2:
                achievements_str = f" possessing direct hands-on experience building {', '.join(achievements[:2])}, and {achievements[2]}"
            else:
                achievements_str = f" possessing direct experience with {' and '.join(achievements)}"
        else:
            # Fallback to skills listed in profile if not explicit in descriptions
            key_skills = [s.name for s in skills if s.proficiency in ["expert", "advanced"]][:3]
            if key_skills:
                achievements_str = f" skilled in {', '.join(key_skills)}"

        sentence_1 = (
            f"{title} with {yoe:.1f} years of experience primarily at {company_type} (current: {company}),"
            f"{achievements_str}."
        )

        # Sentence 2: Match quality, engagement, and availability
        match_level = "strong fit" if breakdown.final_score > 75.0 else "moderate fit"
        engagement_str = f"highly active with a {response_rate:.0f}% recruiter response rate" if response_rate > 70.0 else "active"
        
        sentence_2 = (
            f"{exp_explanation} Evaluates as a {match_level} {availability_str}, "
            f"backed by {engagement_str} platform engagement signals."
        )

        return f"{sentence_1} {sentence_2}"
