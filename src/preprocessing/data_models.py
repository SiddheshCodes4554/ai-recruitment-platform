"""
Data Models Module.

Defines Pydantic schemas validating all profile, career history, signal, and scoring models.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional

__all__ = [
    "Profile",
    "CareerHistoryEntry",
    "EducationEntry",
    "SkillsEntry",
    "CertificationsEntry",
    "LanguagesEntry",
    "ExpectedSalaryRange",
    "RedrobSignals",
    "CandidateRecord",
    "DomainNode",
    "TechNode",
    "ConstraintNode",
    "JDRequirementGraph",
    "ScoreBreakdown",
    "CandidateRankResult"
]

class Profile(BaseModel):
    anonymized_name: str = Field(..., description="Anonymized full name.")
    headline: str = Field(..., description="One-line professional headline.")
    summary: str = Field(..., description="Multi-sentence professional summary.")
    location: str = Field(..., description="City and state/region.")
    country: str = Field(..., description="Country of residence.")
    years_of_experience: float = Field(..., ge=0, le=50, description="Total years of experience.")
    current_title: str = Field(..., description="Current job title.")
    current_company: str = Field(..., description="Current company name.")
    current_company_size: str = Field(..., description="Current company size band.")
    current_industry: str = Field(..., description="Current industry name.")

class CareerHistoryEntry(BaseModel):
    company: str = Field(..., description="Company name.")
    title: str = Field(..., description="Job title.")
    start_date: str = Field(..., description="Start date of the role (YYYY-MM-DD).")
    end_date: Optional[str] = Field(None, description="End date of the role or None if current.")
    duration_months: int = Field(..., ge=0, description="Duration in months.")
    is_current: bool = Field(..., description="True if this is their current job.")
    industry: str = Field(..., description="Industry of the company.")
    company_size: str = Field(..., description="Size of the company.")
    description: str = Field(..., description="Role responsibilities and achievements.")

class EducationEntry(BaseModel):
    institution: str = Field(..., description="Institution name.")
    degree: str = Field(..., description="Degree type (e.g. B.Tech, M.S.).")
    field_of_study: str = Field(..., description="Field of study (e.g. Computer Science).")
    start_year: int = Field(..., ge=1970, le=2030, description="Start year.")
    end_year: int = Field(..., ge=1970, le=2035, description="End year.")
    grade: Optional[str] = Field(None, description="GPA / percentage / class.")
    tier: str = Field("unknown", description="Institution prestige tier.")

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        valid_tiers = {"tier_1", "tier_2", "tier_3", "tier_4", "unknown"}
        if v not in valid_tiers:
            raise ValueError(f"tier must be one of {valid_tiers}")
        return v

class SkillsEntry(BaseModel):
    name: str = Field(..., description="Skill name.")
    proficiency: str = Field(..., description="Proficiency level.")
    endorsements: int = Field(..., ge=0, description="Number of endorsements.")
    duration_months: Optional[int] = Field(0, ge=0, description="Months of usage.")

    @field_validator("proficiency")
    @classmethod
    def validate_proficiency(cls, v: str) -> str:
        valid_proficiencies = {"beginner", "intermediate", "advanced", "expert"}
        if v not in valid_proficiencies:
            raise ValueError(f"proficiency must be one of {valid_proficiencies}")
        return v

class CertificationsEntry(BaseModel):
    name: str = Field(..., description="Certification name.")
    issuer: str = Field(..., description="Issuer name.")
    year: int = Field(..., description="Year of issuance.")

class LanguagesEntry(BaseModel):
    language: str = Field(..., description="Language name.")
    proficiency: str = Field(..., description="Proficiency level.")

    @field_validator("proficiency")
    @classmethod
    def validate_proficiency(cls, v: str) -> str:
        valid_proficiencies = {"basic", "conversational", "professional", "native"}
        if v not in valid_proficiencies:
            raise ValueError(f"proficiency must be one of {valid_proficiencies}")
        return v

class ExpectedSalaryRange(BaseModel):
    min: float = Field(..., ge=0, description="Minimum expected salary in INR LPA.")
    max: float = Field(..., ge=0, description="Maximum expected salary in INR LPA.")

class RedrobSignals(BaseModel):
    profile_completeness_score: float = Field(..., ge=0, le=100)
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int = Field(..., ge=0)
    applications_submitted_30d: int = Field(..., ge=0)
    recruiter_response_rate: float = Field(..., ge=0, le=1)
    avg_response_time_hours: float = Field(..., ge=0)
    skill_assessment_scores: Dict[str, float]
    connection_count: int = Field(..., ge=0)
    endorsements_received: int = Field(..., ge=0)
    notice_period_days: int = Field(..., ge=0, le=180)
    expected_salary_range_inr_lpa: ExpectedSalaryRange
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float = Field(..., ge=-1, le=100)
    search_appearance_30d: int = Field(..., ge=0)
    saved_by_recruiters_30d: int = Field(..., ge=0)
    interview_completion_rate: float = Field(..., ge=0, le=1)
    offer_acceptance_rate: float = Field(..., ge=-1, le=1)
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    @field_validator("preferred_work_mode")
    @classmethod
    def validate_work_mode(cls, v: str) -> str:
        valid_modes = {"remote", "hybrid", "onsite", "flexible"}
        if v not in valid_modes:
            raise ValueError(f"preferred_work_mode must be one of {valid_modes}")
        return v

class CandidateRecord(BaseModel):
    candidate_id: str = Field(..., pattern="^CAND_[0-9]{7}$")
    profile: Profile
    career_history: List[CareerHistoryEntry]
    education: List[EducationEntry]
    skills: List[SkillsEntry]
    certifications: Optional[List[CertificationsEntry]] = []
    languages: Optional[List[LanguagesEntry]] = []
    redrob_signals: RedrobSignals

# JD Requirement Graph Structures
class DomainNode(BaseModel):
    domain: str
    weight: float = Field(..., ge=0, le=1)

class TechNode(BaseModel):
    skill: str
    weight: float = Field(..., ge=0, le=1)

class ConstraintNode(BaseModel):
    min_exp: float = Field(..., ge=0)
    max_exp: float = Field(..., ge=0)
    ideal_min_exp: float = Field(..., ge=0)
    ideal_max_exp: float = Field(..., ge=0)
    locations: List[str]
    max_notice_days: int = Field(..., ge=0)

class JDRequirementGraph(BaseModel):
    domains: List[DomainNode]
    tech_skills: List[TechNode]
    constraints: ConstraintNode

# Online Scoring Result Structs
class ScoreBreakdown(BaseModel):
    semantic_score: float = Field(..., description="FAISS Cosine Similarity score [0, 1]")
    graph_match_score: float = Field(..., description="Requirement graph alignment score [0, 100]")
    career_intel_score: float = Field(..., description="Career intelligence / technical evidence score [0, 100]")
    career_quality_score: float = Field(..., description="Overall career trajectory quality score [0, 100]")
    behavioral_modifier: float = Field(..., description="Activity and availability multiplier [0.5, 1.5]")
    final_score: float = Field(..., description="Computed composite score")

class CandidateRankResult(BaseModel):
    candidate_id: str
    rank: int = Field(..., ge=1, le=100)
    score: float
    reasoning: str
    breakdown: ScoreBreakdown
