import math
from src.preprocessing import CandidateRecord, JDRequirementGraph
from src.preprocessing.data_models import DomainNode, TechNode, ConstraintNode
from src.ranking import CandidateRanker

def test_career_quality_scoring():
    """Verifies that company tiering and tenure calculations evaluate properly in CandidateRanker."""
    mock_data = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Ira Vora",
            "headline": "Backend Engineer",
            "summary": "Experienced engineer.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.9,
            "current_title": "Backend Engineer",
            "current_company": "Mindtree",
            "current_company_size": "10001+",
            "current_industry": "IT Services"
        },
        "career_history": [
            {
                "company": "Mindtree",
                "title": "Backend Engineer",
                "start_date": "2024-03-08",
                "end_date": None,
                "duration_months": 27,
                "is_current": True,
                "industry": "IT Services",
                "company_size": "10001+",
                "description": "Implemented streaming data pipelines."
            }
        ],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-01-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 10,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {
                "min": 10.0,
                "max": 20.0
            },
            "preferred_work_mode": "remote",
            "willing_to_relocate": True,
            "github_activity_score": 50.0,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }
    
    record = CandidateRecord(**mock_data)
    
    # Calculate company score: Mindtree is IT services = 20 points
    # Tenure score: 27 months = 80 points
    # Exp curve: close to 7.0 = high points (~99.7)
    # Expected career quality score: 0.5 * 20 + 0.3 * 99.77 + 0.2 * 80 = 10 + 29.93 + 16 = 55.93
    quality_score = CandidateRanker.calculate_career_quality_score(record)
    
    assert quality_score > 0.0
    assert 50.0 <= quality_score <= 65.0

def test_experience_alignment_scoring():
    """Verifies the ExperienceAlignmentEngine curve targets."""
    from src.engines import ExperienceAlignmentEngine
    
    # Target assertions for min=5, max=9, ideal=6-8
    assert 20.0 <= ExperienceAlignmentEngine.calculate_score(3.0, 5.0, 9.0, 6.0, 8.0) <= 30.0
    assert 50.0 <= ExperienceAlignmentEngine.calculate_score(4.0, 5.0, 9.0, 6.0, 8.0) <= 65.0
    assert 80.0 <= ExperienceAlignmentEngine.calculate_score(5.0, 5.0, 9.0, 6.0, 8.0) <= 90.0
    assert 95.0 <= ExperienceAlignmentEngine.calculate_score(6.0, 5.0, 9.0, 6.0, 8.0) <= 100.0
    assert ExperienceAlignmentEngine.calculate_score(7.0, 5.0, 9.0, 6.0, 8.0) == 100.0
    assert 95.0 <= ExperienceAlignmentEngine.calculate_score(8.0, 5.0, 9.0, 6.0, 8.0) <= 100.0
    assert 85.0 <= ExperienceAlignmentEngine.calculate_score(9.0, 5.0, 9.0, 6.0, 8.0) <= 93.0
    assert 70.0 <= ExperienceAlignmentEngine.calculate_score(10.0, 5.0, 9.0, 6.0, 8.0) <= 80.0
    assert ExperienceAlignmentEngine.calculate_score(12.0, 5.0, 9.0, 6.0, 8.0) <= 35.0
