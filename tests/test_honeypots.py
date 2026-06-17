import json
from pathlib import Path
from src.preprocessing import CandidateRecord
from src.engines import HoneypotDetector
from src.config import SAMPLE_CANDIDATES_PATH

def test_sample_candidates_parsing():
    """Verifies that the sample candidates JSON parses correctly and validations succeed."""
    sample_path = SAMPLE_CANDIDATES_PATH
    assert sample_path.exists(), f"Sample candidates file not found at {sample_path}"
    
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert len(data) > 0
    
    # Parse the first candidate
    record = CandidateRecord(**data[0])
    assert record.candidate_id == "CAND_0000001"
    assert record.profile.anonymized_name == "Ira Vora"

def test_honeypot_logic():
    """Tests that the HoneypotDetector correctly flags anomalous records and gives logical reasons."""
    # Create a mock candidate record with a clear experience contradiction
    mock_data = {
        "candidate_id": "CAND_9999999",
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Software Engineer",
            "summary": "Experienced engineer.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 10.0,  # 10 years experience claimed
            "current_title": "Software Engineer",
            "current_company": "Acme Corp",
            "current_company_size": "51-200",
            "current_industry": "Software"
        },
        "career_history": [
            {
                "company": "Acme Corp",
                "title": "Software Engineer",
                "start_date": "2024-01-01",
                "end_date": "2025-01-01",
                "duration_months": 12,  # Only 1 year of actual work history duration
                "is_current": False,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Wrote code."
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
    is_trap, reasons = HoneypotDetector.is_honeypot(record)
    
    assert is_trap is True
    assert any("Mismatch" in r or "Contradiction" in r for r in reasons)
