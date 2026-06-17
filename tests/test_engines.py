import pytest
from src.preprocessing import CandidateRecord, ScoreBreakdown
from src.engines import CareerIntelligenceEngine, JDGraphParser
from src.explainability import CandidateExplainer

def test_jd_graph_parsing():
    """Tests that the JDGraphParser correctly extracts domains, technologies, and constraints."""
    jd_text = """
    We are looking for a Senior AI Engineer with 6-8 years of experience.
    Must have experience with Elasticsearch, OpenSearch, faiss, and sentence-transformers.
    Strong Python programming skills and ML ranking experience (NDCG, MAP, MRR).
    Nice to have: recommendation systems and vector search.
    Locations: Noida, Pune, Delhi NCR.
    Notice period: sub-30 days preferred.
    """
    
    graph = JDGraphParser.parse_jd(jd_text)
    
    # Assert constraints
    assert graph.constraints.min_exp == 6.0
    assert graph.constraints.max_exp == 8.0
    assert graph.constraints.ideal_min_exp == 7.0
    assert graph.constraints.ideal_max_exp == 7.0
    assert "Noida" in graph.constraints.locations
    assert "Pune" in graph.constraints.locations
    assert graph.constraints.max_notice_days == 30
    
    # Assert domains and technologies
    domain_names = [d.domain for d in graph.domains]
    assert "Search" in domain_names
    assert "Ranking" in domain_names
    assert "Retrieval" in domain_names
    
    # Assert tech keywords normalized
    tech_names = [t.skill for t in graph.tech_skills]
    assert "elasticsearch" in tech_names
    assert "faiss" in tech_names
    assert "ndcg" in tech_names

def test_career_intelligence_scoring_and_penalties():
    """Tests the CareerIntelligenceEngine score calculation and stuffing detection."""
    mock_data = {
        "candidate_id": "CAND_9999999",
        "profile": {
            "anonymized_name": "Alice Developer",
            "headline": "Senior Search Engineer",
            "summary": "Building large-scale search engines and recommendation systems.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Search Engineer",
            "current_company": "BigTech",
            "current_company_size": "10001+",
            "current_industry": "Software"
        },
        "career_history": [
            {
                "company": "BigTech",
                "title": "Search Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 77,
                "is_current": True,
                "industry": "Software",
                "company_size": "10001+",
                "description": "Led migration of search backend to Elasticsearch. Designed semantic search with faiss and sentence-transformers. Improved NDCG by 12% using Learning to Rank. Deployed models using Docker."
            }
        ],
        "education": [],
        "skills": [
            {"name": "elasticsearch", "proficiency": "advanced", "endorsements": 10, "duration_months": 60},
            {"name": "faiss", "proficiency": "advanced", "endorsements": 5, "duration_months": 48},
            {"name": "sentence-transformers", "proficiency": "advanced", "endorsements": 5, "duration_months": 36},
            {"name": "docker", "proficiency": "advanced", "endorsements": 10, "duration_months": 72},
            {"name": "unused-skill", "proficiency": "beginner", "endorsements": 0, "duration_months": 12}
        ],
        "redrob_signals": {
            "profile_completeness_score": 95.0,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-01-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 25,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.9,
            "avg_response_time_hours": 4.0,
            "skill_assessment_scores": {},
            "connection_count": 500,
            "endorsements_received": 25,
            "notice_period_days": 15,
            "expected_salary_range_inr_lpa": {"min": 18.0, "max": 28.0},
            "preferred_work_mode": "remote",
            "willing_to_relocate": True,
            "github_activity_score": 75.0,
            "search_appearance_30d": 150,
            "saved_by_recruiters_30d": 12,
            "interview_completion_rate": 0.95,
            "offer_acceptance_rate": 0.85,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }
    
    record = CandidateRecord(**mock_data)
    result = CareerIntelligenceEngine.evaluate(record)
    
    # Assert evidence score components are triggered
    assert result.career_intelligence_score > 0
    assert result.search_infra_score > 0
    assert result.embeddings_score > 0
    
    # Verify keyword stuffing: "solr" is in PATTERNS (search_infra)
    # If we add "solr" to skills list but not history, it should flag as stuffed.
    mock_data_stuffing = dict(mock_data)
    mock_data_stuffing["skills"] = mock_data["skills"] + [
        {"name": "solr", "proficiency": "advanced", "endorsements": 5, "duration_months": 24}
    ]
    
    record_stuffing = CandidateRecord(**mock_data_stuffing)
    result_stuffing = CareerIntelligenceEngine.evaluate(record_stuffing)
    
    assert "solr" in result_stuffing.stuffed_skills
    assert result_stuffing.stuffing_penalty > 0.0

def test_candidate_explainer():
    """Tests that the CandidateExplainer generates a non-empty, detailed explanation."""
    mock_data = {
        "candidate_id": "CAND_8888888",
        "profile": {
            "anonymized_name": "Bob Explainer",
            "headline": "Lead AI Engineer",
            "summary": "Expert in search and AI.",
            "location": "Noida",
            "country": "India",
            "years_of_experience": 8.0,
            "current_title": "Lead AI Engineer",
            "current_company": "TechCorp",
            "current_company_size": "1000-5000",
            "current_industry": "Software"
        },
        "career_history": [
            {
                "company": "TechCorp",
                "title": "Lead AI Engineer",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 53,
                "is_current": True,
                "industry": "Software",
                "company_size": "1000-5000",
                "description": "Architected recommendations engine. Improved MAP score to 0.82. Managed team of 5."
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
            "expected_salary_range_inr_lpa": {"min": 25.0, "max": 35.0},
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
    
    breakdown_data = {
        "final_score": 88.5,
        "semantic_score": 85.0,
        "graph_match_score": 90.0,
        "career_intel_score": 88.0,
        "career_quality_score": 92.0,
        "experience_alignment_score": 100.0,
        "behavioral_modifier": 95.0
    }
    
    record = CandidateRecord(**mock_data)
    breakdown = ScoreBreakdown(**breakdown_data)
    
    reasoning = CandidateExplainer.generate_reasoning(record, breakdown)
    assert len(reasoning) > 0
    assert "Lead AI Engineer" in reasoning
    assert "8.0 years of experience" in reasoning
    assert "TechCorp" in reasoning
