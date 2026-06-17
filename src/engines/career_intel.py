"""
Career Intelligence Engine Module.

Extracts recruiter-level evidence from candidate profiles, checking action verbs,
metrics, and search capability alignment, while applying stuffing penalties.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Tuple
from src.preprocessing.data_models import CandidateRecord
from src.config import (
    INTEL_BASE_SCORE,
    INTEL_TITLE_BOOST,
    INTEL_ACTION_VERB_BOOST,
    INTEL_METRIC_BOOST,
    INTEL_DIVISOR,
    STUFFING_TIER_1_RATIO,
    STUFFING_TIER_1_PENALTY,
    STUFFING_TIER_2_RATIO,
    STUFFING_TIER_2_PENALTY,
    STUFFING_MAX_PENALTY
)

__all__ = ["CareerIntelligenceResult", "CareerIntelligenceEngine"]

@dataclass
class CareerIntelligenceResult:
    """Dataclass holding the scoring breakdown and list of stuffed keywords."""
    career_intelligence_score: float
    retrieval_score: float
    ranking_score: float
    recommendation_score: float
    search_infra_score: float
    embeddings_score: float
    vector_db_score: float
    production_ml_score: float
    experimentation_score: float
    ab_testing_score: float
    shipping_score: float
    scale_score: float
    leadership_score: float
    stuffed_skills: List[str] = field(default_factory=list)
    stuffing_penalty: float = 0.0
    search_engineering_score: float = 0.0

class CareerIntelligenceEngine:
    """
    Recruiter-level engine that parses candidate profiles for contextual evidence
    of technical expertise, shipping systems, scale, and leadership, while penalizing
    unsubstantiated keyword stuffing in skills lists.
    """

    # 1. Categories and keyword lists (compiled regex patterns)
    PATTERNS: Dict[str, List[re.Pattern]] = {
        "retrieval": [
            re.compile(r"\b(dense passage retrieval|information retrieval|search retrieval|retrieval|bm25|tf-idf|tfidf|dpr)\b", re.IGNORECASE)
        ],
        "ranking": [
            re.compile(r"\b(learning to rank|re-ranking|re-rank|ndcg|mrr|map|cross-encoder|relevance scoring|ranking|ltr)\b", re.IGNORECASE)
        ],
        "recommendation": [
            re.compile(r"\b(recommendation|recommender|rec-sys|collaborative filtering|personalization|matrix factorization)\b", re.IGNORECASE)
        ],
        "search_infra": [
            re.compile(r"\b(search infrastructure|elasticsearch|opensearch|solr|lucene|search engine|hybrid search|semantic search|semantic match|neural search)\b", re.IGNORECASE)
        ],
        "embeddings": [
            re.compile(r"\b(sentence transformer|sentence-transformer|embeddings|embedding|bert|bge|cohere embedding|vector representation|semantic search|semantic match|neural search)\b", re.IGNORECASE)
        ],
        "vector_db": [
            re.compile(r"\b(vector database|vector index|faiss|pinecone|qdrant|milvus|weaviate|hnsw)\b", re.IGNORECASE)
        ],
        "production_ml": [
            re.compile(r"\b(production ml|mlops|model drift|drift detection|model monitoring|inference optimization|deployment|model serving)\b", re.IGNORECASE)
        ],
        "experimentation": [
            re.compile(r"\b(experimentation|hypothesis testing|multi-armed bandit|bandit|statistically significant)\b", re.IGNORECASE)
        ],
        "ab_testing": [
            re.compile(r"\b(a/b testing|ab testing|split test|online experiment)\b", re.IGNORECASE)
        ],
        "shipping": [
            re.compile(r"\b(shipped|built|designed|deployed|scaled|optimized|launched|engineered|implemented|architected|developed|created|owned|rolled\s+out|migrated)\b", re.IGNORECASE)
        ],
        "scale": [
            re.compile(r"\b(scale|throughput|latency|ms|ctr|conversion|qps|users|traffic|datasets|millions|billions|accuracy)\b", re.IGNORECASE)
        ],
        "leadership": [
            re.compile(r"\b(led|architected|managed|owned|leader|technical lead|staff engineer|mentored|team)\b", re.IGNORECASE)
        ]
    }

    # 2. Key Action Verbs and Metrics for context boosting
    ACTION_VERB_PATTERN = re.compile(
        r"\b(built|shipped|designed|deployed|scaled|optimized|launched|engineered|implemented|architected|developed|created|led|owned|rolled\s+out|migrated)\b", 
        re.IGNORECASE
    )
    
    METRIC_PATTERN = re.compile(
        r"(\b\d+%\b|%|\b\d+\s*ms\b|\bms\b|\blatency\b|\bthroughput\b|\bctr\b|\bconversion\b|\baccuracy\b|\bmillions\b|\bbillions\b|\bqps\b)", 
        re.IGNORECASE
    )

    @classmethod
    def _contains_word(cls, text: str, word: str) -> bool:
        """Helper to find if a specific skill name is mentioned in a text block.
        
        Args:
            text (str): The search text space.
            word (str): The skill word.
            
        Returns:
            bool: True if word matches.
        """
        escaped_word = re.escape(word)
        pattern = re.compile(r"\b" + escaped_word + r"\b", re.IGNORECASE)
        return bool(pattern.search(text))

    @classmethod
    def detect_keyword_stuffing(cls, record: CandidateRecord) -> Tuple[List[str], float]:
        """Identifies skills listed in skills section with NO work history evidence.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            
        Returns:
            Tuple[List[str], float]: Stuffed skills and penalty value.
        """
        p = record.profile
        experience_text = " ".join([
            p.headline,
            p.summary,
            p.current_title,
            " ".join([f"{job.title} {job.description}" for job in record.career_history])
        ])

        stuffed_skills = []
        search_skills_count = 0
        
        for skill in record.skills:
            skill_name = skill.name
            
            is_search_skill = False
            for patterns in cls.PATTERNS.values():
                for pattern in patterns:
                    if pattern.search(skill_name):
                        is_search_skill = True
                        break
                if is_search_skill:
                    break
                    
            if is_search_skill:
                search_skills_count += 1
                if not cls._contains_word(experience_text, skill_name):
                    stuffed_skills.append(skill_name)
                    
        if search_skills_count == 0:
            return [], 0.0
            
        stuffing_ratio = len(stuffed_skills) / search_skills_count
        
        if len(stuffed_skills) == 0:
            penalty = 0.0
        elif stuffing_ratio <= STUFFING_TIER_1_RATIO:
            penalty = STUFFING_TIER_1_PENALTY
        elif stuffing_ratio <= STUFFING_TIER_2_RATIO:
            penalty = STUFFING_TIER_2_PENALTY
        else:
            penalty = STUFFING_MAX_PENALTY
            
        return stuffed_skills, penalty

    @classmethod
    def _evaluate_sentence(cls, sentence: str, category_patterns: List[re.Pattern], is_title_or_headline: bool) -> float:
        """Scores a single sentence for a given category's presence.
        
        Args:
            sentence (str): The sentence text.
            category_patterns (List[re.Pattern]): Compiled category regex.
            is_title_or_headline (bool): True if title context.
            
        Returns:
            float: Sentence category score.
        """
        sentence_score = 0.0
        
        keyword_matched = False
        for pattern in category_patterns:
            if pattern.search(sentence):
                keyword_matched = True
                break
                
        if not keyword_matched:
            return 0.0
            
        sentence_score += INTEL_BASE_SCORE
        
        if is_title_or_headline:
            sentence_score += INTEL_TITLE_BOOST
            
        if cls.ACTION_VERB_PATTERN.search(sentence):
            sentence_score += INTEL_ACTION_VERB_BOOST
            
        if cls.METRIC_PATTERN.search(sentence):
            sentence_score += INTEL_METRIC_BOOST
            
        return sentence_score

    @classmethod
    def score_category(cls, record: CandidateRecord, category: str) -> float:
        """Calculates score for a specific category across profile.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            category (str): The category name.
            
        Returns:
            float: The capped category score.
        """
        patterns = cls.PATTERNS.get(category, [])
        if not patterns:
            return 0.0
            
        p = record.profile
        total_category_score = 0.0
        
        for sentence in re.split(r'[.!?]\s*', p.headline):
            if sentence:
                total_category_score += cls._evaluate_sentence(sentence, patterns, is_title_or_headline=True)
                
        for sentence in re.split(r'[.!?]\s*', p.summary):
            if sentence:
                total_category_score += cls._evaluate_sentence(sentence, patterns, is_title_or_headline=False)

        for job in record.career_history:
            total_category_score += cls._evaluate_sentence(job.title, patterns, is_title_or_headline=True)
            for sentence in re.split(r'[.!?]\s*', job.description):
                if sentence:
                    total_category_score += cls._evaluate_sentence(sentence, patterns, is_title_or_headline=False)

        return min(10.0, total_category_score)

    @classmethod
    def evaluate(cls, record: CandidateRecord) -> CareerIntelligenceResult:
        """Runs the Career Intelligence evaluation.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            
        Returns:
            CareerIntelligenceResult: The calculated result breakdown.
        """
        scores = {}
        for cat in cls.PATTERNS.keys():
            scores[cat] = cls.score_category(record, cat)
            
        total_raw = sum(scores.values())
        raw_score = min(100.0, (total_raw / INTEL_DIVISOR) * 100.0)

        retrieval_val = scores["retrieval"] * 10
        ranking_val = scores["ranking"] * 10
        recommendation_val = scores["recommendation"] * 10
        embeddings_val = scores["embeddings"] * 10
        vector_db_val = scores["vector_db"] * 10
        search_infra_val = scores["search_infra"] * 10
        production_ml_val = scores["production_ml"] * 10

        base_search_score = (
            retrieval_val + 
            ranking_val + 
            recommendation_val + 
            embeddings_val + 
            vector_db_val + 
            search_infra_val + 
            production_ml_val
        ) / 7.0

        has_retrieval = retrieval_val >= 60.0
        has_ranking = ranking_val >= 60.0
        has_rec = recommendation_val >= 60.0
        has_embeddings = embeddings_val >= 60.0
        has_vector_db = vector_db_val >= 60.0
        has_search_infra = search_infra_val >= 60.0
        has_prod_ml = production_ml_val >= 60.0

        p = record.profile
        experience_text = " ".join([
            p.headline,
            p.summary,
            p.current_title,
            " ".join([f"{job.title} {job.description}" for job in record.career_history])
        ]).lower()
        has_eval_metrics = any(metric in experience_text for metric in ["ndcg", "map", "mrr"])

        search_bonus = 0.0
        if has_retrieval and has_ranking and has_embeddings and has_eval_metrics:
            search_bonus = 15.0
        elif has_retrieval and has_ranking and has_prod_ml:
            search_bonus = 10.0
        elif has_retrieval and has_ranking:
            search_bonus = 5.0

        search_engineering_score = min(100.0, base_search_score + search_bonus)
        
        merged_raw_score = 0.50 * raw_score + 0.50 * search_engineering_score

        stuffed_skills, penalty = cls.detect_keyword_stuffing(record)
        final_score = merged_raw_score * (1.0 - penalty)
        
        return CareerIntelligenceResult(
            career_intelligence_score=round(final_score, 2),
            retrieval_score=round(retrieval_val, 1),
            ranking_score=round(ranking_val, 1),
            recommendation_score=round(recommendation_val, 1),
            search_infra_score=round(search_infra_val, 1),
            embeddings_score=round(embeddings_val, 1),
            vector_db_score=round(vector_db_val, 1),
            production_ml_score=round(production_ml_val, 1),
            experimentation_score=round(scores["experimentation"] * 10, 1),
            ab_testing_score=round(scores["ab_testing"] * 10, 1),
            shipping_score=round(scores["shipping"] * 10, 1),
            scale_score=round(scores["scale"] * 10, 1),
            leadership_score=round(scores["leadership"] * 10, 1),
            stuffed_skills=stuffed_skills,
            stuffing_penalty=round(penalty, 2),
            search_engineering_score=round(search_engineering_score, 2)
        )
