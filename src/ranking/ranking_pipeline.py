"""
Online Ranking Pipeline Module.

Orchestrates the candidate retrieval, scoring, honeypot filtering,
and reasoning generation stages.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import time
import pickle
import csv
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

from src.config.config import (
    FAISS_INDEX_PATH,
    METADATA_CACHE_PATH,
    JOB_DESCRIPTION_DOCX_PATH,
    SENTENCE_TRANSFORMER_MODEL
)
from src.preprocessing.data_models import CandidateRecord, ScoreBreakdown, CandidateRankResult
from src.preprocessing.preprocessor import ProfilePreprocessor
from src.engines.honeypot_detector import HoneypotDetector
from src.engines.jd_graph import JDGraphParser
from src.engines.career_intel import CareerIntelligenceEngine
from src.engines.experience_alignment import ExperienceAlignmentEngine
from src.ranking.ranker import CandidateRanker
from src.explainability.explainer import CandidateExplainer
from src.utils.logger import get_logger

__all__ = ["extract_docx_text", "run_ranking_pipeline"]

logger = get_logger("ranking_pipeline")

def extract_docx_text(docx_path: Path) -> str:
    """Reads docx XML document structure without requiring compiled binary external libraries.
    
    Args:
        docx_path (Path): Path to the docx document.
        
    Returns:
        str: The extracted plain text content.
    """
    try:
        try:
            import docx
            doc = docx.Document(docx_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            pass
            
        with zipfile.ZipFile(docx_path) as docx_zip:
            xml_content = docx_zip.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            def get_text(element):
                text_list = []
                for child in element.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                    if child.text:
                        text_list.append(child.text)
                return "".join(text_list)
            
            lines = []
            body = root.find('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body')
            if body is not None:
                for child in body:
                    tag = child.tag.split('}')[-1]
                    if tag == 'p':
                        text = get_text(child)
                        if text:
                            lines.append(text)
            return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error reading docx: {e}")
        return ""

def run_ranking_pipeline(
    candidates_file: Path,
    out_csv_path: Path,
    jd_override_text: str = None
) -> List[Tuple[CandidateRecord, float, ScoreBreakdown]]:
    """Orchestrates the online ranking pipeline.
    
    Loads FAISS index, metadata cache, JD text (or docx), builds the JD graph,
    queries the index for the top 5,000 candidates, computes multi-dimensional scores,
    filters honeypots, ranks deterministically, generates reasonings, and exports the top 100.
    
    Args:
        candidates_file (Path): Path to the candidates dataset file.
        out_csv_path (Path): Path to save the ranked CSV results.
        jd_override_text (str, optional): Plaintext JD override.
        
    Returns:
        List[Tuple[CandidateRecord, float, ScoreBreakdown]]: Ranked results for the top 100.
    """
    pipeline_start = time.time()
    logger.info("=== Executing Candidate Ranking Pipeline ===")
    
    # 1. Load FAISS index & Metadata Cache
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Please run scripts/precompute.py first.")
    if not METADATA_CACHE_PATH.exists():
        raise FileNotFoundError(f"Metadata cache not found at {METADATA_CACHE_PATH}. Please run scripts/precompute.py first.")
        
    logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    
    logger.info(f"Loading metadata cache from {METADATA_CACHE_PATH}...")
    with open(METADATA_CACHE_PATH, "rb") as f:
        metadata_cache: List[Dict[str, Any]] = pickle.load(f)
        
    logger.info(f"Loaded {len(metadata_cache)} cached candidate profiles.")

    # 2. Load Job Description
    jd_text = ""
    if jd_override_text:
        jd_text = jd_override_text
        logger.info("Using overriding Job Description text...")
    else:
        if not JOB_DESCRIPTION_DOCX_PATH.exists():
            raise FileNotFoundError(f"Job Description docx not found at {JOB_DESCRIPTION_DOCX_PATH}")
        logger.info(f"Reading Job Description from {JOB_DESCRIPTION_DOCX_PATH}...")
        jd_text = extract_docx_text(JOB_DESCRIPTION_DOCX_PATH)
        
    if not jd_text.strip():
        raise ValueError("Job Description content is empty!")

    # 3. Parse Job Description into JDRequirementGraph
    logger.info("Parsing Job Description into requirement graph...")
    jd_graph = JDGraphParser.parse_jd(jd_text)
    logger.info(f"  Parsed constraints: YoE {jd_graph.constraints.min_exp}-{jd_graph.constraints.max_exp} "
                f"(ideal {jd_graph.constraints.ideal_min_exp}-{jd_graph.constraints.ideal_max_exp}), "
                f"locations: {jd_graph.constraints.locations}, notice: {jd_graph.constraints.max_notice_days} days")
    logger.info(f"  Active Domains: {[d.domain for d in jd_graph.domains]}")
    logger.info(f"  Active Technologies: {[t.skill for t in jd_graph.tech_skills]}")

    # 4. Generate query embedding for JD
    logger.info(f"Loading SentenceTransformer model: {SENTENCE_TRANSFORMER_MODEL}...")
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cpu")
    
    logger.info("Encoding Job Description vector...")
    jd_vector = model.encode([jd_text], convert_to_numpy=True, normalize_embeddings=True)
    jd_vector = jd_vector.astype(np.float32)

    # 5. Retrieve top 5,000 candidates via FAISS Inner Product
    logger.info("Performing semantic search against 100k candidate index...")
    k = min(5000, len(metadata_cache))
    distances, indices = index.search(jd_vector, k)
    similarities = distances[0]
    candidate_indices = indices[0]

    # 6. Compute composite scores and filter honeypots
    logger.info(f"Evaluating and scoring top {k} retrieved candidates...")
    retrieved_records: List[CandidateRecord] = []
    retrieved_similarities: List[float] = []
    
    for i, cache_idx in enumerate(candidate_indices):
        if cache_idx == -1:
            continue
        sim = similarities[i]
        meta = metadata_cache[cache_idx]
        record = CandidateRecord.model_validate(meta)
        retrieved_records.append(record)
        retrieved_similarities.append(sim)

    # 7. Score, Filter and Rank deterministically
    ranked_results = CandidateRanker.rank_candidates(retrieved_records, retrieved_similarities, jd_graph)
    
    top_100 = ranked_results[:100]
    logger.info(f"Ranked and selected top {len(top_100)} candidates.")

    # 8. Export Top 100 to CSV
    logger.info(f"Writing final ranking results to {out_csv_path}...")
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_csv_path, "w", encoding="utf-8", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank_pos, (record, score, breakdown) in enumerate(top_100, 1):
            reasoning = CandidateExplainer.generate_reasoning(record, breakdown)
            writer.writerow([record.candidate_id, rank_pos, round(score, 4), reasoning])
            
    logger.info(f"CSV exported successfully. Output: {out_csv_path}")

    # 9. Output Diagnostics
    print("\n" + "="*40)
    print("           DIAGNOSTIC SUMMARY           ")
    print("="*40)
    
    # Core stats
    total_duration = time.time() - pipeline_start
    print(f"Total pipeline execution time: {total_duration:.2f} seconds")
    
    # Score distribution and percentiles
    final_scores = [item[1] for item in top_100]
    if final_scores:
        print(f"Score Range: Min {min(final_scores):.4f} | Max {max(final_scores):.4f}")
        print(f"Score Mean: {np.mean(final_scores):.4f} | Median: {np.median(final_scores):.4f}")
        print(f"Score StdDev: {np.std(final_scores):.4f}")
        
        p99, p95, p90, p75, p50 = np.percentile(final_scores, [99, 95, 90, 75, 50])
        print(f"Score Percentiles:")
        print(f"  - 99th: {p99:.4f}")
        print(f"  - 95th: {p95:.4f}")
        print(f"  - 90th: {p90:.4f}")
        print(f"  - 75th: {p75:.4f}")
        print(f"  - 50th: {p50:.4f}")

    # Average scores
    intel_scores = [breakdown.career_intel_score for _, _, breakdown in top_100]
    avg_intel = np.mean(intel_scores) if intel_scores else 0.0
    print(f"Average Career Intelligence Score in Top 100: {avg_intel:.2f}/100.0")

    trust_scores = []
    for record, _, _ in top_100:
        trust, _, _ = HoneypotDetector.evaluate(record)
        trust_scores.append(trust)
    avg_trust = np.mean(trust_scores) if trust_scores else 0.0
    print(f"Average Trust Score in Top 100: {avg_trust:.2f}/100.0 (Filtered Rate: 0.0% honeypots)")

    engagement_scores = [breakdown.behavioral_modifier for _, _, breakdown in top_100]
    avg_engagement = np.mean(engagement_scores) if engagement_scores else 0.0
    print(f"Average Recruiter Engagement (Behavioral) Score in Top 100: {avg_engagement:.2f}/100.0")

    # Experience Alignment Score Diagnostics
    exp_align_scores = []
    for record, _, _ in top_100:
        score = ExperienceAlignmentEngine.calculate_score(
            yoe=record.profile.years_of_experience,
            min_exp=jd_graph.constraints.min_exp,
            max_exp=jd_graph.constraints.max_exp,
            ideal_min=jd_graph.constraints.ideal_min_exp,
            ideal_max=jd_graph.constraints.ideal_max_exp
        )
        exp_align_scores.append(score)
        
    if exp_align_scores:
        print(f"Experience Alignment Score in Top 100:")
        print(f"  - Min   : {np.min(exp_align_scores):.2f}/100.0")
        print(f"  - Max   : {np.max(exp_align_scores):.2f}/100.0")
        print(f"  - Mean  : {np.mean(exp_align_scores):.2f}/100.0")
        print(f"  - Median: {np.median(exp_align_scores):.2f}/100.0")

    # Keyword stuffing penalty counts
    minor_penalties = 0
    moderate_penalties = 0
    severe_penalties = 0
    
    # Top ranking signals (core capabilities)
    has_retrieval_count = 0
    has_ranking_count = 0
    has_rec_count = 0
    has_semantic_count = 0
    has_prod_ml_count = 0
    
    for record, _, _ in top_100:
        intel_res = CareerIntelligenceEngine.evaluate(record)
        pen = round(intel_res.stuffing_penalty, 2)
        if pen == 0.05:
            minor_penalties += 1
        elif pen == 0.10:
            moderate_penalties += 1
        elif pen == 0.15:
            severe_penalties += 1
            
        if intel_res.retrieval_score >= 60.0:
            has_retrieval_count += 1
        if intel_res.ranking_score >= 60.0:
            has_ranking_count += 1
        if intel_res.recommendation_score >= 60.0:
            has_rec_count += 1
        if (intel_res.embeddings_score >= 60.0 or 
            intel_res.vector_db_score >= 60.0 or 
            intel_res.search_infra_score >= 60.0):
            has_semantic_count += 1
        if intel_res.production_ml_score >= 60.0:
            has_prod_ml_count += 1
            
    print(f"\nKeyword Stuffing Penalties in Top 100:")
    print(f"  - Minor (-5%): {minor_penalties} candidates")
    print(f"  - Moderate (-10%): {moderate_penalties} candidates")
    print(f"  - Severe (-15%): {severe_penalties} candidates")
    
    print(f"\nTop Ranking Signals (Core Capabilities in Top 100):")
    print(f"  - Retrieval Systems: {has_retrieval_count} candidates")
    print(f"  - Ranking Systems: {has_ranking_count} candidates")
    print(f"  - Recommendation Systems: {has_rec_count} candidates")
    print(f"  - Semantic Search: {has_semantic_count} candidates")
    print(f"  - Production ML: {has_prod_ml_count} candidates")

    # Top domains represented in top 100
    domain_counts = {}
    for d in jd_graph.domains:
        domain_counts[d.domain] = 0
        
    for record, _, _ in top_100:
        exp_text = " ".join([job.description for job in record.career_history]).lower()
        for d_node in jd_graph.domains:
            keywords = JDGraphParser.DOMAIN_TAXONOMY.get(d_node.domain, [])
            if any(kw in exp_text for kw in keywords):
                domain_counts[d_node.domain] += 1
                
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop Domains in Top 100:")
    for dom, count in sorted_domains:
        print(f"  - {dom}: {count} candidates")

    # Top skills represented in top 100
    skill_counts = {}
    for record, _, _ in top_100:
        for s in record.skills:
            s_name = s.name
            skill_counts[s_name] = skill_counts.get(s_name, 0) + 1
            
    sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Skills in Top 100:")
    for skill, count in sorted_skills[:10]:
        print(f"  - {skill}: {count} candidates")
    print("="*40 + "\n")

    return top_100
