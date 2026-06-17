"""
JD Requirement Graph Module.

Converts raw job descriptions into structured hiring intent graphs,
and matches candidates against the requirement nodes.
"""

import re
from typing import Dict, List, Set, Any, Tuple
from src.preprocessing.data_models import JDRequirementGraph, DomainNode, TechNode, ConstraintNode, CandidateRecord

__all__ = ["JDGraphParser"]

class JDGraphParser:
    """
    Parses raw job description text into a structured JDRequirementGraph
    representing recruitment constraints, required domains, tech skills, and negative patterns.
    """

    # Domain keywords taxonomies
    DOMAIN_TAXONOMY: Dict[str, List[str]] = {
        "Retrieval": ["retrieval", "vector search", "dense passage retrieval", "dpr", "information retrieval", "hybrid search"],
        "Ranking": ["ranking", "learning to rank", "ltr", "re-ranking", "re-rank", "ndcg", "mrr", "map", "relevance score"],
        "Recommendation": ["recommendation", "recommender", "rec-sys", "collaborative filtering", "personalization"],
        "Search": ["search engine", "elasticsearch", "opensearch", "solr", "lucene", "hybrid search", "search infrastructure"],
        "NLP": ["nlp", "natural language processing", "text processing", "tokenization", "bert", "text representation"],
        "LLM": ["llm", "large language model", "fine-tuning", "lora", "qlora", "peft", "rag", "generative ai"],
        "MLOps": ["mlops", "production", "monitoring", "latency", "scale", "docker", "kubernetes", "drift detection"]
    }

    # Technology mapping lists
    TECH_KEYWORDS: Dict[str, str] = {
        "sentence-transformers": "embeddings",
        "sentence_transformers": "embeddings",
        "faiss": "vector_indexing",
        "pinecone": "vector_indexing",
        "qdrant": "vector_indexing",
        "milvus": "vector_indexing",
        "weaviate": "vector_indexing",
        "elasticsearch": "search_engine",
        "opensearch": "search_engine",
        "solr": "search_engine",
        "ndcg": "evaluation",
        "mrr": "evaluation",
        "map": "evaluation",
        "python": "programming",
        "lora": "llm_fine_tuning",
        "qlora": "llm_fine_tuning",
        "peft": "llm_fine_tuning",
        "xgboost": "learning_to_rank",
        "docker": "deployment",
        "kubernetes": "deployment"
    }

    # Centralized magic numbers as class constants
    PENALTY_COMPANY_HOPPER: float = 15.0
    PENALTY_ACADEMIC_ONLY: float = 25.0
    PENALTY_SERVICES_ONLY: float = 30.0
    
    WEIGHT_CONSTRAINT_EXP: float = 0.4
    WEIGHT_CONSTRAINT_LOC: float = 0.3
    WEIGHT_CONSTRAINT_NOTICE: float = 0.3

    @classmethod
    def parse_jd(cls, jd_text: str) -> JDRequirementGraph:
        """Parses JD text using deterministic regex rules to construct the graph nodes.
        
        Args:
            jd_text (str): The raw job description text.
            
        Returns:
            JDRequirementGraph: The structured job requirements.
        """
        cleaned_text = jd_text.lower()

        # 1. Parse Experience range (e.g. 5-9 years, ideal 6-8 years)
        min_exp, max_exp = 5.0, 9.0
        ideal_min_exp, ideal_max_exp = 6.0, 8.0

        # Regex for ranges like "5-9 years" or "5 to 9 years"
        range_match = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)\s*years", cleaned_text)
        if range_match:
            min_exp = float(range_match.group(1))
            max_exp = float(range_match.group(2))
            
        # Regex for ideal ranges like "ideal: 6-8" or "ideal is 6-8" or "ideally 6-8"
        ideal_match = re.search(r"ideal(?:ly|\s+(?:is|being))?\s*(\d+)\s*(?:-|to)\s*(\d+)", cleaned_text)
        if ideal_match:
            ideal_min_exp = float(ideal_match.group(1))
            ideal_max_exp = float(ideal_match.group(2))
        else:
            # Fallback range logic
            ideal_min_exp = min_exp + 1.0 if min_exp + 1.0 < max_exp else min_exp
            ideal_max_exp = max_exp - 1.0 if max_exp - 1.0 > min_exp else max_exp

        # 2. Parse Location Preferences (Noida, Pune, etc.)
        locations = []
        if "noida" in cleaned_text:
            locations.append("Noida")
        if "pune" in cleaned_text:
            locations.append("Pune")
        if "hyderabad" in cleaned_text:
            locations.append("Hyderabad")
        if "mumbai" in cleaned_text:
            locations.append("Mumbai")
        if "delhi" in cleaned_text or "ncr" in cleaned_text:
            locations.append("Delhi NCR")
        if not locations:
            locations = ["Noida", "Pune", "India"]  # Default challenge locations

        # 3. Notice Period Preferences (e.g. "sub-30-day notice", "30 days")
        max_notice_days = 90  # Default limit
        notice_match = re.search(r"(sub-30|30|60|90)\s*-\s*day\s*notice", cleaned_text)
        if notice_match:
            days_str = notice_match.group(1)
            if days_str == "sub-30":
                max_notice_days = 30
            else:
                max_notice_days = int(days_str)
        elif "30 days" in cleaned_text or "30-day" in cleaned_text or "sub-30" in cleaned_text:
            max_notice_days = 30
        elif "60 days" in cleaned_text or "60-day" in cleaned_text:
            max_notice_days = 60

        # 4. Extract active domain nodes and calculate weights
        domain_nodes = []
        for domain, keywords in cls.DOMAIN_TAXONOMY.items():
            # Find mention count of keywords in JD
            mentions = 0
            for kw in keywords:
                if kw in cleaned_text:
                    mentions += 1
            if mentions > 0:
                # Calculate weight: more mentions = higher relevance (capped at 1.0)
                weight = min(1.0, 0.4 + (mentions * 0.2))
                domain_nodes.append(DomainNode(domain=domain, weight=weight))
        
        # Ensure we have at least defaults if JD is short
        if not domain_nodes:
            domain_nodes = [
                DomainNode(domain="Retrieval", weight=1.0),
                DomainNode(domain="Ranking", weight=1.0),
                DomainNode(domain="Search", weight=1.0)
            ]

        # 5. Extract active tech nodes and calculate weights
        tech_nodes = []
        for tech, category in cls.TECH_KEYWORDS.items():
            tech_clean = tech.replace("_", "-")
            if tech_clean in cleaned_text or tech in cleaned_text:
                # Default weight based on role focus (Must-haves vs preferred)
                weight = 0.8
                # Boost weights of absolute must-haves
                if tech in ["faiss", "sentence-transformers", "ndcg", "mrr", "map", "python"]:
                    weight = 1.0
                elif tech in ["lora", "qlora", "peft", "xgboost"]:
                    weight = 0.6  # Preferred nice-to-haves
                tech_nodes.append(TechNode(skill=tech, weight=weight))
                
        if not tech_nodes:
            tech_nodes = [
                TechNode(skill="sentence-transformers", weight=1.0),
                TechNode(skill="faiss", weight=1.0),
                TechNode(skill="ndcg", weight=1.0),
                TechNode(skill="python", weight=1.0)
            ]

        constraint_node = ConstraintNode(
            min_exp=min_exp,
            max_exp=max_exp,
            ideal_min_exp=ideal_min_exp,
            ideal_max_exp=ideal_max_exp,
            locations=locations,
            max_notice_days=max_notice_days
        )

        return JDRequirementGraph(
            domains=domain_nodes,
            tech_skills=tech_nodes,
            constraints=constraint_node
        )

    @classmethod
    def calculate_match_score(cls, record: CandidateRecord, graph: JDRequirementGraph) -> Tuple[float, Dict[str, float]]:
        """Evaluates a candidate record against the parsed JD requirement graph.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            graph (JDRequirementGraph): The parsed job requirements.
            
        Returns:
            Tuple[float, Dict[str, float]]: A tuple containing the overall graph_match_score
            (0-100) and a breakdown dictionary.
        """
        # 1. Evaluate Domains Match (S_domain)
        p = record.profile
        experience_text = " ".join([
            p.headline, p.summary, p.current_title,
            " ".join([f"{job.title} {job.description}" for job in record.career_history])
        ]).lower()

        domain_scores = []
        domain_weights = []
        for d_node in graph.domains:
            # Check if domain keywords are mentioned in experience
            keywords = cls.DOMAIN_TAXONOMY.get(d_node.domain, [])
            mentions = 0
            for kw in keywords:
                if kw in experience_text:
                    mentions += 1
            
            # Domain matched score: 1.0 if any keyword is mentioned, boost if multiple
            match_score = min(1.0, mentions * 0.5)
            domain_scores.append(match_score)
            domain_weights.append(d_node.weight)
            
        s_domain = sum(s * w for s, w in zip(domain_scores, domain_weights)) / sum(domain_weights) if domain_weights else 0.0

        # 2. Evaluate Technology Match (S_tech)
        tech_scores = []
        tech_weights = []
        
        # Build candidate skills map for easy lookup
        skills_map = {s.name.lower(): s for s in record.skills}
        
        for t_node in graph.tech_skills:
            tech_clean = t_node.skill.replace("_", "-").lower()
            score = 0.0
            
            # Check skill list proficiency
            if tech_clean in skills_map:
                prof = skills_map[tech_clean].proficiency
                if prof == "expert":
                    score = 1.0
                elif prof == "advanced":
                    score = 0.8
                elif prof == "intermediate":
                    score = 0.6
                else:
                    score = 0.3
                    
            # Check experience descriptions (add bonus if listed in career history text)
            if tech_clean in experience_text:
                score = min(1.0, score + 0.2)
                
            tech_scores.append(score)
            tech_weights.append(t_node.weight)
            
        s_tech = sum(s * w for s, w in zip(tech_scores, tech_weights)) / sum(tech_weights) if tech_weights else 0.0

        # 3. Evaluate Constraints (S_constraint)
        # 3a. Experience Match Score
        yoe = p.years_of_experience
        c = graph.constraints
        if c.ideal_min_exp <= yoe <= c.ideal_max_exp:
            s_exp = 1.0
        elif c.min_exp <= yoe <= c.max_exp:
            s_exp = 0.8
        elif (c.min_exp - 1.5) <= yoe <= (c.max_exp + 1.5):
            s_exp = 0.5
        else:
            s_exp = 0.1
            
        # 3b. Location Match Score
        cand_loc = p.location.lower()
        cand_country = p.country.lower()
        willing_relocate = record.redrob_signals.willing_to_relocate
        
        loc_matched = False
        for loc in c.locations:
            if loc.lower() in cand_loc:
                loc_matched = True
                break
                
        if loc_matched:
            s_loc = 1.0
        elif willing_relocate and cand_country == "india":
            s_loc = 0.8
        elif cand_country == "india":
            s_loc = 0.5
        else:
            s_loc = 0.2  # Outside India, likely visa required
            
        # 3c. Notice Period Match Score
        notice_days = record.redrob_signals.notice_period_days
        if notice_days <= 30:
            s_notice = 1.0
        elif notice_days <= 60:
            s_notice = 0.7
        elif notice_days <= 90:
            s_notice = 0.4
        else:
            s_notice = 0.1
            
        s_constraint = (cls.WEIGHT_CONSTRAINT_EXP * s_exp) + (cls.WEIGHT_CONSTRAINT_LOC * s_loc) + (cls.WEIGHT_CONSTRAINT_NOTICE * s_notice)

        # 4. Check for Negative Signals (Applied as deductions to prevent simple stuffing)
        penalty = 0.0
        
        # 4a. Company Hopper check
        num_jobs = len(record.career_history)
        total_months = sum(job.duration_months for job in record.career_history)
        avg_tenure = total_months / num_jobs if num_jobs > 0 else 0.0
        if avg_tenure < 18.0 and num_jobs >= 3:
            penalty += cls.PENALTY_COMPANY_HOPPER
            
        # 4b. Pure Academic check
        job_titles = [job.title.lower() for job in record.career_history]
        academic_keywords = ["research assistant", "phd candidate", "postdoc", "professor", "lecturer", "academic researcher"]
        is_academic_only = all(any(kw in title for kw in academic_keywords) for title in job_titles)
        if is_academic_only and num_jobs > 0:
            penalty += cls.PENALTY_ACADEMIC_ONLY
            
        # 4c. Services Only check
        it_services_firms = {"infosys", "wipro", "tcs", "accenture", "capgemini", "cognizant", "hcl", "mindtree", "tech mahindra", "mphasis"}
        is_services_only = all(job.company.lower() in it_services_firms for job in record.career_history)
        if is_services_only and num_jobs > 0:
            penalty += cls.PENALTY_SERVICES_ONLY

        # Calculate final composite match score (out of 100)
        raw_graph_score = ((1.0 * s_domain + 1.2 * s_tech + 0.8 * s_constraint) / 3.0) * 100.0
        final_graph_score = max(0.0, raw_graph_score - penalty)

        breakdown = {
            "domain_score": round(s_domain * 100, 1),
            "tech_score": round(s_tech * 100, 1),
            "constraint_score": round(s_constraint * 100, 1),
            "negative_penalty": round(penalty, 1)
        }

        return round(final_graph_score, 2), breakdown
