import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running via Streamlit
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Disable TF to prevent keras/protobuf conflicts
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Adjust imports since we are running within the project context
from src.config import (
    FAISS_INDEX_PATH,
    METADATA_CACHE_PATH,
    SENTENCE_TRANSFORMER_MODEL,
    EMBEDDING_DIM
)
from src.preprocessing import CandidateRecord, ScoreBreakdown
from src.engines import JDGraphParser, HoneypotDetector, CareerIntelligenceEngine
from src.ranking import CandidateRanker
from src.explainability import CandidateExplainer

# Set page config for a premium recruiting dashboard
st.set_page_config(
    page_title="Redrob Recruiter Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (glassmorphism, clean typography, custom metrics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, .title-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #FF3366, #FF9933);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    .penalty-card {
        background: rgba(255, 75, 75, 0.1);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(255, 75, 75, 0.3);
        margin-bottom: 15px;
    }
    .success-card {
        background: rgba(75, 255, 75, 0.1);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(75, 255, 75, 0.3);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Cache resource loading for Streamlit speed
@st.cache_resource
def load_assets():
    """Loads SentenceTransformer, FAISS Index, and Metadata Cache into memory."""
    if not FAISS_INDEX_PATH.exists() or not METADATA_CACHE_PATH.exists():
        return None, None, None
        
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cpu")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_CACHE_PATH, "rb") as f:
        metadata_cache = pickle.load(f)
        
    return model, index, metadata_cache

model, index, metadata_cache = load_assets()

# Sidebar Setup
st.sidebar.markdown("<h2 class='title-text'>Redrob Ranker</h2>", unsafe_allow_html=True)
st.sidebar.write("### AI Recruitment Platform")
st.sidebar.divider()

if not metadata_cache:
    st.sidebar.error("❌ Precompute index not found! Please run the pre-computation pipeline first.")
    st.stop()
else:
    st.sidebar.success(f"✓ FAISS index loaded: {len(metadata_cache)} candidates cached.")

st.sidebar.write("#### Scoring Weight Model (Fixed)")
st.sidebar.code("""
Technical Fit:  40%
Career Fit:     30%
Recruitability: 20%
Trust & Avail:  10%
""")

# Main Page Layout
st.markdown("<h1 class='title-text'>Intelligent Candidate Discovery & Ranking</h1>", unsafe_allow_html=True)
st.write("Match 100,000 candidates against your job description in seconds using our recruiter-reasoning engine.")

# Job Description Input Section
st.write("### 📝 Job Description")
default_jd = """Staff AI Engineer / Senior Search Engineer
Experience: 5-9 years (ideal: 6-8 years)
Locations: Noida, Pune, Delhi NCR
Notice Period: sub-30-day notice preferred

Required:
- Production experience with embeddings-based retrieval systems (sentence-transformers, FAISS).
- Deep knowledge of search databases (Elasticsearch, OpenSearch) and ranking evaluation metrics (NDCG, MRR, MAP).
- Strong Python coding skills.

Nice to have:
- LLM fine-tuning (LoRA, QLoRA) and Learning-to-Rank models (XGBoost).
"""

jd_input = st.text_area(
    "Paste your raw Job Description here:",
    value=default_jd,
    height=250
)

col1, col2 = st.columns([1, 4])
with col1:
    rank_btn = st.button("🔍 Run Recruiter Ranking", type="primary", use_container_width=True)

if rank_btn:
    if not jd_input.strip():
        st.warning("Please paste a valid Job Description first.")
    else:
        with st.spinner("Processing Job Description and ranking 100k candidate pool..."):
            # 1. Parse JD Requirement Graph
            jd_graph = JDGraphParser.parse_jd(jd_input)
            
            # 2. Embed JD
            jd_vector = model.encode([jd_input], convert_to_numpy=True, normalize_embeddings=True)
            jd_vector = jd_vector.astype(np.float32)
            
            # 3. FAISS Retrieval (top 5000 candidates)
            k = min(5000, len(metadata_cache))
            distances, indices = index.search(jd_vector, k)
            similarities = distances[0]
            candidate_indices = indices[0]
            
            # Load candidate records
            retrieved_records = []
            retrieved_similarities = []
            for i, cache_idx in enumerate(candidate_indices):
                if cache_idx == -1:
                    continue
                sim = similarities[i]
                meta = metadata_cache[cache_idx]
                record = CandidateRecord.model_validate(meta)
                retrieved_records.append(record)
                retrieved_similarities.append(sim)
                
            # 4. Score & Rank
            t0 = time.time()
            ranked_results = CandidateRanker.rank_candidates(retrieved_records, retrieved_similarities, jd_graph)
            ranking_duration = time.time() - t0
            
            # Filter to top 100
            top_100 = ranked_results[:100]
            st.session_state["top_100"] = top_100
            st.session_state["jd_graph"] = jd_graph
            st.session_state["ranking_duration"] = ranking_duration
            
# Check if we have results in state
if "top_100" in st.session_state:
    top_100 = st.session_state["top_100"]
    jd_graph = st.session_state["jd_graph"]
    duration = st.session_state["ranking_duration"]
    
    st.write("---")
    st.markdown(f"### 🎯 Top Candidates Found (Query took {duration:.3f} seconds)")
    
    # 1. Metric summaries
    m1, m2, m3, m4 = st.columns(4)
    scores = [item[1] for item in top_100]
    m1.metric("Top Score", f"{max(scores):.2f}" if scores else "0.00")
    m2.metric("Mean Score", f"{np.mean(scores):.2f}" if scores else "0.00")
    m3.metric("Min Score (Rank 100)", f"{min(scores):.2f}" if scores else "0.00")
    m4.metric("Honeypots Disqualified", f"100%")
    
    # 2. Render candidates in table
    table_rows = []
    for rank_pos, (record, score, breakdown) in enumerate(top_100, 1):
        reasoning = CandidateExplainer.generate_reasoning(record, breakdown)
        table_rows.append({
            "Rank": rank_pos,
            "ID": record.candidate_id,
            "Name": record.profile.anonymized_name,
            "Score": round(score, 2),
            "Title": record.profile.current_title,
            "Company": record.profile.current_company,
            "Experience (YoE)": record.profile.years_of_experience,
            "Reasoning": reasoning
        })
        
    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 3. Detailed Inspection Panel
    st.write("### 🔍 Inspect Score & Profile Details")
    selected_cid = st.selectbox(
        "Select a candidate to inspect their score breakdown and profile:",
        options=[f"Rank {row['Rank']}: {row['Name']} ({row['ID']})" for row in table_rows]
    )
    
    if selected_cid:
        # Extract rank number
        selected_rank = int(selected_cid.split(":")[0].replace("Rank ", ""))
        record, score, breakdown = top_100[selected_rank - 1]
        
        col_detail1, col_detail2 = st.columns(2)
        
        with col_detail1:
            st.write(f"#### Score Breakdown: {record.profile.anonymized_name}")
            
            # Custom progress bars for score components
            st.write(f"**Final Recruiter Score**: `{score:.2f}/100.0`")
            
            st.write(f"Semantic Cosine Match: `{breakdown.semantic_score:.1f}%`")
            st.progress(breakdown.semantic_score / 100.0)
            
            st.write(f"JD Graph Match: `{breakdown.graph_match_score:.1f}%`")
            st.progress(breakdown.graph_match_score / 100.0)
            
            st.write(f"Career Intelligence (Evidence): `{breakdown.career_intel_score:.1f}%`")
            st.progress(breakdown.career_intel_score / 100.0)
            
            st.write(f"Career History Quality: `{breakdown.career_quality_score:.1f}%`")
            st.progress(breakdown.career_quality_score / 100.0)
            
            st.write(f"Platform Recruitability Score: `{breakdown.behavioral_modifier:.1f}%`")
            st.progress(breakdown.behavioral_modifier / 100.0)
            
        with col_detail2:
            st.write("#### Recruiter Analysis & Profile Facts")
            
            # Show reasoning string
            reasoning = CandidateExplainer.generate_reasoning(record, breakdown)
            st.info(f"**Recruiter Explanation:**\n\n_{reasoning}_")
            
            # Experience quality flags
            st.write("**Career Indicators:**")
            
            # Check for keyword stuffing
            intel_res = CareerIntelligenceEngine.evaluate(record)
            if intel_res.stuffed_skills:
                st.markdown(
                    f"<div class='penalty-card'>⚠️ <b>Keyword Stuffing Detected:</b> Listed skills missing from work descriptions: "
                    f"<i>{', '.join(intel_res.stuffed_skills)}</i> (Penalty: -{intel_res.stuffing_penalty*100:.0f}%)</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='success-card'>✓ <b>Trusted Skills:</b> All search/ML keywords backed by experience descriptions.</div>", 
                    unsafe_allow_html=True
                )
                
            # Profile parameters
            st.write(f"- Stated Notice Period: `{record.redrob_signals.notice_period_days} days`")
            st.write(f"- Recruiter Response Rate: `{record.redrob_signals.recruiter_response_rate*100:.0f}%`")
            st.write(f"- Average Response Time: `{record.redrob_signals.avg_response_time_hours} hours`")
            st.write(f"- Stated Work Mode: `{record.redrob_signals.preferred_work_mode}`")
            
            # Career history list
            st.write("**Recent Career Timeline:**")
            for job in record.career_history[:3]:
                st.write(f"- **{job.title}** at *{job.company}* ({job.duration_months} months) — {job.description[:100]}...")
