"""
Honeypot Detector Module.

Detects logical contradictions, timeline anomalies, and unrealistic expertise
claims in candidate profiles to identify synthetic honeypot traps.
"""

import json
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any
from src.preprocessing.data_models import CandidateRecord

__all__ = ["HoneypotDetector"]

class HoneypotDetector:
    """
    Profile Trust & Consistency Engine.
    Detects logical contradictions, timeline anomalies, and unrealistic expertise
    claims in candidate profiles to identify synthetic honeypot traps.
    """

    # Reference date representing the dataset's current date (June 17, 2026)
    CURRENT_DATE = datetime(2026, 6, 17)

    @classmethod
    def parse_date(cls, date_str: str) -> datetime:
        """Parses a date string in YYYY-MM-DD format, returning datetime object.
        
        Args:
            date_str (str): The date string to parse.
            
        Returns:
            datetime: The parsed datetime object, or CURRENT_DATE if None/invalid.
        """
        if not date_str:
            return cls.CURRENT_DATE
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return cls.CURRENT_DATE

    @classmethod
    def evaluate(cls, record: CandidateRecord) -> Tuple[float, float, List[str]]:
        """Evaluates a candidate profile for trust and consistency.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            
        Returns:
            Tuple[float, float, List[str]]: A tuple containing:
                - trust_score (float): Score from 0 to 100 (100 = perfectly consistent).
                - anomaly_score (float): Score from 0 to 100 (100 - trust_score).
                - reasons (List[str]): Explanations of any detected anomalies.
        """
        trust_score = 100.0
        reasons: List[str] = []
        p = record.profile

        # 1. Experience Contradiction (Profile YoE vs Sum of Career History)
        profile_yoe = p.years_of_experience
        total_history_months = sum(job.duration_months for job in record.career_history)
        calc_years = total_history_months / 12.0
        
        # Mismatch greater than 1.0 year is a major anomaly
        yoe_diff = abs(profile_yoe - calc_years)
        if yoe_diff > 1.0:
            reasons.append(
                f"Years of Experience Mismatch: profile claims {profile_yoe} years, "
                f"but career history durations sum to {calc_years:.2f} years (difference of {yoe_diff:.2f} years)"
            )
            trust_score -= 50.0

        # 2. Timeline Inconsistencies (Job Duration vs Date Differences)
        for job in record.career_history:
            start_dt = cls.parse_date(job.start_date)
            end_dt = cls.parse_date(job.end_date) if job.end_date else cls.CURRENT_DATE
            dur_months = job.duration_months
            
            # Chronology check: start date after end date
            if start_dt > end_dt:
                reasons.append(
                    f"Timeline Chronology Error in job at {job.company}: "
                    f"start_date ({job.start_date}) is after end_date ({job.end_date or 'Present'})"
                )
                trust_score -= 50.0
                continue
                
            # Date span mismatch
            actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
            # Allow minor variance of 2 months for rounding/boundary offsets
            if abs(actual_months - dur_months) > 2:
                reasons.append(
                    f"Job Duration Mismatch at {job.company}: "
                    f"record lists {dur_months} months, but date range spans {actual_months} months"
                )
                trust_score -= 50.0

        # 3. Education Chronology Issues
        for edu in record.education:
            start_yr = edu.start_year
            end_yr = edu.end_year
            if start_yr > end_yr:
                reasons.append(
                    f"Education Chronology Error at {edu.institution}: "
                    f"start_year ({start_yr}) is after end_year ({end_yr})"
                )
                trust_score -= 50.0

        # 4. Skill-Duration Inconsistencies & Unrealistic Expertise Claims
        # Checks if skills listed as expert or advanced have 0 duration_months
        expert_zero_dur_skills = []
        for s in record.skills:
            if s.proficiency in ["expert", "advanced"] and s.duration_months == 0:
                expert_zero_dur_skills.append(s.name)
                
        num_anomalous_skills = len(expert_zero_dur_skills)
        if num_anomalous_skills > 0:
            if num_anomalous_skills >= 3:
                # 3 or more expert/advanced skills with 0 duration is highly unrealistic (unrealistic claims)
                reasons.append(
                    f"Unrealistic Expertise Claims: candidate claims expert/advanced proficiency "
                    f"in {num_anomalous_skills} skills but has 0 months of usage: {expert_zero_dur_skills}"
                )
                trust_score -= 50.0
            else:
                # Minor penalty for 1-2 skills
                reasons.append(
                    f"Suspicious Skill Duration: expert/advanced proficiency claimed in "
                    f"{expert_zero_dur_skills} with 0 months of usage"
                )
                trust_score -= 15.0 * num_anomalous_skills

        # 5. Redrob Platform Signals Chronology
        signals = record.redrob_signals
        signup_dt = cls.parse_date(signals.signup_date)
        last_active_dt = cls.parse_date(signals.last_active_date)
        
        if signup_dt > last_active_dt:
            reasons.append(
                f"Platform Activity Inconsistency: signup_date ({signals.signup_date}) "
                f"is after last_active_date ({signals.last_active_date})"
            )
            trust_score -= 40.0
            
        if signup_dt > cls.CURRENT_DATE:
            reasons.append(f"Future Platform Signup: signup_date ({signals.signup_date}) is in the future")
            trust_score -= 40.0
            
        if last_active_dt > cls.CURRENT_DATE:
            reasons.append(f"Future Platform Activity: last_active_date ({signals.last_active_date}) is in the future")
            trust_score -= 40.0

        # 6. Excessive Keyword Stuffing check
        # Evaluate how many skills are "stuffed" (listed in skills but missing from experience descriptions)
        experience_text = " ".join([
            p.headline, p.summary, p.current_title,
            " ".join([f"{job.title} {job.description}" for job in record.career_history])
        ]).lower()
        
        stuffed_count = 0
        total_skills_checked = len(record.skills)
        
        for s in record.skills:
            skill_clean = s.name.lower()
            if skill_clean not in experience_text:
                stuffed_count += 1
                
        if total_skills_checked > 0:
            stuffing_ratio = stuffed_count / total_skills_checked
            if stuffing_ratio > 0.6 and total_skills_checked >= 8:
                reasons.append(
                    f"Excessive Keyword Stuffing: {stuffed_count} out of {total_skills_checked} "
                    f"skills ({stuffing_ratio * 100:.1f}%) have no supporting evidence in career history"
                )
                trust_score -= 20.0

        # Clamp scores
        final_trust_score = max(0.0, min(100.0, trust_score))
        anomaly_score = 100.0 - final_trust_score

        return final_trust_score, anomaly_score, reasons

    @classmethod
    def is_honeypot(cls, record: CandidateRecord) -> Tuple[bool, List[str]]:
        """Determines if a candidate is a honeypot trap based on consistency evaluation.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            
        Returns:
            Tuple[bool, List[str]]: A tuple containing:
                - is_trap (bool): True if candidate is flagged as honeypot.
                - reasons (List[str]): List of detected anomaly reasons.
        """
        trust_score, _, reasons = cls.evaluate(record)
        
        # Check for major logic contradictions (indicated by reasons containing specific terms)
        has_major_anomaly = False
        for r in reasons:
            if "Mismatch" in r or "Chronology Error" in r or "Unrealistic Expertise" in r or "Contradiction" in r:
                has_major_anomaly = True
                break
                
        # Classify as honeypot if trust is low or a major anomaly was flagged
        is_trap = (trust_score < 60.0) or has_major_anomaly
        return is_trap, reasons
