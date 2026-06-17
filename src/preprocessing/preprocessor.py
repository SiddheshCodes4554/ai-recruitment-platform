"""
Profile Preprocessor Module.

Preprocesses raw candidate profiles into clean structured text and metadata caches.
"""

import re
from typing import Dict, Any, List
from src.preprocessing.data_models import CandidateRecord

__all__ = ["ProfilePreprocessor"]

class ProfilePreprocessor:
    """Preprocesses raw candidate profiles into clean structured text and metadata caches."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Removes extra whitespaces, newlines, and normalizes unicode characters.
        
        Args:
            text (str): The raw text to sanitize.
            
        Returns:
            str: The sanitized text.
        """
        if not text:
            return ""
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        return text.strip()

    @staticmethod
    def build_text_representation(record: CandidateRecord) -> str:
        """Synthesizes a candidate profile into a rich, structured natural language text block
        optimized for semantic encoding with Sentence Transformers.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            
        Returns:
            str: The synthesized text block representation of the profile.
        """
        p = record.profile
        
        # 1. Headline and Summary
        headline = ProfilePreprocessor.clean_text(p.headline)
        summary = ProfilePreprocessor.clean_text(p.summary)
        
        text_parts = [
            f"Candidate Name: {p.anonymized_name}",
            f"Current Title: {p.current_title}",
            f"Current Company: {p.current_company} (Size: {p.current_company_size}, Industry: {p.current_industry})",
            f"Headline: {headline}",
            f"Professional Summary: {summary}",
            f"Total Years of Experience: {p.years_of_experience} years",
            f"Location: {p.location}, {p.country}"
        ]

        # 2. Career History
        history_parts = []
        for i, job in enumerate(record.career_history, 1):
            comp = job.company
            title = job.title
            dur = job.duration_months
            desc = ProfilePreprocessor.clean_text(job.description)
            is_curr = "Current Role" if job.is_current else f"Past Role (Ended: {job.end_date})"
            
            history_parts.append(
                f"Role {i}: {title} at {comp} ({dur} months, {is_curr}). "
                f"Industry: {job.industry}, Size: {job.company_size}. "
                f"Job Description: {desc}"
            )
            
        if history_parts:
            text_parts.append("Work Experience:\n" + "\n".join(history_parts))

        # 3. Skills
        skills_parts = []
        for s in record.skills:
            dur_str = f" for {s.duration_months} months" if s.duration_months else ""
            skills_parts.append(f"{s.name} ({s.proficiency} proficiency{dur_str})")
            
        if skills_parts:
            text_parts.append("Skills: " + ", ".join(skills_parts))

        # 4. Education
        edu_parts = []
        for edu in record.education:
            grade_str = f", Grade: {edu.grade}" if edu.grade else ""
            edu_parts.append(
                f"{edu.degree} in {edu.field_of_study} from {edu.institution} "
                f"(Years: {edu.start_year}-{edu.end_year}, Tier: {edu.tier}{grade_str})"
            )
            
        if edu_parts:
            text_parts.append("Education:\n" + "\n".join(edu_parts))

        # 5. Certifications
        if record.certifications:
            certs = [f"{c.name} by {c.issuer} ({c.year})" for c in record.certifications]
            text_parts.append("Certifications: " + ", ".join(certs))

        # Combine all parts into a clean single text block
        return "\n\n".join(text_parts)

    @staticmethod
    def extract_metadata_cache(record: CandidateRecord) -> Dict[str, Any]:
        """Dumps the full validated CandidateRecord dictionary to preserve all fields
        for online consistency scoring and reasoning.
        
        Args:
            record (CandidateRecord): The candidate profile record.
            
        Returns:
            Dict[str, Any]: The Pydantic model dumped as a dictionary.
        """
        return record.model_dump()
