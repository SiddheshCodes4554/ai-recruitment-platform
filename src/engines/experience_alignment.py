"""
Experience Alignment Engine Module.

Computes a non-linear Gaussian bell-curve scoring model based on candidate
years of experience relative to JD ranges.
"""

import math
from src.config import EXP_ALIGN_SIGMA_LEFT, EXP_ALIGN_SIGMA_RIGHT

__all__ = ["ExperienceAlignmentEngine"]

class ExperienceAlignmentEngine:
    """
    Experience Alignment Engine.
    Models human recruiter preference for candidate years of experience relative to
    minimum, maximum, and ideal job description experience parameters.
    Uses a non-linear dual-sided Gaussian bell curve to smoothly decay scores outside the ideal range.
    """

    @classmethod
    def calculate_score(
        cls,
        yoe: float,
        min_exp: float,
        max_exp: float,
        ideal_min: float,
        ideal_max: float
    ) -> float:
        """Computes the Experience Alignment Score (0-100) using a smooth dual-sided Gaussian decay.
        
        Args:
            yoe (float): Candidate years of experience.
            min_exp (float): Minimum JD experience.
            max_exp (float): Maximum JD experience.
            ideal_min (float): Ideal minimum JD experience.
            ideal_max (float): Ideal maximum JD experience.
            
        Returns:
            float: The calculated Experience Alignment Score (0-100).
        """
        if yoe < 0:
            return 0.0

        if ideal_min <= yoe <= ideal_max:
            return 100.0
            
        if yoe < ideal_min:
            # Decay on the left side of the ideal range
            # Calibration: EXP_ALIGN_SIGMA_LEFT matches:
            # yoe = 5.0 -> ~85.0
            # yoe = 4.0 -> ~52.0
            # yoe = 3.0 -> ~23.0
            score = 100.0 * math.exp(-((yoe - ideal_min) ** 2) / (2.0 * (EXP_ALIGN_SIGMA_LEFT ** 2)))
        else:
            # Decay on the right side of the ideal range
            # Calibration: EXP_ALIGN_SIGMA_RIGHT matches:
            # yoe = 9.0 -> ~92.0
            # yoe = 10.0 -> ~71.7
            # yoe = 12.0 -> ~26.3
            score = 100.0 * math.exp(-((yoe - ideal_max) ** 2) / (2.0 * (EXP_ALIGN_SIGMA_RIGHT ** 2)))

        return round(max(0.0, min(100.0, score)), 2)
