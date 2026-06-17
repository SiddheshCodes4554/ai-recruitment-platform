#!/usr/bin/env python3
"""
Intelligent Candidate Discovery & Ranking CLI Runner.
"""

import argparse
import sys
from pathlib import Path
from src.ranking.ranking_pipeline import run_ranking_pipeline
from src.utils.logger import get_logger

logger = get_logger("cli")

def main() -> None:
    """Main execution point for the candidate discovery CLI."""
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Intelligent Candidate Discovery & Ranking CLI"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to the candidates.jsonl dataset file"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to write the ranked submission CSV output file"
    )
    
    args = parser.parse_args()
    
    candidates_path = Path(args.candidates)
    out_path = Path(args.out)
    
    try:
        run_ranking_pipeline(candidates_path, out_path)
    except Exception as e:
        logger.error(f"Error executing ranking pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
