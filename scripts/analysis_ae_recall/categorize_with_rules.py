#!/usr/bin/env python3
"""
Rule-based Categorization Script
"""
from dotenv import load_dotenv
import argparse
import json
import logging
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from categorize_with_llm import MDRClassification, format_mdr, load_fda_results

DEATH_KEYPHRASES = [
    "death",
    "died",
    "passed away",
    "expired",
    "could not be resuscitated",
    "deaths",
    "die"
]
INJURY_KEYPHRASES = [
    "injury",
    "injured",
    "serious",
    "needs surgery",
]

def categorize_event_rule_based(event: Dict):
    event_text = format_mdr(event.get("mdr_texts", []))
    is_death = False
    for death_keyphrase in DEATH_KEYPHRASES:
        if death_keyphrase in event_text:
            is_death = True
            break
    if is_death:
        return MDRClassification(reasoning="", event_type="death", fda_device_problem_codes=[])
    
    is_injury = False
    for injury_keyphrase in INJURY_KEYPHRASES:
        if injury_keyphrase in event_text:
            is_injury = True
            break
    if is_injury:
        return MDRClassification(reasoning="", event_type="injury", fda_device_problem_codes=[])
    else:
        return MDRClassification(reasoning="", event_type="malfunction", fda_device_problem_codes=[])

def categorize_events_rule_based(fda_results: List[Dict]) -> List[Dict]:
    """Categorize adverse event using rules."""
    agreements = []
    for r in fda_results:
        if r.get('has_ae'):
            for event in r.get('adverse_events', []):
                event["rule_analysis"] = categorize_event_rule_based(event).model_dump()
                agreement = event["event_type"].lower() == event["rule_analysis"]["event_type"]
                agreements.append(agreement)
    return fda_results, agreements

def main():
    parser = argparse.ArgumentParser(description="LLM Categorization Script")
    parser.add_argument("--input", required=True,
                       help="Path to JSON file with FDA results")
    parser.add_argument("--device-names-jsonl", required=True,
                       help="Path to JSON file with device names")
    parser.add_argument("--output", required=False, default="output/adverse_events_analysis_results_rules.json",
                       help="Path to LLM extraction results")
    parser.add_argument("--log-file", type=str, default="output/log_categorize_with_llm.txt")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=args.log_file)
    
    fda_results = load_fda_results(args.input, args.device_names_jsonl)

    results, agreements = categorize_events_rule_based(fda_results)
    print("agreement rate", np.mean(agreements))
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
