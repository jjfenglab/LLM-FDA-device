import sys
import json
import argparse
import asyncio
import threading
import pandas as pd
import os
import re
import logging
from dotenv import load_dotenv

from queue import Queue
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm
import nltk
from nltk.util import ngrams

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.common import load_jsonl
from scripts.utils.pdf_utils import extract_text_from_pdf
from scripts.analysis_annotation_studies.summarize_annotation_results import load_annotations
from survey_validation_trends_all_devices import load_device_numbers, check_device_availability
from compare_llm_results_with_previous_paper import load_validation_data

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

PHRASE_SET_DICT = {
    "is_multisite": set([
        'clinical sites',
        'multiple sites',
        'multiple centers',
        'multiple hospitals',
        'multisite',
        'multicenter',
        'multi-site',
        'multi-center',
        'clinical sites',
    ]),
    "is_prospective": set([
        'prospective',
        'randomized controlled trial',
        'randomized control trial',
    ]),
    "has_clinical_testing": set([
        'clinical testing',
        'clinical site',
        'multiple sites',
        'clinical center',
        'hospital',
        'patient data',
    ]),
    "human_device_team_testing": set([
        "human device team testing",
        "reader study",
    ]),
    "intended_use_and_clinical_applications": set([
        "different intended use",
        "expanded indication",
    ]),
    "operational_and_workflow_change": set([
        "different operation",
        "improved operation",
        "updated operation",
        "different workflow",
        "improved workflow",
        "updated workflow",
    ]),
    "algorithm_or_software_feature_changes": set([
        "different algorithm",
        "improved algorithm",
        "new algorithm",
        "updated algorithm",
        "retrained algorithm",
        "modified algorithm",
        "different software",
        "improved software",
        "new software",
        "updated software",
        "modified software",
        "new feature",
    ]),
    "hardware_changes": set([
        "different hardware",
        "improved hardware",
        "new hardware",
        "modified hardware",
        "added hardware",
    ]),
    "body_part_changes": set([
        "different body part",
        "different body region",
    ]),
}
REGEX_DICT = {
    k: re.compile(r"\b(" + "|".join(re.escape(phrase) for phrase in v) + r")\b", re.IGNORECASE)
    for k, v in PHRASE_SET_DICT.items()
}

def process_device_rule_based(device_number):
    pdf_path = PROJECT_ROOT / f'data/raw/device_summaries/{device_number}.pdf'        
    if not pdf_path.exists():
        print(f"PDF not found for device {device_number} at {pdf_path}, skipping...")
        return

    pdf_text = extract_text_from_pdf(str(pdf_path))
    extraction_dict = {'device_number': device_number}
    for k, regex in REGEX_DICT.items():
        extraction_dict[k] = bool(regex.search(pdf_text))
    return extraction_dict

def get_common_ngrams(args):
    llm_extract_dicts = load_jsonl(args.llm_output_file)

    KEYPHRASES = [
        'The summary explicitly states',
        'The summary states',
        'The text states',
        'The summary mentions'
    ]
    pattern = r"'([^\"]*)'"
    multisite_ngrams = []
    prospective_ngrams = []
    for llm_extract in llm_extract_dicts:
        if llm_extract['is_multisite']:
            for keyphrase in KEYPHRASES:
                if keyphrase in llm_extract['num_sites_reason']:
                    llm_reason = llm_extract['num_sites_reason']
                    extracted_strings = re.findall(pattern, llm_reason)
                    for str_extract in extracted_strings:
                        tokens = nltk.word_tokenize(str_extract)
                        print("bigrams", list(ngrams(tokens, 2)))
                        # multisite_ngrams += list(ngrams(tokens, 1))
                        multisite_ngrams += list(ngrams(tokens, 2))
                        # multisite_ngrams += list(ngrams(tokens, 3))
        if llm_extract['is_prospective']:
            for keyphrase in KEYPHRASES:
                if keyphrase in llm_extract['is_prospective_reason']:
                    llm_reason = llm_extract['is_prospective_reason']
                    extracted_strings = re.findall(pattern, llm_reason)
                    for str_extract in extracted_strings:
                        tokens = nltk.word_tokenize(str_extract)
                        print("bigrams", list(ngrams(tokens, 2)))
                        # prospective_ngrams += list(ngrams(tokens, 1))
                        prospective_ngrams += list(ngrams(tokens, 2))
                        # prospective_ngrams += list(ngrams(tokens, 3))
    print("multisite_ngrams")
    print(pd.value_counts(multisite_ngrams).iloc[:30])
    print("prospective_ngrams")
    print(pd.value_counts(prospective_ngrams).iloc[:30])

def main(args):
    # Investigate common phrases in multi-site validations
    get_common_ngrams(args)

    # Load device numbers
    validation_df = load_validation_data()
    device_annotations = load_annotations(args.annotation_files)
    available_devices = validation_df['approval_number'].tolist() + device_annotations['device number'].tolist()

    all_rule_extractions = []
    for device_num in tqdm(available_devices):
        extract_res = process_device_rule_based(device_num)
        print(extract_res)
        all_rule_extractions.append(extract_res)
    all_extract_strs = [json.dumps(extract_dict) for extract_dict in all_rule_extractions]
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(all_extract_strs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create rule based comparator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--llm-output-file",
        type=str,
        default="output/aiml_devices_validation_results.jsonl",
        help="Path to the LLM output JSONL file."
    )
    parser.add_argument("--annotation_files", nargs="+", 
                       default=[
                        "../analysis_annotation_studies/annotated_data/annotator1_xiao.csv",
                        "../analysis_annotation_studies/annotated_data/annotator2_patrick.csv", 
                        "../analysis_annotation_studies/annotated_data/annotator3_jean.csv",
                        "../analysis_annotation_studies/annotated_data/annotator4_adarsh.csv",
                        ],
                       help="Paths to annotator CSV files")
    parser.add_argument(
        "--input-file",
        type=str,
        default="../../data/aiml_device_numbers_071025.json",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(OUTPUT_DIR / "aiml_device_rule_extractions.jsonl"),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(OUTPUT_DIR / "log_rule_based.txt"),
    )
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=args.log_file)
    
    main(args)