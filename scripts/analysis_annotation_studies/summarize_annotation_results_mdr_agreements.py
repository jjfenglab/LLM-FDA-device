import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.common import wald_confidence_interval, format_ci_string
from scripts.analysis_annotation_studies.summarize_annotation_results_mdr import load_annotations, load_random_orderings


def load_extractions(rule_extract_path, extract_key):
    """Load rule-based extractions from JSON file"""
    with open(rule_extract_path, 'r') as f:
        data = json.load(f)

    # Create mapping from report_number to extraction
    extraction_dict = {}
    for device in data:
        if device.get('has_ae'):
            for event in device.get('adverse_events', []):
                report_num = event.get('report_number')
                if report_num and extract_key in event:
                    extraction_dict[report_num] = event[extract_key]

    return extraction_dict


def determine_ground_truth(annotations_df):
    """Determine ground truth labels based on human preferences"""

    # Determine ground truth for each field
    ground_truth = {}

    for index, row in annotations_df.iterrows():
        report_num = row['report number']
        ground_truth[report_num] = {}

        # Determine event_type ground truth
        if row['event_type'] == 'Same':
            # If same, use either option (they're identical)
            ground_truth[report_num]['event_type'] = row['event_1']
        elif row['event_type'] == 'Option 1 better':
            ground_truth[report_num]['event_type'] = row['event_1']
        else:  # Option 2 better
            ground_truth[report_num]['event_type'] = row['event_2']

    return ground_truth


def evaluate_rule_agreement(ground_truth, rule_extractions):
    """Evaluate how often rule-based extractions agree with ground truth"""

    agreements = {
        'event_type': {'agree': 0, 'disagree': 0, 'missing': 0}
    }

    disagreement_examples = []

    for report_num, gt_labels in ground_truth.items():
        if report_num not in rule_extractions:
            agreements['event_type']['missing'] += 1
            continue

        rule_labels = rule_extractions[report_num]

        # Compare event_type
        gt_event = gt_labels.get('event_type', '').lower().strip()
        rule_event = rule_labels.get('event_type', '').lower().strip()

        if gt_event == rule_event:
            agreements['event_type']['agree'] += 1
        else:
            agreements['event_type']['disagree'] += 1
            disagreement_examples.append({
                'report_number': report_num,
                'ground_truth': gt_event,
                'rule_based': rule_event
            })

    # Calculate agreement percentages
    total_event = agreements['event_type']['agree'] + agreements['event_type']['disagree']
    event_pct = agreements['event_type']['agree'] / total_event if total_event > 0 else 0

    # Calculate confidence intervals
    event_ci = wald_confidence_interval(agreements['event_type']['agree'], total_event)

    results = {
        'field': ['event_type'],
        'agree': [agreements['event_type']['agree']],
        'disagree': [agreements['event_type']['disagree']],
        'missing': [agreements['event_type']['missing']],
        'total': [total_event],
        'agreement_pct': [event_pct],
        'ci_lower': [event_ci[1]],
        'ci_upper': [event_ci[2]],
        'ci_string': [format_ci_string(event_ci[0], event_ci[1], event_ci[2])]
    }

    return pd.DataFrame(results), disagreement_examples


def main():
    parser = argparse.ArgumentParser(description="Compare rule-based annotations with human annotations")
    parser.add_argument("--annotation", nargs="*", required=True,
                       help="Path(s) to human annotation CSV file(s)")
    parser.add_argument("--event-type-extract-file", type=str,
                       help="Path to event type extraction JSON file")
    parser.add_argument("--extraction-key", type=str,
                       help="Path to event type extraction JSON file")
    parser.add_argument("--results", required=True,
                       help="Path to save results CSV file")
    parser.add_argument("--disagreements", default=None,
                       help="Path to save disagreement examples CSV file")

    args = parser.parse_args()

    # Load data
    rule_extractions = load_extractions(args.event_type_extract_file, args.extraction_key)
    annotations_df = load_annotations(args.annotation)
    
    print(f"Loaded {len(rule_extractions)} extractions")
    print(f"Loaded {len(annotations_df)} human annotations")
    print(f"Unique MDRs: {annotations_df['report number'].unique().size}")

    # Determine ground truth from human annotations
    ground_truth = determine_ground_truth(annotations_df)
    print(f"Determined ground truth for {len(ground_truth)} reports")

    # Evaluate agreement between rule-based and ground truth
    results_df, disagreement_examples = evaluate_rule_agreement(ground_truth, rule_extractions)

    print("\n" + "="*50)
    print("RULE-BASED ANNOTATION AGREEMENT WITH HUMAN LABELS")
    print("="*50)
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(args.results, index=False)
    print(f"\nResults saved to {args.results}")

    # Save disagreement examples if requested
    if args.disagreements and disagreement_examples:
        disagreements_df = pd.DataFrame(disagreement_examples)
        disagreements_df.to_csv(args.disagreements, index=False)
        print(f"Disagreement examples saved to {args.disagreements}")
        print(f"Total disagreements: {len(disagreement_examples)}")


if __name__ == "__main__":
    main()
