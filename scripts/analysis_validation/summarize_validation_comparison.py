import sys
import json
import argparse
import logging

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from scipy import stats
from compare_llm_results_with_previous_paper import load_validation_data

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.common import wald_confidence_interval, wald_confidence_interval_cts, format_ci_string

# Constants
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(results_file: str) -> List[Dict]:
    """Load LLM results from JSONL file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

# Preprocess validation predicate numbers
def parse_predicate_numbers(predicate_str):
    """Parse predicate numbers from validation data, handling empty, single, and multiple values."""
    if pd.isna(predicate_str) or predicate_str == '':
        return []
    # Split by semicolon and clean up
    predicates = [p.strip().upper() for p in str(predicate_str).split(';') if p.strip()]
    return predicates

# Compare predicates
def check_predicate_match(row):
    """Check if LLM primary predicate matches any validation predicate."""
    llm_predicate = str(row['primary_predicate']).upper().strip() if pd.notna(row['primary_predicate']) else ''
    validation_predicates = row['validation_predicates_clean']
    
    if not llm_predicate or not validation_predicates:
        # Both empty/null - consider a match
        return not llm_predicate and not validation_predicates
    
    return llm_predicate in validation_predicates
    
def generate_comparison_report(llm_results: List[Dict], validation_df: pd.DataFrame, output_csv_file: str, output_report_file: str):
    """Compare LLM results with the validation data and generate a report."""
    # Convert LLM results to DataFrame for easier comparison
    llm_df = pd.DataFrame(llm_results)
    
    validation_df['validation_predicates_clean'] = validation_df['predicate_number'].apply(parse_predicate_numbers)
    
    # Merge on device number
    merged_df = pd.merge(
        llm_df,
        validation_df[['approval_number', 'num_sites', 'is_prospective', 'predicate_number', 'validation_predicates_clean']],
        left_on='device_number',
        right_on='approval_number',
        how='inner',
        suffixes=('_llm', '_paper')
    )
    
    # Compare num_sites
    # Handle nan values in paper data and None values in LLM data
    has_num_sites = 'num_sites_paper' in merged_df.columns
    if has_num_sites:
        merged_df['num_sites_paper_clean'] = pd.to_numeric(merged_df['num_sites_paper'], errors='coerce')
        merged_df['num_sites_match'] = (
            (merged_df['num_sites_llm'].isna() & merged_df['num_sites_paper_clean'].isna()) |
            (merged_df['num_sites_llm'].fillna(0).astype(int) == merged_df['num_sites_paper_clean'].fillna(0).astype(int))
        )
        merged_df['num_sites_abs_error'] = np.abs(merged_df['num_sites_llm'].fillna(0).astype(int) - merged_df['num_sites_paper_clean'].fillna(0).astype(int))
    else:
        merged_df['num_sites_paper_clean'] = pd.to_numeric(merged_df['num_sites'], errors='coerce')
    
    # Compare is_prospective
    merged_df['is_prospective_match'] = merged_df['is_prospective_llm'].fillna(0).astype(int) == merged_df['is_prospective_paper'].fillna(0).astype(int)
    
    if 'primary_predicate' in merged_df:
        merged_df['predicate_match'] = merged_df.apply(check_predicate_match, axis=1)

    merged_df.to_csv(output_csv_file)
    
    # Generate performance metrics table
    performance_table = calculate_performance_metrics(merged_df)
    
    # Save performance table to CSV
    performance_csv_file = output_csv_file.replace('.csv', '_performance_metrics.csv')
    performance_table.to_csv(performance_csv_file, index=False)
    
    # Calculate accuracies
    is_prospective_accuracy = merged_df['is_prospective_match'].mean() * 100
    is_prospective_disagreements = merged_df[~merged_df['is_prospective_match']]
    if has_num_sites:
        num_sites_accuracy = merged_df['num_sites_match'].mean() * 100
        num_sites_disagreements = merged_df[~merged_df['num_sites_match']]
        num_sites_mae_ci = wald_confidence_interval_cts(merged_df['num_sites_abs_error'])
    if 'predicate_match' in merged_df:
        predicate_accuracy = merged_df['predicate_match'].mean() * 100
        predicate_disagreements = merged_df[~merged_df['predicate_match']]
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("LLM VALIDATION COMPARISON REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal devices compared: {len(merged_df)}")
    report.append(f"Total devices in validation data: {len(validation_df)}")
    report.append(f"Total devices processed by LLM: {len(llm_df)}")
    report.append(f"Devices successfully matched: {len(merged_df)}")
    
    report.append("\n" + "-" * 40)
    report.append("ACCURACY METRICS")
    report.append("-" * 40)
    if 'predicate_match' in merged_df:
        report.append(f"predicate accuracy: {predicate_accuracy:.2f}% ({merged_df['predicate_match'].sum()}/{len(merged_df)})")
    if has_num_sites:
        report.append(f"num_sites accuracy: {num_sites_accuracy:.2f}% ({merged_df['num_sites_match'].sum()}/{len(merged_df)})")
        report.append(f"num_sites mae: {num_sites_mae_ci[0]:.2f} ({num_sites_mae_ci[1]:.2f}, {num_sites_mae_ci[1]:.2f})")
    report.append(f"is_prospective accuracy: {is_prospective_accuracy:.2f}% ({merged_df['is_prospective_match'].sum()}/{len(merged_df)})")
    
    report.append("\n" + "-" * 40)
    report.append("PERFORMANCE METRICS TABLE")
    report.append("-" * 40)
    report.append(performance_table.to_string(index=False))
    
    report.append("\n" + "-" * 40)
    report.append("PREDICATE DISAGREEMENTS")
    report.append("-" * 40)
    if 'predicate_match' in merged_df:
        if len(predicate_disagreements) == 0:
            report.append("No disagreements found!")
        else:
            report.append(f"Found {len(predicate_disagreements)} disagreements:")
            for _, row in predicate_disagreements.iterrows():
                report.append(f"  Device: {row['device_number']}")
                report.append(f"    LLM Primary: {row['primary_predicate']}")
                report.append(f"    Validation: {row['validation_predicates_clean']}")
                report.append("")

    report.append("\n" + "-" * 40)
    if has_num_sites:
        report.append("NUM_SITES DISAGREEMENTS")
        report.append("-" * 40)
        if len(num_sites_disagreements) == 0:
            report.append("No disagreements found!")
        else:
            report.append(f"Found {len(num_sites_disagreements)} disagreements:")
            for _, row in num_sites_disagreements.iterrows():
                report.append(f"  Device: {row['device_number']}")
                report.append(f"    LLM: {row['num_sites_llm']}")
                report.append(f"    Paper: {row['num_sites_paper']}")
                report.append("")
    
    report.append("\n" + "-" * 40)
    report.append("IS_PROSPECTIVE DISAGREEMENTS")
    report.append("-" * 40)
    if len(is_prospective_disagreements) == 0:
        report.append("No disagreements found!")
    else:
        report.append(f"Found {len(is_prospective_disagreements)} disagreements:")
        for _, row in is_prospective_disagreements.iterrows():
            report.append(f"  Device: {row['device_number']}")
            report.append(f"    LLM: {row['is_prospective_llm']}")
            report.append(f"    Paper: {row['is_prospective_paper']}")
            report.append("")
    
    # Devices not found in validation data
    devices_not_in_validation = set(llm_df['device_number']) - set(validation_df['approval_number'])
    if devices_not_in_validation:
        report.append("\n" + "-" * 40)
        report.append("DEVICES NOT FOUND IN VALIDATION DATA")
        report.append("-" * 40)
        report.append(f"Count: {len(devices_not_in_validation)}")
        report.append("Devices: " + ", ".join(sorted(devices_not_in_validation)))
    
    # Save report
    report_text = "\n".join(report)
    with open(output_report_file, 'w') as f:
        f.write(report_text)
    
    # print(report_text)
    print(f"Performance metrics saved to: {performance_csv_file}")
    print(f"Detailed metrics saved to: {output_csv_file}")

def calculate_performance_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance metrics table with accuracy, TPR, FPR for each field using sklearn."""
    # Define the fields to analyze
    fields = [
        ('predicate_match', 'Predicate Device', ''),
        ('num_sites_match', 'Number of Sites', 'num_sites_abs_error'), 
        ('is_multisite_match', 'Multi-site', ''),
        ('is_prospective_match', 'Prospective', '')
    ]
    
    # Calculate is_multisite_match
    merged_df['is_multisite_llm'] = merged_df['is_multisite']
    merged_df['is_multisite_paper'] = merged_df['num_sites_paper_clean'] > 1
    merged_df['is_multisite_match'] = merged_df['is_multisite_llm'] == merged_df['is_multisite_paper']
    
    results = []
    
    for field_name, field_label, field_cts_error in fields:
        if field_name not in merged_df.columns:
            continue
            
        # Calculate accuracy using sklearn
        accuracy = merged_df[field_name].mean() * 100

        # For binary fields, calculate comprehensive metrics
        if field_name in ['is_multisite_match', 'is_prospective_match']:
            # Ground truth from validation paper
            if field_name == 'is_multisite_match':
                y_true = merged_df['is_multisite_paper'].fillna(False).astype(int)
                y_pred = merged_df['is_multisite_llm'].fillna(False).astype(int)
            elif field_name == 'is_prospective_match':
                y_true = merged_df['is_prospective_paper'].fillna(False).astype(int)
                y_pred = merged_df['is_prospective_llm'].fillna(False).astype(int)

            # Use sklearn confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            tn, fp, fn, tp = cm.ravel()
            total = len(y_true)

            # Calculate metrics with proper handling of zero denominators
            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # Sensitivity/Recall
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # False Positive Rate
            accuracy_prop = (tp + tn) / total if total > 0 else 0

            # Calculate confidence intervals
            _, acc_lower, acc_upper = wald_confidence_interval(tp + tn, total) if total > 0 else (0, 0, 0)
            _, tpr_lower, tpr_upper = wald_confidence_interval(tp, tp + fn) if (tp + fn) > 0 else (0, 0, 0)
            _, spec_lower, spec_upper = wald_confidence_interval(tn, tn + fp) if (tn + fp) > 0 else (0, 0, 0)

            results.append({
                'Field': field_label,
                'Accuracy (%)': f"{accuracy:.1f}",
                'TPR': f"{tpr:.3f} ({tp}/{tp+fn})",
                '1 - FPR': f"{1 - fpr:.3f} ({tn}/{fp+tn})",
                'Accuracy CI': format_ci_string(accuracy_prop, acc_lower, acc_upper),
                'TPR CI': format_ci_string(tpr, tpr_lower, tpr_upper) if (tp + fn) > 0 else "N/A",
                '1 - FPR CI': format_ci_string(1 - fpr, spec_lower, spec_upper) if (tn + fp) > 0 else "N/A",
            })

        else:
            # For non-binary fields, TPR/FPR not applicable
            # Calculate accuracy CI for match fields
            total = len(merged_df)
            correct = merged_df[field_name].sum()
            accuracy_prop = correct / total if total > 0 else 0
            _, acc_lower, acc_upper = wald_confidence_interval(correct, total) if total > 0 else (0, 0, 0)
            mae_ci = None
            if field_cts_error:
                print(field_cts_error)
                mae_ci = wald_confidence_interval_cts(merged_df[field_cts_error])
                print(mae_ci)

            results.append({
                'Field': field_label,
                'Accuracy (%)': f"{accuracy:.1f}",
                'MAE CI': format_ci_string(*mae_ci) if field_cts_error else 'N/A',
                'TPR': "N/A",
                '1 - FPR': "N/A",
                'Accuracy CI': format_ci_string(accuracy_prop, acc_lower, acc_upper),
                'TPR CI': "N/A",
                '1 - FPR CI': "N/A",
            })
    
    return pd.DataFrame(results)

def main(args):
    """Main function to generate comparison report."""
    # Load validation data
    validation_df = load_validation_data()
    print(f"Loaded {len(validation_df)} devices from validation data")
    
    # Load LLM results
    llm_results = load_jsonl(args.input_file)
    print(f"Loaded {len(llm_results)} results from LLM processing")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison report
    generate_comparison_report(llm_results, validation_df, args.output_file, args.log_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate validation comparison report from LLM results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(OUTPUT_DIR / "llm_validation_comparison_new.jsonl"),
        help="Path to the JSONL file containing LLM results."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(OUTPUT_DIR / "validation_comparison_detailed.csv"),
        help="Path to the output csv."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(OUTPUT_DIR / "log_compare_summary.txt"),
        help="Path to the log."
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=args.log_file)
    print(args.log_file)
    main(args)