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


def load_llm_extractions(llm_extract_path):
    """Load LLM extractions from JSON file"""
    with open(llm_extract_path, 'r') as f:
        data = json.load(f)

    # Create mapping from report_number to llm_analysis
    extraction_dict = {}
    for device in data:
        if device.get('has_ae'):
            for event in device.get('adverse_events', []):
                report_num = event.get('report_number')
                if report_num and 'llm_analysis' in event:
                    extraction_dict[report_num] = event['llm_analysis']

    return extraction_dict


def load_annotations(annotation_files):
    """Load human annotations from CSV file"""
    all_dfs = []
    for annot_f in annotation_files:
        df = pd.read_csv(annot_f)
        # df = df.drop_duplicates(subset='report number', keep='last')
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)
    print("final", all_dfs.shape)
    return all_dfs.reset_index()


def load_random_orderings(random_orderings_files):
    """Load random orderings that map which source (vendor vs human) for each column"""
    all_dfs = []
    for rand_file in random_orderings_files:
        df = pd.read_csv(rand_file)[['report number', 'event_1', 'event_2', 'problem_1', 'problem_2', 'event_1_true', 'event_2_true', 'problem_1_true', 'problem_2_true']]
        all_dfs.append(df)
    return pd.concat(all_dfs).reset_index()


def calculate_inter_annotator_agreement(annotations_df, random_orderings_df, fields=['event_type', 'device_problem_codes']):
    """
    Calculate inter-annotator agreement for reports with multiple annotations
    Returns agreement rate
    """
    # Group by report number to find reports with multiple annotations
    assert np.all(annotations_df['report number'] == random_orderings_df['report number'])
    assert np.all(annotations_df['event_1'] == random_orderings_df['event_1'])
    assert np.all(annotations_df['event_2'] == random_orderings_df['event_2'])
    assert np.all(annotations_df['problem_1'] == random_orderings_df['problem_1'])
    assert np.all(annotations_df['problem_2'] == random_orderings_df['problem_2'])
    merged_df = pd.concat([annotations_df, random_orderings_df.drop('report number', axis=1)], axis=1)
    report_groups = merged_df.groupby('report number')

    results = []
    report_numbers = []
    all_annotations = []
    for field in fields:
        # Collect all pairwise agreements
        pairwise_agrees = []
        for report_num, group in report_groups:
            if group.shape[0] == 1:
                continue
            if np.all(group[field] == 'Same'):
                continue
            annotations = np.array([])
            for i in range(group.shape[0]):
                annotations = np.append(
                    annotations,
                    "Same" if group[field].iloc[i] == "Same" else (
                        group['event_1_true'].iloc[i] if group[field].iloc[i] == "Option 1 better" else group['event_2_true'].iloc[i]
                    )
                )
            
            agreement = np.all(annotations == annotations[0])
            all_annotations.append(annotations)
            report_numbers.append(report_num)
            # if np.any(group[field] == 'Same'):
            #     print(report_num, field)
            #     # print(group[field])
            #     # print(group['event_1_true'])
            #     # print(group['event_2_true'])
            #     print(annotations)
            # if not agreement:
            #     print(report_num, field)
            #     print(group[field])
            #     print(group['event_1_true'])
            #     print(group['event_2_true'])
            #     print(annotations)
            
            pairwise_agrees.append(agreement)
            
        mean_agree = np.mean(pairwise_agrees)
        n_pairs = len(pairwise_agrees)

        # Calculate 95% CI using normal approximation
        se = np.std(pairwise_agrees, ddof=1) / np.sqrt(n_pairs)
        ci_lower = mean_agree - 1.96 * se
        ci_upper = mean_agree + 1.96 * se

        results.append({
            'field': field,
            'metric': 'Percent Agreement',
            'value': mean_agree,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_comparisons': n_pairs,
            'n_reports': len(report_groups)
        })

    annotation_df = pd.DataFrame(all_annotations)
    annotation_df["report_number"] = report_numbers
    return pd.DataFrame(results), annotation_df


def evaluate_preferences(annotations_df, random_orderings_df):
    """Evaluate how often human annotator preferred LLM labels vs vendor labels

    Each report number's votes are weighted by 1/num_votes to ensure equal influence
    """

    # Merge annotations with random orderings
    assert np.all(annotations_df['report number'] == random_orderings_df['report number'])
    merged_df = pd.concat([annotations_df, random_orderings_df.drop('report number', axis=1)], axis=1)

    # Count preferences with weighting by report number
    preferred_dict = {
        'event_type': {
            'llm': 0.0,
            'vendor': 0.0,
            'tie': 0.0,
        },
        'device_problem_codes': {
            'llm': 0.0,
            'vendor': 0.0,
            'tie': 0.0,
        },
    }
    dict_keys = list(preferred_dict.keys())

    # Group by report number to apply weighting
    report_groups = merged_df.groupby('report number')

    for report_num, group in report_groups:
        weight = 1.0 / group.shape[0]
        
        for index, row in group.iterrows():
            for field in dict_keys:
                if row[field] == 'Same':
                    preferred_dict[field]['tie'] += weight
                elif row[field] == 'Option 1 better':
                    option1 = row['event_1_true']
                    preferred_dict[field][option1] += weight
                    # if option1 == "vendor":
                    #     print(report_num)
                else:
                    option2 = row['event_2_true']
                    preferred_dict[field][option2] += weight
                    # if option2 == "vendor":
                    #     print(report_num)
                
    results = pd.DataFrame({
        'field': dict_keys,
        'llm': [preferred_dict[k]['llm'] for k in dict_keys],
        'vendor': [preferred_dict[k]['vendor'] for k in dict_keys],
        'tie': [preferred_dict[k]['tie'] for k in dict_keys],
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Summarize MDR annotation results comparing LLM vs vendor labels")
    # parser.add_argument("--llm-extract", required=True,
    #                    help="Path to LLM extraction JSON file")
    parser.add_argument("--annotations", nargs="*", required=True,
                       help="Path to human annotation CSV file")
    parser.add_argument("--random-orderings", nargs="*", required=True,
                       help="Path to random orderings CSV file")
    parser.add_argument("--results", required=True,
                       help="Path to save results CSV file")
    parser.add_argument("--iaa-results", required=False, default=None,
                       help="Path to save inter-annotator agreement results CSV file")

    args = parser.parse_args()

    # Load data
    annotations_df = load_annotations(args.annotations)
    print(f"Unique MDRs {annotations_df['report number'].unique().size}")
    random_orderings_df = load_random_orderings(args.random_orderings)

    print(f"Loaded {len(annotations_df)} human annotations")
    print(f"Loaded {len(random_orderings_df)} random orderings")

    # Calculate inter-annotator agreement
    annotations23_df = load_annotations(args.annotations[2:])
    random_orderings23_df = load_random_orderings(args.random_orderings[2:])
    iaa_results, _ = calculate_inter_annotator_agreement(annotations23_df, random_orderings23_df)

    print("\n" + "="*50)
    print("INTER-ANNOTATOR AGREEMENT")
    print("="*50)
    for _, row in iaa_results.iterrows():
        print(f"\n{row['field']}:")
        print(f"  {row['metric']}: {row['value']:.3f} (95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}])")
        print(f"  N comparisons: {row['n_comparisons']}")
        print(f"  N reports with multiple annotations: {row['n_reports']}")

    # DEATH inter-annotator agreements
    death_mask = (annotations_df.event_1 == "Death") | (annotations_df.event_2 == "Death")
    death_annotations_df = annotations_df[death_mask]
    death_random_orderings_df = random_orderings_df[death_mask]
    _, death_iaa_results = calculate_inter_annotator_agreement(death_annotations_df, death_random_orderings_df, fields=["event_type"])
    print("---------DEATH------------")
    print(death_iaa_results)

    
    # Evaluate preferences
    results_df = evaluate_preferences(annotations_df, random_orderings_df)

    print("\n" + "="*50)
    print("ANNOTATION VALIDATION RESULTS")
    print("="*50)
    print(results_df.to_string(index=False))

    results_df['non_tie'] = results_df['llm'] + results_df['vendor']
    results_df['llm'] = results_df['llm']/results_df['non_tie']
    results_df['vendor'] = results_df['vendor']/results_df['non_tie']
    print("PERCENTAGES")
    print(results_df)

    # Save results
    results_df.to_csv(args.results, index=False)
    print(f"Results saved to {args.results}")

    if args.iaa_results:
        iaa_results.to_csv(args.iaa_results, index=False)
        print(f"Inter-annotator agreement results saved to {args.iaa_results}")


if __name__ == "__main__":
    main()
