#!/usr/bin/env python3
"""
Visualization module for Adverse Events Analysis
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class VisualizationManager:
    """Handles all visualization tasks for adverse events analysis."""
    
    def __init__(self, output_fig_event_type: Path, output_fig_problem_type: Path):
        self.output_fig_event_type = output_fig_event_type
        self.output_fig_problem_type = output_fig_problem_type
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _plot_event_type_confusion_matrix(self, results: List[Dict]):
        """Plot confusion matrix between death vs injury vs malfunction codes.
        """
        EVENT_TYPE_DICT = {
            'Malfunction': 0,
            'Injury': 1,
            'Death': 2,
        }
        vendor_labels = []
        llm_labels = []

        for result in results:
            if result["has_ae"]:
                for event in result["adverse_events"]:
                    if len(event["mdr_texts"]) == 0:
                        continue

                    vendor_label = event.get("event_type", "")
                    llm_label = event.get("llm_analysis", {}).get("event_type", "")
                    # vendor_label = EVENT_TYPE_DICT[event.get("event_type", "").lower()]
                    # llm_label = EVENT_TYPE_DICT[event.get("llm_analysis", {}).get("event_type", "").lower()]
                    vendor_labels.append(vendor_label)
                    llm_labels.append(llm_label)

        df = pd.DataFrame({'Vendor': vendor_labels, 'LLM': llm_labels})
        g = sns.jointplot(data=df, x='LLM', y='Vendor', kind='hist', color='steelblue', height=7,
                         marginal_kws={'bins': 3})
        g.set_axis_labels('LLM-assigned Event Type', 'Vendor-assigned Event Type', fontsize=18)

        # Add count annotations to the center plot
        confusion_matrix = np.zeros((3, 3))
        for v, l in zip(vendor_labels, llm_labels):
            confusion_matrix[EVENT_TYPE_DICT[v], EVENT_TYPE_DICT[l]] += 1

        # Color the marginal histograms with blue palette based on height
        col_totals = confusion_matrix.sum(axis=0)
        row_totals = confusion_matrix.sum(axis=1)

        # Color top marginal bars
        for i, patch in enumerate(g.ax_marg_x.patches):
            norm_height = col_totals[i] / col_totals.max()
            patch.set_facecolor(plt.cm.Blues(0.3 + 0.7 * norm_height))

        # Color right marginal bars
        for i, patch in enumerate(g.ax_marg_y.patches):
            norm_height = row_totals[i] / row_totals.max()
            patch.set_facecolor(plt.cm.Blues(0.3 + 0.7 * norm_height))

        ax = g.ax_joint

        # Increase tick label font sizes
        ax.tick_params(axis='both', labelsize=16)

        for i in range(3):
            for j in range(3):
                count = int(confusion_matrix[i, j])
                ax.text(j, i, str(count), ha='center', va='center',
                       fontsize=18, color='white' if count > max(confusion_matrix.flatten())/2 else 'black')

        # Add column totals to top marginal histogram
        col_totals = confusion_matrix.sum(axis=0)
        for j in range(3):
            g.ax_marg_x.text(
                j,
                col_totals[j] + 40,
                # g.ax_marg_x.get_ylim()[1] * 0.35,
                f'{int(col_totals[j])}',
                ha='center',
                va='bottom',
                fontsize=18) #, color='white' if col_totals[j] > 1000 else 'black')

        # Add row totals to right marginal histogram
        row_totals = confusion_matrix.sum(axis=1)
        for i in range(3):
            g.ax_marg_y.text(
                row_totals[i] + 20,
                # g.ax_marg_y.get_xlim()[1] * 0.9,
                i,
                f'{int(row_totals[i])}',
                ha='left', va='center', fontsize=18) #, color='white' if row_totals[i] > 1000 else 'black')

        plt.tight_layout()
        plt.savefig(self.output_fig_event_type, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_product_problems_distributions(self, llm_product_problems_dict, product_problems_dict):
        plt.clf()
        product_probs_df = pd.DataFrame({
            "product_problem": [k.capitalize() for k,v in product_problems_dict.items()],
            "num": [v for k,v in product_problems_dict.items()],
            "Source": ["Manufacturer"] * len(product_problems_dict),
        }).sort_values("num", ascending=False).head(15)
        llm_product_probs_df = pd.DataFrame({
            "product_problem": [k.capitalize() for k, v in llm_product_problems_dict.items()],
            "num": [v for k,v in llm_product_problems_dict.items()],
            "Source": ["LLM"] * len(llm_product_problems_dict),
        }).sort_values("num", ascending=False).head(15)
        all_product_prob_df = pd.concat([llm_product_probs_df, product_probs_df])
        sns.set_context("notebook", font_scale=1.4)
        g = sns.catplot(
            all_product_prob_df,
            kind="bar",
            y="product_problem",
            x="num",
            col="Source",
            palette='Blues_r',
            sharey=False,
            height=9,       # each facet’s height
            aspect=1,        # width = height * aspect
        )
        for ax in g.axes.flat:
            ax.set_ylabel('')
            ax.set_xlabel('Count')

            # Wrap y-tick labels at word boundaries (every 4 words)
            labels = [label.get_text() for label in ax.get_yticklabels()]
            wrapped_labels = []
            for label in labels:
                words = label.split()
                wrapped = '\n'.join([' '.join(words[i:i+4]) for i in range(0, len(words), 4)])
                wrapped_labels.append(wrapped)
            ax.set_yticklabels(wrapped_labels)

            # Add custom titles for each subplot
            source = ax.get_title().split(' = ')[1]
            if source == 'Manufacturer':
                ax.set_title('Vendor-assigned device product problem', x=0.6, ha='right')
            elif source == 'LLM':
                ax.set_title('LLM-assigned device product problem', x=0.6, ha='right')
        
        
        # Create bar chart for subcategories
        plt.tight_layout()
        plt.savefig(self.output_fig_problem_type)
        plt.close()
    
    def _count_dpp_count_granularity(self, results: List[Dict]):
        """Counts how often the LLM proposes more codes than the vendor. Counts how often the LLM proposes a code that is more granular (deeper in the IMDRF hierarchy, so longer IMDRF code) than the vendor.

        Args:
            results (List[Dict]): List of device results with adverse events
        """
        llm_more_codes = 0
        vendor_more_codes = 0
        equal_codes = 0
        llm_more_granular = 0
        vendor_more_granular = 0
        equally_granular = 0
        total_events = 0

        for result in results:
            if result.get("has_ae"):
                for event in result["adverse_events"]:
                    vendor_codes = event.get("product_problems_imdrf", [])
                    llm_codes = event.get("llm_analysis", {}).get("fda_device_problem_codes_imdrf", [])

                    # Filter out None values
                    vendor_codes = [c for c in vendor_codes if c is not None]
                    llm_codes = [c for c in llm_codes if c is not None]

                    total_events += 1

                    # Count code quantity comparison
                    if len(llm_codes) > len(vendor_codes):
                        llm_more_codes += 1
                    elif len(vendor_codes) > len(llm_codes):
                        vendor_more_codes += 1
                    else:
                        equal_codes += 1

                    # Count granularity (depth in hierarchy = length of IMDRF code)
                    if vendor_codes and llm_codes:
                        print("VENDOR", [code.split(':')[-1] for code in vendor_codes])
                        print("LLM", [code.split(':')[-1] for code in llm_codes])
                        max_vendor_depth = max(len(code.split(':')[-1]) for code in vendor_codes)
                        max_llm_depth = max(len(code.split(':')[-1]) for code in llm_codes)

                        if max_llm_depth > max_vendor_depth:
                            llm_more_granular += 1
                        elif max_llm_depth < max_vendor_depth:
                            vendor_more_granular += 1
                        else:
                            equally_granular += 1

        logging.info(f"\nDevice Problem Code Analysis (n={total_events} events):")
        logging.info(f"LLM proposes more codes: {llm_more_codes} ({100*llm_more_codes/total_events:.1f}%)")
        logging.info(f"Vendor proposes more codes: {vendor_more_codes} ({100*vendor_more_codes/total_events:.1f}%)")
        logging.info(f"Equal number of codes: {equal_codes} ({100*equal_codes/total_events:.1f}%)")
        logging.info(f"LLM proposes more granular codes: {llm_more_granular} ({100*llm_more_granular/total_events:.1f}%)")
        logging.info(f"vendor proposes more granular codes: {vendor_more_granular} ({100*vendor_more_granular/total_events:.1f}%)")
        logging.info(f"same granular codes: {equally_granular} ({100*equally_granular/total_events:.1f}%)")

    def create_visualizations(self, results: List[Dict]):
        """Create analysis visualizations."""
        # categories = defaultdict(int)
        llm_product_problems_dict = defaultdict(int)
        product_problems_dict = defaultdict(int)
        event_types = defaultdict(int)
        llm_event_types = defaultdict(int)
        
        for result in results:
            if result["has_ae"]:
                for event in result["adverse_events"]:
                    if len(event["mdr_texts"]) == 0:
                        print("NO TEXT!")
                        continue
                    llm_product_prob_list = event.get("llm_analysis", {}).get("fda_device_problem_codes", [])
                    llm_event = event.get("llm_analysis", {}).get("event_type", None)
                    product_problems = event["product_problems"]
                    for subcategory in llm_product_prob_list:
                        llm_product_problems_dict[subcategory] += 1
                    for prob in product_problems:
                        product_problems_dict[prob] += 1
                    event_type = event.get("event_type", None)
                    event_types[event_type.lower()] += 1
                    if llm_event is None:
                        print(f"NONE! {event_type}")
                        continue
                    llm_event_types[llm_event.lower()] += 1
        
        # Event type distribution
        self._plot_event_type_confusion_matrix(results)
        
        # Product problems distribution
        self._plot_product_problems_distributions(llm_product_problems_dict, product_problems_dict)

        self._count_dpp_count_granularity(results)

def map_to_imdrf_code(results: List[Dict], device_problems_df):
    """Inserts the IMDRF code as another entry in each MDR dict

    Args:
        results (List[Dict]): List of device results with adverse events
        device_problems_df (pd.DataFrame): DataFrame with CDRH Preferred Term to IMDRF Code mappings

    Returns:
        List[Dict]: Updated results with IMDRF codes added
    """
    # Create mapping from CDRH Preferred Term to IMDRF Code
    term_to_imdrf = dict(zip(device_problems_df['CDRH Preferred Term'], device_problems_df['IMDRF Code']))

    for result in results:
        if result.get("has_ae"):
            for event in result["adverse_events"]:
                # Map vendor-assigned product problems
                if "product_problems" in event and event["product_problems"]:
                    event["product_problems_imdrf"] = [
                        term_to_imdrf.get(term, None) for term in event["product_problems"]
                    ]

                # Map LLM-assigned device problem codes
                if "llm_analysis" in event and "fda_device_problem_codes" in event["llm_analysis"]:
                    event["llm_analysis"]["fda_device_problem_codes_imdrf"] = [
                        term_to_imdrf.get(term, None) for term in event["llm_analysis"]["fda_device_problem_codes"]
                    ]

    return results


def main():
    """Main function to run visualizations on existing analysis results."""
    parser = argparse.ArgumentParser(description="Adverse Events Analysis Visualization Script")
    parser.add_argument("--results-file",
                       help="Path to analysis results JSON file")
    parser.add_argument("--judge-results-files",
                        nargs="*",
                       help="Path to judge validation results JSON file (optional)")
    parser.add_argument("--output-fig-event-type")
    parser.add_argument("--output-fig-problem-type")
    parser.add_argument("--log-file")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s \n%(message)s', filename= args.log_file)
    logging.info(args)

    device_problems_df = pd.read_csv(PROJECT_ROOT / "data/FDA-CDRH_NCIt_Subsets.csv")
    device_problems_df = device_problems_df[device_problems_df['CDRH Subset Name'] == "Medical Device Problem"]
    
    # Load results from analysis
    with open(args.results_file, 'r') as f:
        results = json.load(f)
        results = map_to_imdrf_code(results, device_problems_df)
    
    # Load + process judge results
    all_judge_results = []
    for judge_file in args.judge_results_files:
        with open(judge_file, 'r') as f:
            judge_res_dict = json.load(f)
            all_judge_results.append(judge_res_dict)
    
    all_judge_dfs = []
    for judge_results, judge_file in zip(all_judge_results, args.judge_results_files):
        judge = "gpt" if "gpt" in judge_file else "claude"
        judge_result_product_prob_dict = {}
        judge_result_product_prob_dict["task"] = "product problem"
        judge_result_product_prob_dict["judge"] = judge
        judge_result_event_type_dict = {}
        judge_result_event_type_dict["task"] = "event type"
        judge_result_event_type_dict["judge"] = judge
        num_judged = 0
        for res in judge_results:
            for ae in res['adverse_events']:
                if 'validation' not in ae or 'device_problem_codes' not in ae['validation']:
                    print("judge did not run successfully")
                elif len(ae['mdr_texts']) == 0:
                    print("mdr text was empty")
                else:
                    product_prob_winner = ae['validation']["device_problem_codes"]['judge_result_converted']
                    event_winner = ae['validation']["event_type"]['judge_result_converted']
                    judge_result_product_prob_dict[product_prob_winner] = judge_result_product_prob_dict.get(product_prob_winner, 0) + 1
                    judge_result_event_type_dict[event_winner] = judge_result_event_type_dict.get(event_winner, 0) + 1
                    num_judged += 1
                    
        all_judge_dfs += [judge_result_product_prob_dict, judge_result_event_type_dict]
        logging.info("JUDGE RESULTS: device product problem %s", judge_result_product_prob_dict)
        logging.info("JUDGE RESULTS: event type %s", judge_result_event_type_dict)
        print(f"JUDGE {judge} NUM JUDGED {num_judged}")
    all_judge_df = pd.DataFrame(all_judge_dfs).sort_values('task')
    all_judge_percent_df = all_judge_df.copy()
    all_judge_percent_df['non_ties'] = all_judge_percent_df.llm + all_judge_percent_df.vendor
    all_judge_percent_df['llm'] = all_judge_percent_df.llm/all_judge_percent_df.non_ties
    all_judge_percent_df['vendor'] = all_judge_percent_df.vendor/all_judge_percent_df.non_ties
    logging.info("COUNTS")
    logging.info(all_judge_df)
    logging.info("PERCENT")
    logging.info(all_judge_percent_df.round(2))
    print(all_judge_df)
    print(all_judge_percent_df.round(2))
    
    # Create visualizations
    viz_manager = VisualizationManager(args.output_fig_event_type, args.output_fig_problem_type)
    viz_manager.create_visualizations(results)
    
    print("LOG FILE", args.log_file)
    print("png event type", args.output_fig_event_type)
    print("png problem type", args.output_fig_problem_type)


if __name__ == "__main__":
    main()