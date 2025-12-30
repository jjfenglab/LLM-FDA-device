#!/usr/bin/env python3
"""
Script to analyze clinical study trends over time for AI/ML devices.
Creates plots showing how is_prospective and is_multisite change over time.
"""

import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

from summarize_validation_comparison import load_jsonl

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FILE = PROJECT_ROOT / "data/raw/validation/zou_clinical_data.csv"

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")


def extract_year_from_device_number(device_number: str) -> Optional[int]:
    """
    Extract year from device number using the first two digits.
    Examples: K210822 -> 2021, DEN090011 -> 2009, P140011 -> 2014
    """
    # Remove prefix letters and get first two digits
    digits = re.findall(r'\d+', device_number)
    if not digits:
        return None
    
    first_digits = digits[0][:2]
    if len(first_digits) < 2:
        return None
    
    try:
        year_suffix = int(first_digits)
        
        # Convert 2-digit year to 4-digit year
        # Assume years 90-99 are 1990s, 00-89 are 2000s+
        if year_suffix >= 90:
            return 1900 + year_suffix
        else:
            return 2000 + year_suffix
    except ValueError:
        return None

def get_device_year(device_number: str, metadata_lookup: Dict[str, Dict]) -> Optional[int]:
    """
    Get the year for a device, first trying metadata, then fallback to device number parsing.
    """
    # Try to get year from metadata
    if device_number in metadata_lookup:
        metadata = metadata_lookup[device_number].get('metadata', {})
        date_received = metadata.get('date_received')
        
        if date_received:
            try:
                # Parse various date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y', '%d/%m/%Y']:
                    try:
                        parsed_date = datetime.strptime(date_received, fmt)
                        return parsed_date.year
                    except ValueError:
                        continue
            except Exception:
                pass
    
    # Fallback to parsing device number
    return extract_year_from_device_number(device_number)

def load_and_process_llm_results(input_file, metadata_lookup: Dict[str, Dict]) -> pd.DataFrame:
    """Load and process LLM results from JSONL file."""
    print("Loading LLM results...")
    llm_data = load_jsonl(input_file)
    
    if not llm_data:
        print("No LLM data found!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(llm_data)
    
    # Add year information
    df['year'] = df['device_number'].apply(lambda x: get_device_year(x, metadata_lookup))
    
    # Filter out devices without year information
    initial_count = len(df)
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    print(f"Processed {len(df)} LLM results (dropped {initial_count - len(df)} without year info)")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    return df

def load_and_process_validation_data() -> pd.DataFrame:
    """Load and process validation data from CSV file."""
    print("Loading validation data...")
    
    if not VALIDATION_FILE.exists():
        print(f"Validation file not found: {VALIDATION_FILE}")
        return pd.DataFrame()
    
    # Load validation data
    val_df = pd.read_csv(VALIDATION_FILE)
    
    # Ensure required columns exist
    required_cols = ['approval_number', 'is_prospective', 'num_sites']
    missing_cols = [col for col in required_cols if col not in val_df.columns]
    if missing_cols:
        print(f"Missing columns in validation data: {missing_cols}")
        return pd.DataFrame()
    
    # Rename approval_number to device_number for consistency
    val_df = val_df.rename(columns={'approval_number': 'device_number'})
    
    # Extract year from device numbers (validation data doesn't have metadata)
    val_df['year'] = val_df['device_number'].apply(extract_year_from_device_number)
    
    # Filter out devices without year information
    initial_count = len(val_df)
    val_df = val_df.dropna(subset=['year'])
    val_df['year'] = val_df['year'].astype(int)
    
    print(f"Processed {len(val_df)} validation results (dropped {initial_count - len(val_df)} without year info)")
    if len(val_df) > 0:
        print(f"Year range: {val_df['year'].min()} - {val_df['year'].max()}")
    
    return val_df

def create_yearly_summary(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Create yearly summary statistics with grouping for years <=2015."""
    if df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying original data
    df_copy = df.copy()
    
    # Group years <= 2015 into one category
    df_copy['year_grouped'] = df_copy['year'].apply(lambda x: '≤2015' if x <= 2015 else str(x))
    
    yearly_stats = df_copy.groupby('year_grouped').agg({
        'is_prospective': ['count', 'sum'],
        'is_multisite': ['count', 'sum']
    }).round(3)
    
    # Flatten column names
    yearly_stats.columns = [f"{col[1]}_{col[0]}" for col in yearly_stats.columns]
    
    # Calculate percentages
    yearly_stats['prospective_rate'] = (yearly_stats['sum_is_prospective'] / yearly_stats['count_is_prospective'] * 100).round(1)
    yearly_stats['multi_sites_rate'] = (yearly_stats['sum_is_multisite'] / yearly_stats['count_is_multisite'] * 100).round(1)
    
    # Calculate not prospective and not multi-sites percentages
    yearly_stats['not_prospective_rate'] = 100 - yearly_stats['prospective_rate']
    yearly_stats['not_multi_sites_rate'] = 100 - yearly_stats['multi_sites_rate']
    
    # Add total counts
    yearly_stats['total_devices'] = yearly_stats['count_is_prospective']
    
    # Add source identifier
    yearly_stats['source'] = source_name
    
    return yearly_stats.reset_index().rename(columns={'year_grouped': 'year'})

def prepare_x_axis_data(summary_df: pd.DataFrame) -> Tuple[List, List]:
    """Prepare x-axis data for plotting, handling mixed string/numeric year labels."""
    if summary_df.empty:
        return [], []
    
    # Sort data: ≤2015 first, then numeric years
    sorted_df = summary_df.copy()
    sorted_df['sort_key'] = sorted_df['year'].apply(lambda x: 0 if x == '≤2015' else int(x))
    sorted_df = sorted_df.sort_values('sort_key')
    
    x_labels = sorted_df['year'].tolist()
    x_positions = list(range(len(x_labels)))
    
    return x_positions, x_labels

def plot_2_llm_with_negatives(llm_summary: pd.DataFrame, plot_file: str):
    """Plot 2: LLM results with positive and negative percentages."""
    if llm_summary.empty:
        print("No LLM data for Plot 2")
        return
    
    # Prepare data for plotting
    sorted_data = llm_summary.sort_values('sort_key')
    
    # Create long-form DataFrame for seaborn
    plot_data = []
    for _, row in sorted_data.iterrows():
        plot_data.extend([
            {'Year': row['year'], 'Percentage': row['prospective_rate'], 'Category': 'Prospective'},
            {'Year': row['year'], 'Percentage': row['multi_sites_rate'], 'Category': 'Multi-site'},
            # {'Year': row['year'], 'Percentage': row['not_prospective_rate'], 'Category': 'Not Prospective', 'Type': 'Negative'},
            # {'Year': row['year'], 'Percentage': row['not_multi_sites_rate'], 'Category': 'Not Multi-sites', 'Type': 'Negative'}
        ])
    
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(7, 5))
    
    # Plot with seaborn
    sns.lineplot(data=plot_df, x='Year', y='Percentage', hue='Category', style="Category", 
                 markers=True, linewidth=3)
    
    plt.xlabel('Year of FDA submission', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10, loc='center left')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    sns.despine()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot 2 saved to: {plot_file}")
    plt.close()

def plot_trends(llm_summary: pd.DataFrame, plot_file: str):
    """Create all five trend plots."""
    # Add sort key to summaries for consistent ordering
    if not llm_summary.empty:
        llm_summary['sort_key'] = llm_summary['year'].apply(lambda x: 0 if x == '≤2015' else int(x))
    
    print("Creating Plot 2: LLM with negatives (percentages)...")
    plot_2_llm_with_negatives(llm_summary, plot_file)

def print_device_counts_by_year(llm_df: pd.DataFrame):
    """Print number of devices received each year."""
    if llm_df.empty:
        return
    
    print("\n" + "="*60)
    print("DEVICE COUNTS BY YEAR")
    print("="*60)
    
    # Group by year with same logic as plotting
    df_copy = llm_df.copy()
    df_copy['year_grouped'] = df_copy['year'].apply(lambda x: '≤2015' if x <= 2015 else str(x))
    
    yearly_counts = df_copy.groupby('year_grouped').size().reset_index(name='device_count')
    yearly_counts['sort_key'] = yearly_counts['year_grouped'].apply(lambda x: 0 if x == '≤2015' else int(x))
    yearly_counts = yearly_counts.sort_values('sort_key')
    
    for _, row in yearly_counts.iterrows():
        print(f"  {row['year_grouped']}: {row['device_count']} devices")
    
    print(f"\nTotal devices: {len(llm_df)}")

def print_summary_statistics(llm_df: pd.DataFrame, val_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if not llm_df.empty:
        print(f"\nLLM Results ({len(llm_df)} devices):")
        print(f"  Years covered: {llm_df['year'].min()} - {llm_df['year'].max()}")
        print(f"  Prospective studies: {llm_df['is_prospective'].sum()} / {len(llm_df)} ({llm_df['is_prospective'].mean()*100:.1f}%)")
        print(f"  Multi-site studies: {llm_df['is_multisite'].sum()} / {len(llm_df)} ({llm_df['is_multisite'].mean()*100:.1f}%)")
    
    if not val_df.empty:
        print(f"\nValidation Data ({len(val_df)} devices):")
        print(f"  Years covered: {val_df['year'].min()} - {val_df['year'].max()}")
        print(f"  Prospective studies: {val_df['is_prospective'].sum()} / {len(val_df)} ({val_df['is_prospective'].mean()*100:.1f}%)")
        print(f"  Multi-site studies: {val_df['is_multisite'].sum()} / {len(val_df)} ({val_df['is_multisite'].mean()*100:.1f}%)")

def main(args):
    """Main function to run the analysis."""
    print("Starting clinical study trends analysis...")
    
    # Load metadata for year lookup
    print("Loading metadata...")
    metadata_data = load_jsonl(args.metadata_file)
    metadata_lookup = {item['device_number']: item for item in metadata_data if 'device_number' in item}
    print(f"Loaded metadata for {len(metadata_lookup)} devices")
    
    # Load and process data
    llm_df = load_and_process_llm_results(args.input_file, metadata_lookup)
    
    if llm_df.empty and val_df.empty:
        print("No data to analyze!")
        return
    
    # Create yearly summaries
    llm_summary = create_yearly_summary(llm_df, "LLM") if not llm_df.empty else pd.DataFrame()
    
    # Print device counts by year
    print_device_counts_by_year(llm_df)
    
    # Create plots
    print("\nCreating trend plots...")
    plot_trends(llm_summary, args.output_file)
    
    print(f"\nAnalysis completed! Results saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare LLM extraction results with previous paper's validation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(OUTPUT_DIR / "aiml_devices_validation_results.jsonl"),
        help="Path to the input JSONL file with pre-extracted devices."
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=str(PROJECT_ROOT / "scripts/analysis_pre_post_associations/output/aiml_device_results_with_metadata.jsonl"),
        help="Path to the metadata JSONL file with device metadata."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(OUTPUT_DIR / "validation_trends.png"),
    )
    args = parser.parse_args()
    main(args)
