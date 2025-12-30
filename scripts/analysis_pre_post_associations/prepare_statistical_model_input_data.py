import json
import argparse
import pandas as pd
from pathlib import Path
import logging
import re
from datetime import datetime, timedelta
from collections import Counter

"""
Script to prepare survival analysis data from merged pre-post market data.
Transforms merged_pre_post_market_data.json into PWP gap-time Cox model input format.
Adapted for simplified predicate features (predicate_has_ae, predicate_has_recall).
"""

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define feature lists for the new simplified format
DEVICE_FEATURE_KEYS = [
    "intended_use_and_clinical_applications",
    "operational_and_workflow_change", 
    "algorithm_or_software_feature_changes",
    "hardware_changes",
    "body_part_changes",
    "human_device_team_testing",
    "has_clinical_testing"
]

# Simplified predicate features (no detailed AE breakdown)
PREDICATE_FEATURE_KEYS = [
    "predicate_has_ae",
    "predicate_has_recall"
]

# Logging will be set up in main() with args

def load_data(input_path):
    """Load the merged pre-post market data"""
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} records from {input_path}")
        return data
    except FileNotFoundError:
        logging.error(f"Input data file not found: {input_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {input_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred loading {input_path}: {e}")
        return None

def get_unique_ae_categories(data):
    """Extract all unique adverse event categories from the data"""
    categories = set()
    category_counts = Counter()
    
    for record in data:
        adverse_events = record.get('adverse_events', {}).get('data', [])
        for ae in adverse_events:
            llm_analysis = ae.get('llm_analysis', {})
            dev_categories = llm_analysis.get('main_category')
            for category in dev_categories:
                categories.add(category)
                category_counts[category] += 1
    
    # Log the categories and their counts
    logging.info("Found the following adverse event categories:")
    for category, count in category_counts.most_common():
        logging.info(f"  - {category}: {count} events")
    
    return sorted(list(categories))

def extract_decision_date_from_device_number(device_number):
    """
    Extract decision date from device number pattern.
    For K-numbers like K240369, extract year (24) and approximate date.
    This is a fallback when decision_date is not available.
    """
    if not device_number or not device_number.startswith('K'):
        return None
    
    try:
        # Extract year from device number (e.g., K240369 -> 24 -> 2024)
        year_part = device_number[1:3]
        if year_part.isdigit():
            year = int(year_part)
            # Convert 2-digit year to 4-digit (assuming 20xx for recent devices)
            if year >= 0 and year <= 50:  # Assuming devices from 2000-2050
                full_year = 2000 + year
            else:
                full_year = 1900 + year
            
            # Use January 1st as default date
            return f"{full_year}0101"
    except (ValueError, IndexError):
        pass
    
    return None

def create_feature_matrix(data, ae_category=None):
    """
    Transform the data into a multi-hot encoding matrix including validation and predicate features.
    Args:
        data: The input data
        ae_category: Optional category to filter adverse events by their main_category
    Returns a tuple: (DataFrame, list_of_change_types)
    """
    # Use the simplified feature keys
    all_feature_names = DEVICE_FEATURE_KEYS + PREDICATE_FEATURE_KEYS
    
    logging.info(f"Using {len(all_feature_names)} features: {all_feature_names}")

    # Create feature matrix
    all_features = []
    print(f'Total number of records: {len(data)}')
    data_with_decision_date = [record for record in data if record.get('decision_date')]
    data_without_decision_date = [record for record in data if not record.get('decision_date')]
    print(f'Number of records with decision date: {len(data_with_decision_date)}')
    print(f'Number of records without decision date: {len(data_without_decision_date)}')
    
    for i, record in enumerate(data):
        try:
            # Log device being processed
            device_num = record.get('device_number', 'UNKNOWN')
            predicate_num = record.get('primary_predicate', 'UNKNOWN')
            
            logging.debug(f"Processing device {device_num} ({i+1}/{len(data)})")
            
            # Create feature vector - directly from top-level fields
            feature_dict = {}
            for feature_name in all_feature_names:
                feature_dict[feature_name] = record.get(feature_name, 0)
                
            feature_df = pd.Series(feature_dict)

            feature_df['device_num'] = device_num
            feature_df['predicate_num'] = predicate_num
            feature_df['primary_product_code'] = record.get('primary_product_code', '')
            feature_df['panel_lead'] = record.get('panel_lead', '')
            feature_df['clearance_type'] = record.get('clearance_type', '')
            feature_df['is_ai_ml'] = record.get('is_ai_ml', 1)
            
            # Try to get decision date, fallback to extracting from device number
            decision_date_str = record.get('decision_date', '').replace("-", "")
            if not decision_date_str:
                decision_date_str = extract_decision_date_from_device_number(device_num)
                if decision_date_str:
                    logging.warning(f"Device {device_num}, predicate {predicate_num}: Using extracted decision date {decision_date_str}")
                else:
                    logging.warning(f"Device {device_num}: No decision date available and couldn't extract from device number. Skipping.")
                    continue
            decision_date = datetime.strptime(decision_date_str, "%Y%m%d")
            
            device_features = []

            # Filter and sort adverse events with valid dates
            ae_records = []
            adverse_events_data = record.get('adverse_events', {}).get('data', [])
            
            # Filter by category if specified
            if ae_category:
                adverse_events_data = [
                    ae for ae in adverse_events_data 
                    if ae_category.lower() in ae.get('llm_analysis', {}).get('main_category', [])
                ]
            
            for ae_record in adverse_events_data:
                # Try date_of_event first, fallback to date_report
                event_date = ae_record.get('date_of_event')
                if event_date is None:
                    event_date = ae_record.get('date_report')
                if event_date is not None:
                    # Ensure date format consistency (remove hyphens)
                    event_date_formatted = str(event_date).replace("-", "")
                    ae_records.append({**ae_record, 'event_date': event_date_formatted})
            
            # Sort by event date
            ae_records.sort(key=lambda x: x['event_date'])
            
            # Process each adverse event
            current_t_start = decision_date_str
            ae_num = 0
            for ae_dict in ae_records:
                event_date = datetime.strptime(ae_dict["event_date"], "%Y%m%d")
                if event_date <= decision_date:
                    continue
                feature_df["ae_num"] = ae_num
                feature_df["event_type"] = ae_dict.get("event_type", "").lower()
                feature_df["t_start"] = current_t_start
                
                if ae_dict['event_date'] == current_t_start:
                    continue
                else:
                    feature_df["t_end"] = ae_dict['event_date']
                    feature_df["event"] = 1
                    device_features.append(feature_df.copy())
                    current_t_start = ae_dict['event_date']
                    ae_num += 1
            
            # Add final record (censoring interval)
            last_date = (datetime.now()).strftime("%Y%m%d")

            if last_date < current_t_start:
                logging.warning(f"Device {device_num}: Calculated last_date ({last_date}) is before current_t_start ({current_t_start}). Using current_t_start as end for censoring interval.")
                last_date = current_t_start

            if not ae_records:
                feature_df["ae_num"] = len(ae_records)
                feature_df["t_start"] = current_t_start
                feature_df["t_end"] = last_date
                feature_df["event"] = 0
                device_features.append(feature_df.copy())
            elif last_date == current_t_start and ae_records:
                logging.debug(f"Device {device_num}: Last event date ({current_t_start}) matches last_date ({last_date}). No censoring interval added.")

            if device_features:
                device_features_df = pd.DataFrame(device_features)
                all_features.append(device_features_df)
            else:
                logging.debug(f"Device {device_num}: No adverse event intervals generated.")
            
        except Exception as e:
            logging.error(f"Error processing record {i} (Device: {record.get('device_number', 'UNKNOWN')}):")
            logging.exception("Detailed error:")
            raise
    
    # Concatenate all features, handling potential empty list
    if not all_features:
        logging.warning(f"No features generated for any device{' for category: ' + ae_category if ae_category else ''}. Returning empty DataFrame and feature types.")
        all_cols = all_feature_names + ['device_num', 'predicate_num', 'is_ai_ml', 'ae_num', 't_start', 't_end', 'event']
        # Return empty DataFrame and the determined feature types
        return pd.DataFrame(columns=all_cols), all_feature_names

    # Return the DataFrame and the list of feature types
    return pd.concat(all_features).reset_index(drop=True), all_feature_names

def process_and_save_features(features_df, feature_names, output_path):
    """Helper function to process and save features DataFrame"""
    # Convert boolean-like columns (0/1) to integers
    bool_cols = feature_names + ['event', 'is_ai_ml']
    for col in bool_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(0).astype(int)
    
    # Convert time columns to appropriate type
    for col in ['t_start', 't_end']:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

    # Drop rows with NaNs in critical columns
    initial_rows = len(features_df)
    features_df.dropna(subset=['t_start', 't_end'], inplace=True)
    if len(features_df) < initial_rows:
        logging.warning(f"Dropped {initial_rows - len(features_df)} rows due to missing/invalid t_start or t_end after conversion.")

    features_df.to_csv(output_path, index=False)
    logging.info(f"Saved to {output_path}")
    logging.info(f"Number of rows: {features_df.shape[0]}, Number of events: {features_df.event.sum()}")

def main(args):
    # Load data
    data = load_data(args.input_file)
    if data is None:
        logging.error("Failed to load base data. Exiting.")
        return

    # Get unique adverse event categories
    ae_categories = get_unique_ae_categories(data)
    if not ae_categories:
        logging.warning("No adverse event categories found in the data.")
        # Still proceed to generate overall data without AE filtering
        ae_categories = []

    # First generate the overall CSV with all adverse events
    features_df, feature_names = create_feature_matrix(data)
    print("NUM EVENTS", features_df.event.sum())
    process_and_save_features(features_df, feature_names, args.output_csv)
    print(f"All adverse events data saved to: {args.output_csv}")

    for category in ae_categories:
        features_df, feature_names = create_feature_matrix(data, category)
        # Create a safe filename by replacing spaces and special characters
        safe_category = re.sub(r'[^a-zA-Z0-9]+', '_', category.lower()).strip('_')
        output_path = args.output_csv.replace(".csv", f'{safe_category}.csv')
        process_and_save_features(features_df, feature_names, output_path)
        print(f"{category} data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare survival analysis data from merged pre-post market data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(OUTPUT_DIR / "merged_pre_post_market_data_mapped.json"),
        help="Path to the merged pre-post market data JSON file."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(OUTPUT_DIR / "ae_survival_data_mapped.csv"),
        help="Path to the main output survival analysis CSV file."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=str(OUTPUT_DIR / "log_prepare_stats.txt"),
        help="Path to the log file."
    )

    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        filename=args.log_file,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main(args)