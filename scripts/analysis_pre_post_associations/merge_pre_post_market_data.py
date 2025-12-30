import json
import argparse
from pathlib import Path
import logging

import pandas as pd

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature Mappings ---
# Direct mapping for device features - these fields already exist with the desired names
DEVICE_FEATURE_KEYS = [
    "intended_use_and_clinical_applications",
    "operational_and_workflow_change", 
    "algorithm_or_software_feature_changes",
    "hardware_changes",
    "body_part_changes",
    "human_device_team_testing",
    "has_clinical_testing"
]


# --- Helper Functions ---
def load_json(file_path):
    """Loads data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    logging.info(f"Successfully loaded {len(data)} records from {file_path}")
    return data

def load_jsonl(file_path):
    """Loads data from a JSON Lines file."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line {i+1} in {file_path}")
                continue
    logging.info(f"Successfully loaded {len(data)} records from {file_path}")
    return data

def process_device_features(record):
    """Processes device features from the new format - direct boolean values."""
    features = {}
    
    for feature_key in DEVICE_FEATURE_KEYS:
        # Convert boolean to integer (1 for True, 0 for False)
        value = record.get(feature_key, False)
        features[feature_key] = 1 if value else 0
    
    return features

def process_adverse_events_data(ae_data, device_problems_df):
    """Processes adverse events data into dicts keyed by device_number."""
    processed_data = {}
    
    if not ae_data:
        return processed_data
    
    # Tracking numbers
    tot_mdrs = 0
    tot_nonempty_mdrs = 0
    tot_510k_mdrs = 0
    tot_510k_nonempty_mdrs = 0
    tot_510k_predicate_mdrs = 0
    for device_data in ae_data:
        device_num = device_data.get('device_number')
        
        if not device_num:
            continue
            
        # Process adverse events for this device
        has_ae = device_data.get('has_ae', False)
        adverse_events = device_data.get('adverse_events', [])
        for ae in adverse_events:
            ae['llm_analysis']['main_category'] = []
            for fda_device_problem_code in ae['llm_analysis']['fda_device_problem_codes']:
                category = device_problems_df.Category[device_problems_df['CDRH Preferred Term'] == fda_device_problem_code]
                if category.size and category.iloc[0] not in ae['llm_analysis']['main_category']:
                    ae['llm_analysis']['main_category'].append(category.iloc[0])
        num_nonempty_mdrs = len([ae for ae in adverse_events if len(ae['mdr_texts'])])
        tot_mdrs += len(adverse_events)
        tot_nonempty_mdrs += num_nonempty_mdrs
        if device_num[0] == "K":
            tot_510k_mdrs += len(adverse_events)
            tot_510k_nonempty_mdrs += num_nonempty_mdrs
            if device_data['predicate_number']:
                tot_510k_predicate_mdrs += len(adverse_events)
    
        # Extract predicate AE and recall data directly from the JSON
        predicate_has_ae = device_data.get('predicate_has_ae', False)
        predicate_has_recall = device_data.get('predicate_has_recall', False)
        
        processed_data[device_num] = {
            'predicate_has_ae': 1 if predicate_has_ae else 0,
            'predicate_has_recall': 1 if predicate_has_recall else 0,
            'adverse_events': {
                'count': len(adverse_events),
                'data': adverse_events
            }
        }
    
    logging.info(f"Processed adverse events data for {len(processed_data)} devices.")
    logging.info(f"Tot number MDRS {tot_mdrs}.")
    logging.info(f"Tot nonempty MDRS {tot_nonempty_mdrs}.")
    logging.info(f"Tot 510k MDRS {tot_510k_mdrs}.")
    logging.info(f"Tot 510k nonempty MDRS {tot_510k_nonempty_mdrs}.")
    logging.info(f"Tot 510k predicate-available MDRS {tot_510k_predicate_mdrs}.")
    return processed_data

# --- Main Execution ---
def main(args):
    logging.info("Starting pre-post market data merging process...")

    # Load data sources
    device_results_data = load_jsonl(args.device_results_file)
    ae_results_data = load_json(args.ae_results_file)
    device_problems_df = pd.read_csv(PROJECT_ROOT / "data/FDA-CDRH_NCIt_Subsets_categorized.csv")
    
    if device_results_data is None:
        raise ValueError("Device results data could not be loaded. Exiting.")
    
    if ae_results_data is None:
        raise ValueError("Adverse events data could not be loaded. Exiting.")

    # Process adverse events data into dictionary
    device_data_lookup = process_adverse_events_data(ae_results_data, device_problems_df)

    # Merge features into device results data
    merged_data = []
    devices_missing_predicate_data = 0
    devices_with_adverse_events = 0
    devices_missing_decision_date = 0

    for record in device_results_data:
        device_num = record.get('device_number')
        predicate_num = record.get('primary_predicate')
        
        if not device_num:
            logging.warning(f"Skipping device record due to missing device_number: {record}")
            continue
            
        if not device_num.startswith("K"):
            logging.info(f"Skipping device: {device_num}")
            continue
            
        if not predicate_num:
            logging.warning(f"Skipping device record due to missing predicate: {device_num}")
            continue

        # Process device features (boolean to integer conversion)
        device_features = process_device_features(record)
        
        # Get metadata from the record
        metadata = record.get('metadata', {})
        decision_date = metadata.get('decision_date', '')
        primary_product_code = metadata.get('primary_product_code', '')
        panel_lead = metadata.get('panel_lead', '')
        clearance_type = metadata.get('clearance_type', '')
        
        if not decision_date:
            devices_missing_decision_date += 1
            print(f"Missing decision date for device: {device_num}")

        # Get device data (includes predicate AE/recall and device AE data)
        device_data = device_data_lookup.get(device_num, {})
        
        if not device_data:
            devices_missing_predicate_data += 1
            # Add default values if missing
            device_data = {
                'predicate_has_ae': 0,
                'predicate_has_recall': 0,
                'adverse_events': {'count': 0, 'data': []}
            }

        # Get device-level adverse events data
        device_adverse_events = device_data.get('adverse_events', {'count': 0, 'data': []})
        if device_adverse_events['count'] > 0:
            devices_with_adverse_events += 1

        # Create merged record
        merged_record = {
            'device_number': device_num,
            'decision_date': decision_date,
            'primary_predicate': predicate_num,
            'primary_product_code': primary_product_code,
            'panel_lead': panel_lead,
            'clearance_type': clearance_type,
            'predicates': record.get('predicates', []),
            'is_ai_ml': 1,  # All devices in this dataset are AI/ML
            **device_features,
            'predicate_has_ae': device_data.get('predicate_has_ae', 0),
            'predicate_has_recall': device_data.get('predicate_has_recall', 0),
            'adverse_events': device_adverse_events
        }

        merged_data.append(merged_record)
    
    logging.info(f"Total records processed: {len(merged_data)}")
    logging.info(f"Devices missing predicate AE/recall data: {devices_missing_predicate_data}")
    logging.info(f"Devices with adverse events: {devices_with_adverse_events}")
    logging.info(f"Devices missing decision date: {devices_missing_decision_date}")

    # Save merged data
    with open(args.output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    print(f"Successfully saved merged data to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge pre-market device features with post-market adverse events data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device_results_file",
        type=str,
        default=str(OUTPUT_DIR / "aiml_device_results_with_metadata.jsonl"),
        help="Path to the device results JSONL file."
    )
    parser.add_argument(
        "--ae_results_file",
        type=str,
        default=str(PROJECT_ROOT / "scripts/analysis_ae_recall/output/adverse_events_analysis_results_event_prob_struct.json"),
        help="Path to the adverse events analysis results JSON file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(OUTPUT_DIR / "merged_pre_post_market_data_mapped.json"),
        help="Path to the output merged data JSON file."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=str(OUTPUT_DIR / "log_merge.txt"),
    )

    args = parser.parse_args()
    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        filename=args.log_file,
        format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)