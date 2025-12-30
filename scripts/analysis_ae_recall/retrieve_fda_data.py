#!/usr/bin/env python3
"""
FDA Data Retrieval Script
Retrieves adverse events and recall data from FDA APIs for medical devices.
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm
import pandas as pd

# python scripts/ae_recall_analysis/ae_recall_analysis.py --input scripts/ae_recall_analysis/input/aiml_devices_for_analysis.json

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.common import *

# Constants
MAUDE_API_URL = "https://api.fda.gov/device/event.json"
RECALL_API_URL = "https://api.fda.gov/device/recall.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parallel processing settings
DEFAULT_MAX_WORKERS = 4
DEFAULT_RATE_LIMIT_DELAY = 0.1

class FDAAPI:
    """FDA API client with rate limiting and connection pooling."""
    
    def __init__(self, rate_limit_delay: float = 0.1):
        self.session = requests.Session()
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
        # Configure session for better performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        if os.environ.get("FDA_API_KEY"):
            self.session.headers.update({'Authorization': f'Bearer {os.environ.get("FDA_API_KEY")}'})
    
    def _rate_limit(self):
        """Implement rate limiting to avoid overwhelming the FDA API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _request(self, url: str, params: Dict) -> Optional[Dict]:
        """Make API request with retry logic and rate limiting."""
        self._rate_limit()
        
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None
    
    def check_has_events(self, device_number: str, before_date: Optional[str] = None, is_recall: bool = False) -> bool:
        """Check if device has adverse events or recalls."""
        if not device_number:
            return False
        
        url = RECALL_API_URL if is_recall else MAUDE_API_URL
        search_field = "k_numbers" if is_recall else "pma_pmn_number"
        date_field = "event_date_initiated" if is_recall else "date_received"
        
        search_query = f"{search_field}:{device_number}"
        if before_date:
            formatted_date = datetime.strptime(before_date, "%Y-%m-%d").strftime("%Y%m%d")
            search_query += f" AND {date_field}:[* TO {formatted_date}]"
        
        data = self._request(url, {"search": search_query, "limit": 1})
        return bool(data and data.get('results'))
    
    def fetch_events(self, device_number: str) -> List[Dict]:
        """Fetch all adverse events for a device."""
        if not device_number:
            return []
        
        all_events = []
        skip = 0
        
        while True:
            params = {"search": f"pma_pmn_number:{device_number}", "limit": 100, "skip": skip}
            data = self._request(MAUDE_API_URL, params)
            
            if not data or not data.get('results'):
                break
            
            for result in data['results']:
                print(result)

            events = [{
                'report_number': result.get('report_number'),
                'date_of_event': result.get('date_of_event'),
                'date_report': result.get('date_report'),
                'event_type': result.get('event_type'),
                'report_source_code': result.get('report_source_code'),
                'product_problems': result.get('product_problems', []),
                'mdr_texts': [
                    f"{text.get('text_type_code', 'MDR text')}: {text.get('text', '')}"
                    for text in result.get('mdr_text', [])]
            } for result in data['results']]
            
            all_events.extend(events)
            skip += len(events)
            
            if len(events) < 100:
                break
        
        return all_events


class FDADataRetriever:
    """Main class for retrieving FDA data."""
    
    def __init__(self, rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY):
        self.fda_api = FDAAPI(rate_limit_delay=rate_limit_delay)
        self.stats = {'total_devices': 0, 'devices_with_ae': 0, 'total_events': 0, 
                     'predicates_with_ae': 0, 'predicates_with_recalls': 0}
    
    def process_device(self, device_data: Dict) -> Dict:
        """Process a single device and retrieve FDA data."""
        device_number = device_data.get("device_number")
        predicate_number = device_data.get("primary_predicate")
        date_received = device_data.get("date_received")
        
        logging.info(f"Processing device {device_number}")
        
        result = {
            "device_number": device_number,
            "predicate_number": predicate_number,
            "date_received": date_received,
            "has_ae": False,
            "adverse_events": [],
            "predicate_has_ae": False,
            "predicate_has_recall": False
        }
        
        # Process subject device events
        events = self.fda_api.fetch_events(device_number)
        if events:
            result["has_ae"] = True
            result["adverse_events"] = events
            self.stats['devices_with_ae'] += 1
            self.stats['total_events'] += len(events)
            
            logging.info(f"Device {device_number}: Found {len(events)} adverse events")
        else:
            logging.info(f"Device {device_number}: No adverse events found")
        
        # Process predicate device
        if predicate_number and date_received:
            result["predicate_has_ae"] = self.fda_api.check_has_events(predicate_number, date_received)
            result["predicate_has_recall"] = self.fda_api.check_has_events(predicate_number, date_received, is_recall=True)
            
            if result["predicate_has_ae"]:
                self.stats['predicates_with_ae'] += 1
                logging.info(f"Device {device_number}: Predicate {predicate_number} has adverse events")
            else:
                logging.info(f"Device {device_number}: Predicate {predicate_number} has no adverse events")
                
            if result["predicate_has_recall"]:
                self.stats['predicates_with_recalls'] += 1
                logging.info(f"Device {device_number}: Predicate {predicate_number} has recalls")
            else:
                logging.info(f"Device {device_number}: Predicate {predicate_number} has no recalls")
        
        return result
    
    def retrieve_data(self, devices: List[Dict], output_file: str, max_workers: int = DEFAULT_MAX_WORKERS) -> List[Dict]:
        """Retrieve FDA data for all devices with parallel processing."""
        logging.info(f"Starting FDA data retrieval for {len(devices)} devices with {max_workers} workers")
        
        self.stats['total_devices'] = len(devices)
        results = []
        
        # Load existing results if available
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
                # Update stats based on existing results
                self.stats['devices_with_ae'] = sum(1 for r in results if r.get('has_ae', False))
                self.stats['total_events'] = sum(len(r.get('adverse_events', [])) for r in results)
                self.stats['predicates_with_ae'] = sum(1 for r in results if r.get('predicate_has_ae', False))
                self.stats['predicates_with_recalls'] = sum(1 for r in results if r.get('predicate_has_recall', False))
            logging.info(f"Loaded {len(results)} existing results from output file")
        
        # Filter out already processed devices
        processed_device_numbers = {r.get('device_number') for r in results}
        remaining_devices = [d for d in devices if d['device_number'] not in processed_device_numbers]
        
        if not remaining_devices:
            logging.info("All devices already processed. Loading final results.")
            return results
        
        logging.info(f"Processing {len(remaining_devices)} remaining devices out of {len(devices)} total")
        
        # Process devices in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all remaining device processing tasks
            future_to_device = {
                executor.submit(self.process_device, device_data): device_data 
                for device_data in remaining_devices
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(remaining_devices), desc="Retrieving FDA data") as pbar:
                for future in as_completed(future_to_device):
                    device_data = future_to_device[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update progress
                        pbar.update(1)
                        pbar.set_postfix({
                            'devices_with_ae': self.stats['devices_with_ae'],
                            'total_events': self.stats['total_events']
                        })
                        
                        # Save progress after every device
                        with open(output_file, 'w') as f:
                            json.dump(results, f, indent=2)
                                
                    except Exception as e:
                        logging.error(f"Error processing device {device_data.get('device_number', 'unknown')}: {e}")
                        pbar.update(1)
                        continue
        
        # Log final summary statistics
        logging.info("=" * 60)
        logging.info("FDA DATA RETRIEVAL SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total devices processed: {self.stats['total_devices']}")
        logging.info(f"Devices with adverse events: {self.stats['devices_with_ae']} ({self.stats['devices_with_ae']/max(self.stats['total_devices'],1)*100:.1f}%)")
        logging.info(f"Total adverse events found: {self.stats['total_events']}")
        logging.info(f"Predicates with adverse events: {self.stats['predicates_with_ae']}")
        logging.info(f"Predicates with recalls: {self.stats['predicates_with_recalls']}")
        logging.info("=" * 60)
        logging.info("FDA data retrieval completed successfully")
        
        return results


def export_mdr_summary_to_csv(output_file: str, csv_filename: str):
    """Export MDR text summary statistics to CSV file.
    
    Args:
        output_file (str): Path to the JSON file containing FDA data results
        csv_filename (str): Output CSV filename
    """
    # Load results from JSON file
    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read output file: {e}")
        return
    
    print(f"Processing MDR texts from {len(results)} devices for CSV export...")
    
    # Prepare data for CSV
    csv_data = []
    
    for result in results:
        device_number = result.get('device_number', '')
        adverse_events = result.get('adverse_events', [])
        
        # Process each adverse event
        for event in adverse_events:
            report_number = event.get('report_number', '')
            mdr_texts = event.get('mdr_texts', [])
            
            # Process each MDR text
            for mdr_text in mdr_texts:
                if isinstance(mdr_text, str) and mdr_text.strip():
                    # Count characters and words
                    text_content = mdr_text.strip()
                    char_count = len(text_content)
                    word_count = len(text_content.split())
                    
                    csv_row = {
                        'device_number': device_number,
                        'mdr_report_number': report_number,
                        'number_of_characters': char_count,
                        'number_of_words': word_count
                    }
                    csv_data.append(csv_row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False)
    
    # Print summary
    total_reports = len(set(row['mdr_report_number'] for row in csv_data))
    total_devices = len(set(row['device_number'] for row in csv_data))
    avg_chars = df['number_of_characters'].mean()
    avg_words = df['number_of_words'].mean()
    
    print(f"MDR summary exported to CSV: {csv_filename}")
    print(f"Total MDR text entries: {len(csv_data)}")
    print(f"Unique devices with MDR texts: {total_devices}")
    print(f"Unique MDR reports: {total_reports}")
    print(f"Average characters per MDR text: {avg_chars:.1f}")
    print(f"Average words per MDR text: {avg_words:.1f}")


def main():
    parser = argparse.ArgumentParser(description="FDA Data Retrieval Script")
    parser.add_argument(
        "--input-all",
        type=str,
        default="../../data/aiml_device_numbers_071025.json",
        help="Path to the JSON file containing device numbers."
    )
    parser.add_argument(
        "--input-510k",
        type=str,
        default="../analysis_pre_post_associations/output/aiml_device_results_with_metadata.jsonl",
        help="Path to the JSON file containing device numbers."
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, 
                       help=f"Maximum number of parallel workers (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT_DELAY,
                       help=f"Rate limit delay for FDA API requests (default: {DEFAULT_RATE_LIMIT_DELAY})")
    parser.add_argument("--log-file", type=str, default="output/log_ae_retrieve.txt")
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(OUTPUT_DIR / "fda_data_retrieval_results.json"),
        help="Path to the output JSON file."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(OUTPUT_DIR / "table1_mdr.csv"),
        help="CSV file to save MDR text summary statistics"
    )
    
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        filename=args.log_file)
    
    # Load devices
    devices = load_jsonl(args.input_510k)
    curr_dev_nums = {d["device_number"] for d in devices}
    with open(args.input_all, 'r') as f:
        all_dev_nums = set(json.load(f)["device_numbers"])
    remaining_dev_nums = all_dev_nums - curr_dev_nums
    for dev_num in remaining_dev_nums:
        devices.append({
            "device_number": dev_num
        })

    # Run FDA data retrieval
    retriever = FDADataRetriever(rate_limit_delay=args.rate_limit)
    results = retriever.retrieve_data(devices, args.output_file, max_workers=args.max_workers)
    
    logging.info(f"FDA data retrieval completed. Results saved to {args.output_file}")
    logging.info(f"Retrieved data for {len(results)} devices")
    logging.info(f"Devices with adverse events: {sum(1 for r in results if r.get('has_ae', False))}")
    logging.info(f"Total adverse events: {sum(len(r.get('adverse_events', [])) for r in results)}")
    
    # Export MDR summary to CSV
    export_mdr_summary_to_csv(args.output_file, args.csv)


if __name__ == "__main__":
    main()