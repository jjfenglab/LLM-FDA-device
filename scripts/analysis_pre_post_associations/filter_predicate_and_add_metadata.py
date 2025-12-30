#!/usr/bin/env python3
"""
Filter Predicate and Add Metadata Script

This script:
1. Loads device data from a JSONL file
2. Handles primary_predicate logic (uses first predicate if empty)
3. Filters out devices with empty primary_predicate
4. Fetches device metadata from FDA API only for remaining devices
5. Adds metadata to each device record
6. Saves results as JSONL
"""

import logging
import json
import requests
import time
import sys
import os
from pathlib import Path
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import argparse

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.common import *

def fetch_device_metadata_from_openfda(device_number, api_key=None, max_retries=3):
    """
    Fetch device metadata from OpenFDA API for a given device number
    
    Args:
        device_number (str): The K number or DEN number to search for
        api_key (str): FDA API key for higher rate limits
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        dict: Device metadata or None if not found/error
    """
    
    # OpenFDA API endpoint for device clearances
    base_url = "https://api.fda.gov/device/510k.json"
    
    # Search parameters - try searching by k_number field
    params = {
        'search': f'k_number:"{device_number}"',
        'limit': 1
    }
    
    # Add API key if provided
    if api_key:
        params['api_key'] = api_key
    
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    device_data = data['results'][0]
                    
                    # Extract the requested fields
                    extracted_data = {
                        'device_name': device_data.get('device_name', ''),
                        'panel_lead': device_data.get('advisory_committee', ''),
                        'primary_product_code': device_data.get('product_code', ''),
                        'decision_date': device_data.get('decision_date', ''),
                        'date_received': device_data.get('date_received', ''),
                        'clearance_type': device_data.get('clearance_type', '')
                    }
                    
                    return extracted_data
                else:
                    return None
            
            elif response.status_code == 429:
                # Rate limited - wait before retry with exponential backoff
                wait_time = min(60, 2 ** attempt * 5)  # Cap at 60 seconds
                time.sleep(wait_time)
                continue
                
            else:
                if attempt == max_retries - 1:
                    return None
        
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return None
            
            # Wait before retry
            time.sleep(2)
    
    return None

class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    
    def __init__(self, max_calls_per_minute):
        self.max_calls_per_minute = max_calls_per_minute
        self.min_interval = 60.0 / max_calls_per_minute
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()

def process_device_batch(device_numbers, api_key, rate_limiter, pbar):
    """Process a batch of device numbers with rate limiting"""
    results = {}
    
    for device_number in device_numbers:
        rate_limiter.wait_if_needed()
        
        try:
            result = fetch_device_metadata_from_openfda(device_number, api_key=api_key)
            if result:
                results[device_number] = result
            
            # Update progress bar
            pbar.update(1)
            
            # Update progress bar description with success/error info
            if result:
                pbar.set_postfix({'status': 'success', 'device': device_number[:8]})
            else:
                pbar.set_postfix({'status': 'error', 'device': device_number[:8]})
                
        except Exception as e:
            pbar.update(1)
            pbar.set_postfix({'status': 'exception', 'device': device_number[:8]})
    
    return results

def fetch_all_metadata(device_numbers, api_key=None, max_workers=1):
    """
    Fetch metadata for all device numbers
    
    Args:
        device_numbers (list): List of device numbers to fetch metadata for
        api_key (str): FDA API key for higher rate limits
        max_workers (int): Number of concurrent workers
    
    Returns:
        dict: Dictionary mapping device numbers to their metadata
    """
    
    if not device_numbers:
        return {}
    
    # Set up rate limiting based on API key availability
    if api_key:
        # With API key: 240 requests per minute, but be conservative
        max_calls_per_minute = 200  # Leave some buffer
        max_workers = min(max_workers, 4)  # Limit concurrent workers
    else:
        # Without API key: 40 requests per minute, be very conservative
        max_calls_per_minute = 35  # Leave buffer
        max_workers = 1  # Single threaded without API key
    
    rate_limiter = RateLimiter(max_calls_per_minute)
    
    print(f"Fetching metadata for {len(device_numbers)} device numbers...")
    print(f"API Key: {'Present' if api_key else 'Not provided (using public rate limits)'}")
    print(f"Rate limit: {max_calls_per_minute} calls/minute")
    print(f"Max workers: {max_workers}")
    print("-" * 50)
    
    all_results = {}
    
    # Process devices with progress bar
    with tqdm(total=len(device_numbers), desc="Fetching device metadata", unit="device") as pbar:
        
        if max_workers == 1:
            # Single-threaded processing
            results = process_device_batch(device_numbers, api_key, rate_limiter, pbar)
            all_results.update(results)
            
        else:
            # Multi-threaded processing with batching
            batch_size = max(1, len(device_numbers) // (max_workers * 4))  # Create multiple batches per worker
            batches = [device_numbers[i:i + batch_size] for i in range(0, len(device_numbers), batch_size)]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {
                    executor.submit(process_device_batch, batch, api_key, rate_limiter, pbar): batch 
                    for batch in batches
                }
                
                for future in as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        all_results.update(batch_results)
                        
                    except Exception as e:
                        print(f"Batch processing error: {e}")
    
    return all_results

def load_existing_device_numbers(output_file):
    """
    Load device numbers that already exist in the output file
    
    Args:
        output_file (str): Path to the output JSONL file
    
    Returns:
        set: Set of device numbers already in the output file
    """
    existing_device_numbers = set()
    
    if Path(output_file).exists():
        try:
            existing_devices = load_jsonl(output_file)
            existing_device_numbers = {device['device_number'] for device in existing_devices}
            print(f"Found {len(existing_device_numbers)} devices already in output file")
        except Exception as e:
            print(f"Warning: Could not load existing output file {output_file}: {e}")
    else:
        print(f"Output file {output_file} does not exist yet, will create new file")
    
    return existing_device_numbers

def main():
    parser = argparse.ArgumentParser(description="Filter by primary predicate and add device metadata")
    parser.add_argument("--device-results-file", default="output/aiml_device_results.jsonl",
                        help="Input device results JSONL file")
    parser.add_argument("--output-file", default="output/aiml_device_results_with_metadata.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of concurrent workers")
    parser.add_argument("--log-file", type=str, default='output/log_metadata.txt')
    
    args = parser.parse_args()
    # --- Logging Setup ---
    logging.basicConfig(level=logging.INFO, filename=args.log_file)
    
    # Get API key from environment
    api_key = os.getenv('FDA_API_KEY')
    
    # Adjust concurrency based on whether we have an API key
    if api_key:
        max_workers = min(args.max_workers, 3)  # Conservative concurrency with API key
        print("Using FDA API key - optimized rate limits and concurrency")
    else:
        max_workers = 1  # Single-threaded without API key
        print("No FDA API key found - using conservative single-threaded processing")
        print("Set FDA_API_KEY environment variable for faster processing")
    
    # Check if device results file exists
    if not Path(args.device_results_file).exists():
        print(f"Error: Device results file '{args.device_results_file}' not found.")
        print(f"Please make sure the file exists or update the device_results_file path.")
        return
    
    # Load device data
    print(f"Loading device data from {args.device_results_file}...")
    devices = load_jsonl(args.device_results_file)

    if not devices:
        print("No device data found")
        return
    
    print(f"Loaded {len(devices)} devices")
    
    # Load existing device numbers from output file
    # Filter out if already existing
    existing_device_numbers = load_existing_device_numbers(args.output_file)
    print("Filtering devices...")
    filtered_devices = []
    for device in devices:
        # Filter out devices already processed
        if device['device_number'] not in existing_device_numbers:
            filtered_devices.append(device)
    
    logging.info(f"Initial num devices: {len(devices)}")
    logging.info(f"Skipped {len(existing_device_numbers)} devices already in output file")
    logging.info(f"New devices to process: {len(filtered_devices)}")
    
    if not filtered_devices:
        print("No new devices to process")
        return
    
    # Get device numbers for metadata fetching
    device_numbers_to_fetch = [device['device_number'] for device in filtered_devices]
    
    # Fetch metadata for remaining devices
    try:
        metadata_dict = fetch_all_metadata(
            device_numbers=device_numbers_to_fetch,
            api_key=api_key,
            max_workers=max_workers
        )
    except Exception as e:
        print(f"Fatal error fetching metadata: {e}")
        return
    
    # Add metadata to devices
    print("Adding metadata to devices...")
    for device in filtered_devices:
        device_number = device['device_number']
        if device_number in metadata_dict:
            device['metadata'] = metadata_dict[device_number]
    
    # Calculate statistics
    devices_with_metadata = sum(1 for device in filtered_devices if 'metadata' in device)
    logging.info(f"Num devices with metadata: {devices_with_metadata}")

    # Create output directory if it doesn't exist
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Append results to existing JSONL file
    print(f"Appending {len(filtered_devices)} new devices to {args.output_file}...")
    try:
        with open(args.output_file, 'a') as f:
            for device in filtered_devices:
                f.write(json.dumps(device) + '\n')
        
        print(f"✓ Appended {len(filtered_devices)} new devices to {args.output_file}")
        
    except Exception as e:
        print(f"Error appending results: {e}")
        return
    
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 