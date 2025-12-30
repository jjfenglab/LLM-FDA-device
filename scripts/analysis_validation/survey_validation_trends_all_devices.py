import sys
import json
import argparse
import asyncio
import threading
import pandas as pd
import os
import logging
from dotenv import load_dotenv

from queue import Queue
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.utils.pdf_utils import extract_text_from_pdf  # noqa: E402
from scripts.utils.gpt_utils import calculate_gpt_cost  # noqa: E402
from scripts.utils.extract_primary_predicate import extract_potential_device_numbers  # noqa: E402

from compare_llm_results_with_previous_paper import process_devices_parallel

# Constants
DEFAULT_MODEL_NAME = "gpt-4.1"
GPT_SEED = 25
MAX_CONCURRENT_REQUESTS = 15

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDFS_DIR = "data/raw/device_summaries"
def load_device_numbers(input_file: str) -> List[str]:
    """Load device numbers from the JSON file."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data['device_numbers']

def check_device_availability(device_number: str) -> bool:
    """
    Check if a device has either a text file or PDF file available.
    Returns True if at least one source is available, False otherwise.
    """
    pdf_path = PROJECT_ROOT / PDFS_DIR / f"{device_number}.pdf"
    # Check if PDF file exists
    if pdf_path.exists():
        return True
    
    return False


async def main_async(args):
    """Main async function to compare LLM results with validation data."""
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load device numbers
    device_numbers = load_device_numbers(args.input_file)
    print(f"Loaded {len(device_numbers)} device numbers from {args.input_file}")
    
    # Filter out devices that have neither text files nor PDF files
    print("Checking device availability (text files or PDF files)...")
    available_devices = []
    unavailable_devices = []
    
    for device in device_numbers:
        if check_device_availability(device):
            available_devices.append(device)
        else:
            unavailable_devices.append(device)
    
    print(f"Available devices (with text or PDF): {len(available_devices)}")
    print(f"Unavailable devices (no text or PDF): {len(unavailable_devices)}")
    
    if unavailable_devices:
        print(f"Skipping {len(unavailable_devices)} devices without text or PDF files:")
    
    # Apply limit if specified
    if args.num_devices and args.num_devices > 0:
        devices_to_process = available_devices[:args.num_devices]
        print(f"Processing first {args.num_devices} available devices")
    else:
        devices_to_process = available_devices
        print(f"Processing all {len(devices_to_process)} available devices")

    # Process devices
    if not devices_to_process:
        print("No devices to process.")
        return
    
    # List to store all results for comparison
    all_results = []
    
    print(f"Processing {len(devices_to_process)} devices...")
    total_cost = await process_devices_parallel(
        devices_to_process=devices_to_process,
        model=args.model_name,
        output_file=args.output_file,
        all_results=all_results
    )
    
    print(f"\nFeature extraction completed. Results saved to {args.output_file}")
    print(f"Total API cost for this run: ${total_cost:.4f}")
    
def main(args):
    """Main function wrapper."""
    asyncio.run(main_async(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare LLM extraction results with previous paper's validation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="../../data/aiml_device_numbers_071025.json",
        help="Path to the JSON file containing device numbers."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(OUTPUT_DIR / "aiml_devices_validation_results.jsonl"),
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="OpenAI model to use for extraction."
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=None,
        help="Process only the first N devices from the validation list. If not specified, process all devices."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(OUTPUT_DIR / "log_validation_survey.txt"),
    )
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=args.log_file)
    
    load_dotenv()    
    if not os.environ.get("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    main(args)