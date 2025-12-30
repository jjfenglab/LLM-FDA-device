import sys
import os
from dotenv import load_dotenv
import json
import argparse
import asyncio
import threading
import pandas as pd
import queue
from queue import Queue
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.common import *
from scripts.utils.pdf_utils import extract_text_from_pdf  # noqa: E402
from scripts.prompts.prompts import PROMPT_DEVICE_FEATURES_CONSOLIDATED  # noqa: E402
from scripts.utils.gpt_utils import calculate_gpt_cost  # noqa: E402
from scripts.utils.extract_primary_predicate import extract_potential_device_numbers  # noqa: E402

# Constants
DEFAULT_MODEL_NAME = "gpt-4.1"
GPT_SEED = 25
MAX_CONCURRENT_REQUESTS = 15

# Constants
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Functions imported from other modules - see imports above

def load_device_numbers(device_list_file: str) -> List[str]:
    """Load device numbers from a JSON file."""
    with open(device_list_file, 'r') as f:
        data = json.load(f)
        return data['device_numbers']

def make_default_result(device_number):
    return {
        "device_number": device_number,
        "predicates": [],
        "primary_predicate": "",
        "intended_use_and_clinical_applications": False,
        "operational_and_workflow_change": False,
        "algorithm_or_software_feature_changes": False,
        "hardware_changes": False,
        "body_part_changes": False,
        "human_device_team_testing": False,
        "has_clinical_testing": False,
    }

async def extract_device_features(
    client: AsyncOpenAI,
    device_number: str,
    pdf_text: str,
    potential_predicates: List[str],
    model: str = DEFAULT_MODEL_NAME,
    seed: int = GPT_SEED
) -> Tuple[Dict, float]:
    """Extract all device features using the consolidated prompt."""
    default_result = make_default_result(device_number)

    assert device_number[0] == "K"

    # Define the JSON schema for all expected fields
    json_schema = {
        "name": "device_features_extraction",
        "description": "Structured extraction of device features including predicates, differences, and validation methods.",
        "schema": {
            "type": "object",
            "properties": {
                # Task 1: Predicate Identification
                "predicates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of predicate device K numbers"
                },
                "primary_predicate": {
                    "type": "string",
                    "description": "Primary predicate K number"
                },
                # Task 2: Device Differences (all boolean)
                "intended_use_and_clinical_applications": {"type": "boolean"},
                "operational_and_workflow_change": {"type": "boolean"},
                "algorithm_or_software_feature_changes": {"type": "boolean"},
                "hardware_changes": {"type": "boolean"},
                "body_part_changes": {"type": "boolean"},
                # Task 3: Validation Methods (all boolean)
                "human_device_team_testing": {"type": "boolean"},
                "has_clinical_testing": {"type": "boolean"}
            },
            "required": [
                "predicates", "primary_predicate",
                "intended_use_and_clinical_applications", "operational_and_workflow_change",
                "algorithm_or_software_feature_changes", "hardware_changes",
                "body_part_changes",
                "human_device_team_testing", "has_clinical_testing"
            ],
            "additionalProperties": False
        }
    }

    try:
        prompt = PROMPT_DEVICE_FEATURES_CONSOLIDATED.format(
            text=pdf_text,
            potential_predicate_numbers=potential_predicates
        )
        
        response = await client.chat.completions.create(
            model=model,
            seed=seed,
            response_format={"type": "json_schema", "json_schema": json_schema},
            messages=[
                {"role": "system", "content": "You are a helpful assistant who analyzes medical device summaries and always responds with valid JSON matching the requested schema."},
                {"role": "user", "content": prompt}
            ]
        )
        
        cost = calculate_gpt_cost(model=model, usage=response.usage)
        
        try:
            llm_output = json.loads(response.choices[0].message.content)
            
            # Add device_number to the result
            result = {
                "device_number": device_number,
                **llm_output
            }
            
            return result, cost
            
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error for device {device_number}: {str(e)}")
            print(f"Raw response content: {response.choices[0].message.content}")
            # Return default structure
            default_result["error"] = f"JSON decode error: {str(e)}"
            return default_result, cost
            
    except Exception as e:
        print(f"Error processing device {device_number}: {str(e)}")
        # Return default structure
        default_result['error'] = str(e)
        return default_result, 0.0

async def process_single_device(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    device_number: str,
    model: str,
    results_queue: Queue
) -> None:
    """Process a single device with semaphore control for rate limiting."""
    async with semaphore:
        pdf_path = PROJECT_ROOT / f'data/raw/device_summaries/{device_number}.pdf'
        
        if not pdf_path.exists():
            print(f"PDF not found for device {device_number} at {pdf_path}, skipping...")
            results_queue.put((make_default_result(device_number), 0.0))
            return
        
        if device_number[0] != "K":
            results_queue.put((make_default_result(device_number), 0.0))
            return
        
        try:
            # Extract text synchronously (file I/O)
            pdf_text = extract_text_from_pdf(str(pdf_path))
            
            if not pdf_text:
                print(f"Warning: No text extracted from PDF for {device_number}.")
                return
            
            # Extract potential predicates
            potential_predicates = extract_potential_device_numbers(pdf_text, ignore_set={device_number})
            
            # Process with LLM asynchronously
            result, cost = await extract_device_features(
                client=client,
                device_number=device_number,
                pdf_text=pdf_text,
                potential_predicates=potential_predicates,
                model=model,
                seed=GPT_SEED
            )

            # Put result in queue for writing
            results_queue.put((result, cost))

        except Exception as e:
            print(f"Error processing device {device_number}: {str(e)}")

def write_results_worker(results_queue: Queue, output_file: str, total_devices: int, stop_event: threading.Event, cost_queue: Queue):
    """Worker thread to write results to file as they become available."""
    written_count = 0
    total_cost = 0.0
    
    with open(output_file, 'a') as f:
        while not stop_event.is_set() or not results_queue.empty():
            try:
                # Wait for results with timeout
                result, cost = results_queue.get(timeout=1.0)
                total_cost += cost
                
                if result:
                    f.write(json.dumps(result) + '\n')
                    f.flush()  # Ensure data is written immediately
                
                written_count += 1
                if written_count % 10 == 0:
                    print(f"Written {written_count}/{total_devices} results. Running cost: ${total_cost:.4f}")
                
                results_queue.task_done()
            except queue.Empty:
                continue # Keep waiting
    
    # Put the final cost in the cost queue for the main thread to retrieve
    cost_queue.put(total_cost)
    return total_cost

async def process_devices_parallel(
    devices_to_process: set,
    model: str,
    output_file: str
) -> float:
    """Process devices in parallel with rate limiting."""
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Initialize async OpenAI client
    client = AsyncOpenAI()
    
    # Filter out already processed devices
    remaining_devices = devices_to_process.copy()
    if os.path.exists(output_file):
        precalc_results = load_jsonl(output_file)
        processed_devices = {res['device_number'] for res in precalc_results}
        remaining_devices = remaining_devices - processed_devices
        print(f"Found {len(processed_devices)} already processed devices. Processing remaining {len(remaining_devices)} devices.")
        for dev in remaining_devices:
            print(dev)
        
    # Create queue for results
    results_queue = Queue()

    # Create queue for cost
    cost_queue = Queue()
    
    # Start writer thread
    stop_event = threading.Event()
    writer_thread = threading.Thread(
        target=write_results_worker,
        args=(results_queue, output_file, len(remaining_devices), stop_event, cost_queue)
    )
    writer_thread.start()
    
    # Create tasks for remaining devices
    tasks = []
    for device_number in remaining_devices:
        task = process_single_device(
            semaphore=semaphore,
            client=client,
            device_number=device_number,
            model=model,
            results_queue=results_queue
        )
        tasks.append(task)
    
    # Process remaining devices with progress bar
    print(f"Processing {len(remaining_devices)} devices with up to {MAX_CONCURRENT_REQUESTS} concurrent requests...")
    await tqdm.gather(*tasks, desc="Processing devices")
    
    # Signal writer thread to stop and wait for it
    stop_event.set()
    writer_thread.join()
    
    # Get the total cost from the writer thread
    total_cost = cost_queue.get() if not cost_queue.empty() else 0.0
    return total_cost

async def main_async(args):
    """Main async function to extract all device features."""
    
    # Create output directory if it doesn't exist
    # Load target device numbers
    all_devices = set(load_device_numbers(args.device_list_file))
    print(f"Loaded {len(all_devices)} target device numbers from {args.device_list_file}")

    total_cost = await process_devices_parallel(
        devices_to_process=all_devices,
        model=args.model_name,
        output_file=args.output_file
    )
    
    print(f"Feature extraction completed. Results saved to {args.output_file}")
    print(f"Total API cost for this run: ${total_cost:.4f}")
    

def main(args):
    """Main function wrapper."""
    asyncio.run(main_async(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract all device features from 510(k) summaries using consolidated prompt. Only processes devices starting with 'K'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device_list_file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "aiml_device_numbers_071025.json"),
        help="Path to JSON file containing device numbers."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(OUTPUT_DIR / "aiml_device_results.jsonl"),
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="OpenAI model to use for extraction."
    )

    parsed_args = parser.parse_args()
    load_dotenv()
    main(parsed_args) 
