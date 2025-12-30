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

from scripts.common import load_jsonl  # noqa: E402
from scripts.utils.pdf_utils import extract_text_from_pdf  # noqa: E402
from scripts.utils.gpt_utils import calculate_gpt_cost  # noqa: E402
from scripts.utils.extract_primary_predicate import extract_potential_device_numbers  # noqa: E402

# Constants
DEFAULT_MODEL_NAME = "gpt-4.1"
GPT_SEED = 25
MAX_CONCURRENT_REQUESTS = 15
VALIDATION_CSV_PATH = "data/raw/validation/zou_clinical_data.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_CLINICAL_STUDY_ANALYSIS = """You will be shown a parsed text from a 510(k) summary of a medical device released by the FDA. 
Your task is to extract key information about the device's predicate(s) and clinical study characteristics.
Output one flat JSON object with the specified keys and value types.

Here is a text from the parsed 510(k) summary: {text}

Instructions:

Task 1: Predicate Identification
- "predicates": (List[str]) All predicate device K numbers explicitly stated in the summary, selected from: {potential_predicate_numbers}
- "primary_predicate": (str) The primary predicate K number, chosen from "predicates". There should be one and only one primary predicate for a device. If none can be confidently identified, use an empty string.

Guidelines:
- Notice that not all device numbers mentioned in the summary are predicates of the target device. Please read the text to find the device numbers that are explicitly stated as predicates.

Task 2: Validation Study Characteristics
- "num_sites_reason": (str) Explain how the number of sites (single vs multisite and exact number of sites) was determined, along with a quote. If the exact number of sites is not mentioned, extract any lower bound that is mentioned. If there is no information whatsoever, mention this instead.
- "num_sites": (int) The number of sites used to validate the subject device, either prospectively or retrospectively. If the summary does not provide the exact number but mentions a lower bound, use that instead. Otherwise, if no information is available at all, set to null.
- "is_multisite": (boolean) The number of sites used to validate the subject device, either prospectively or retrospectively, is two or more. False otherwise.
- "is_prospective_reason": (str) Explain the reasoning for whether prospective validation was conducted, along with a quote.
- "is_prospective": (boolean) True if there was data was collected prospectively to validate the device's performance. False otherwise.

Suggestions for analyzing validation studies:
- Look for mentions of clinical studies, validation studies, or performance evaluations
- Search for terms like "prospective", "retrospective", "concurrent", "real-time data collection"
- Count the number of sites/centers/institutions mentioned in clinical evaluation
- If multiple studies are mentioned, focus on the primary clinical validation study

Examples:
- "A prospective study was conducted at the US and Korea" → num_sites: 2, num_sites_reason: "The text did not state the exact number of sites, but it does mention validating across two countries.", is_prospective: true, is_prospective_reason: "The study is explicitly described as prospective"
- "Retrospective analysis of data from 3 hospitals" → num_sites: 3, num_sites_reason: "The text mentions '3 hospitals' used for data analysis", is_prospective: false, is_prospective_reason: "The analysis is described as retrospective"
- "Multi-site validation study across 10 centers" → num_sites: 10, num_sites_reason: "The text states '10 centers' were involved", is_prospective: false, is_prospective_reason: "No indication of prospective data collection, appears to be validation on existing data"
"""

class ClinicalStudyFeatures(BaseModel):
    """Pydantic model for clinical study features extraction."""
    # Task 1: Predicate Identification
    predicates: List[str] = Field(description="List of predicate device K numbers")
    primary_predicate: str = Field(description="Primary predicate K number")
    
    # Task 2: Clinical Study Characteristics  
    num_sites: Optional[int] = Field(description="Number of clinical sites used in evaluation, null if not mentioned")
    num_sites_reason: str = Field(description="Concise one sentence explaining how the number of sites was determined")
    is_multisite: bool = Field(description="True if two or more sites were used for validation")
    is_prospective: bool = Field(description="True if test data were collected concurrently with device deployment")
    is_prospective_reason: str = Field(description="Concise one sentence explaining the reasoning for prospective/retrospective determination")


def load_validation_data() -> pd.DataFrame:
    """Load and preprocess the validation data from Zou et al. paper."""
    df = pd.read_csv(PROJECT_ROOT / VALIDATION_CSV_PATH)
    # Uppercase the approval numbers
    df['approval_number'] = df['approval_number'].str.upper()
    df['num_sites'] = df['num_sites'].str.replace(">","")
    return df


async def extract_clinical_study_features(
    client: AsyncOpenAI,
    device_number: str,
    pdf_text: str,
    potential_predicates: List[str],
    model: str = DEFAULT_MODEL_NAME,
    seed: int = GPT_SEED
) -> Tuple[Dict, float]:
    """Extract clinical study features using the clinical study analysis prompt."""
    prompt = PROMPT_CLINICAL_STUDY_ANALYSIS.format(
        text=pdf_text,
        potential_predicate_numbers=potential_predicates
    )
    logging.info(prompt)
    try:
        response = await client.chat.completions.parse(
            model=model,
            seed=seed,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who analyzes medical device summaries and always responds with valid JSON matching the requested schema."},
                {"role": "user", "content": prompt}
            ],
            response_format=ClinicalStudyFeatures,
        )
        
        cost = calculate_gpt_cost(model=model, usage=response.usage)
        
        llm_output = response.choices[0].message.parsed.model_dump()
        
        # Add device_number to the result
        result = {
            "device_number": device_number,
            **llm_output
        }
        
        return result, cost
            
    except Exception as e:
        print(f"Error for device {device_number}: {str(e)}")
        print(f"Raw response content: {response.choices[0].message}")
        # Return default structure
        result = {
            "device_number": device_number,
            "predicates": [],
            "primary_predicate": "",
            "num_sites": None,
            "num_sites_reason": "Error in LLM response processing",
            "is_multisite": False,
            "is_prospective": False,
            "is_prospective_reason": "Error in LLM response processing",
            "error": f"Error: {str(e)}"
        }
        return result, cost

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
            result, cost = await extract_clinical_study_features(
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

def write_results_worker(results_queue: Queue, output_file: str, total_devices: int, stop_event: threading.Event, cost_queue: Queue, all_results: List[Dict]):
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
                    all_results.append(result)  # Store for comparison
                
                written_count += 1
                if written_count % 10 == 0:
                    print(f"Written {written_count}/{total_devices} results. Running cost: ${total_cost:.4f}")
                
                results_queue.task_done()
                
            except Exception:
                # Timeout or other error - continue if stop_event is not set
                continue
    
    # Put the final cost in the cost queue for the main thread to retrieve
    cost_queue.put(total_cost)
    return total_cost

async def process_devices_parallel(
    devices_to_process: List[str],
    model: str,
    output_file: str,
    all_results: List[Dict]
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
        remaining_devices = [dev for dev in remaining_devices if dev not in processed_devices]
        print(f"Found {len(processed_devices)} already processed devices. Processing remaining {len(remaining_devices)} devices.")
        print(remaining_devices)
    
    # Create queues for results and cost
    results_queue = Queue()
    cost_queue = Queue()
    
    # Start writer thread
    stop_event = threading.Event()
    writer_thread = threading.Thread(
        target=write_results_worker,
        args=(results_queue, output_file, len(remaining_devices), stop_event, cost_queue, all_results)
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
    """Main async function to compare LLM results with validation data."""
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load validation data
    validation_df = load_validation_data()
    print(f"Loaded {len(validation_df)} devices from validation data")
    
    # Get device numbers from validation data
    all_devices = validation_df['approval_number'].tolist()
    
    # Apply limit if specified
    if args.num_devices and args.num_devices > 0:
        devices_to_process = all_devices[:args.num_devices]
        print(f"Processing first {args.num_devices} devices")
    else:
        devices_to_process = all_devices
        print(f"Processing all {len(devices_to_process)} devices")

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
        "--output-file",
        type=str,
        default=str(OUTPUT_DIR / "llm_validation_comparison_new.jsonl"),
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
        default=str(OUTPUT_DIR / "log_validation.txt"),
    )
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=args.log_file)
    
    load_dotenv()    
    if not os.environ.get("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    main(args)