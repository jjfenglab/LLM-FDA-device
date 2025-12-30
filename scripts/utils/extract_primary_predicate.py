import sys
import json
import argparse
import re
from typing import List, Set, Optional, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.pdf_utils import extract_text_from_pdf  # noqa: E402
from scripts.prompts.prompts import PROMPT_GET_PREDICATES  # noqa: E402
from scripts.utils.gpt_utils import calculate_gpt_cost  # noqa: E402

DEFAULT_MODEL_NAME = "gpt-4.1-mini"
GPT_SEED = 25

def load_device_numbers(device_list_file: str) -> List[str]:
    """Load device numbers from a JSON file."""
    try:
        with open(device_list_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'device_numbers' in data:
                return data['device_numbers']
            elif isinstance(data, list):
                return data
            else:
                raise ValueError(f"Unexpected format in device list file {device_list_file}")
    except Exception as e:
        print(f"Error loading device numbers: {e}")
        raise

def extract_potential_device_numbers(text: str, ignore_set: Optional[Set[str]] = None) -> List[str]:
    """
    Find potential predicate device numbers from a 510(k) summary text.
    Valid formats: K######, P######, DEN######.
    """
    pattern = r'\b(K\d{6}|P\d{6}|DEN\d{6})\b'
    matches = re.findall(pattern, text.upper()) # Search in uppercase text
    unique_matches = sorted(list(set(matches))) # Sort for consistent order

    if ignore_set:
        # Ensure items in ignore_set are uppercase for comparison
        upper_ignore_set = {item.upper() for item in ignore_set}
        unique_matches = [match for match in unique_matches if match not in upper_ignore_set]
        
    return unique_matches

def get_primary_predicate_from_llm(
    client: OpenAI,
    device_summary_text: str,
    potential_predicates: List[str],
    model: str,
    seed: int
) -> Tuple[Dict[str, Any], float]:
    """
    Call LLM to extract primary predicate and all predicates from text.
    """
    if not potential_predicates:
        return {"predicates": [], "primary_predicate": ""}, 0.0

    prompt = PROMPT_GET_PREDICATES.format(
        text=device_summary_text,
        potential_predicate_numbers=potential_predicates
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Respond with a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            seed=seed
        )
        
        cost = calculate_gpt_cost(model=model, usage=response.usage)
        raw_content = response.choices[0].message.content
        
        try:
            predicate_info = json.loads(raw_content)
            # Ensure essential keys exist, defaulting to safe values
            if "predicates" not in predicate_info:
                predicate_info["predicates"] = []
            if "primary_predicate" not in predicate_info:
                predicate_info["primary_predicate"] = ""
            # Ensure primary_predicate is a string
            if not isinstance(predicate_info.get("primary_predicate"), str):
                 predicate_info["primary_predicate"] = ""
                 
            return predicate_info, cost
        except json.JSONDecodeError as e_json:
            print(f"JSON Decode Error for predicate extraction: {e_json}. Raw content: '{raw_content}'")
            return {"predicates": [], "primary_predicate": "", "error": f"JSON decode error: {e_json}", "raw_content": raw_content}, cost

    except Exception as e:
        print(f"Error during OpenAI API call for predicate extraction: {e}")
        return {"predicates": [], "primary_predicate": "", "error": str(e)}, 0.0


def main(args):
    client = OpenAI()

    try:
        device_numbers_to_process = load_device_numbers(args.device_list_file)
        print(f"Loaded {len(device_numbers_to_process)} device numbers from {args.device_list_file}")
    except Exception as e:
        print(f"Failed to load device numbers: {e}")
        sys.exit(1)

    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    total_api_cost = 0.0
    results = []
    errors_occurred = 0

    print(f"Starting primary predicate extraction. Output will be saved to {args.output_file}")

    for device_number in tqdm(device_numbers_to_process, desc="Extracting predicates"):
        current_result = {
            "device_number": device_number,
            "primary_predicate": "",
            "all_predicates": [],
            "all_extracted_device_numbers": []
        }
        
        if not device_number.startswith("K"):
            print(f"Device {device_number} does not start with 'K'. Skipping LLM processing.")
            results.append(current_result) # Appends with default empty predicate fields
            continue

        pdf_path = PROJECT_ROOT / f'data/raw/device_summaries/{device_number}.pdf'
        if not pdf_path.exists():
            print(f"PDF not found for {device_number} at {pdf_path}. Skipping.")
            current_result["error"] = "PDF not found" # Add error key for this specific failure
            errors_occurred += 1
            results.append(current_result)
            continue

        try:
            pdf_text = extract_text_from_pdf(str(pdf_path))
            if not pdf_text:
                print(f"Warning: No text extracted from PDF for {device_number}.")
                current_result["error"] = "No text extracted from PDF" # Add error key
                errors_occurred += 1
                results.append(current_result)
                continue
        
            potential_predicates = extract_potential_device_numbers(pdf_text, ignore_set={device_number})
            current_result["all_extracted_device_numbers"] = potential_predicates
            
            if not potential_predicates:
                print(f"No potential predicates found for {device_number} (excluding self).")
                results.append(current_result) # Appends with empty primary_predicate and all_predicates
                continue

            llm_response, cost = get_primary_predicate_from_llm(
                client=client,
                device_summary_text=pdf_text,
                potential_predicates=potential_predicates,
                model=args.model_name,
                seed=GPT_SEED
            )
            total_api_cost += cost
            
            if "error" in llm_response:
                print(f"LLM error for device {device_number}: {llm_response['error']}")
                if "raw_content" in llm_response:
                    print(f"LLM raw content on error for {device_number}: {llm_response['raw_content']}")
                errors_occurred +=1
                # primary_predicate and all_predicates remain default empty in current_result
            else:
                current_result["primary_predicate"] = llm_response.get("primary_predicate", "")
                current_result["all_predicates"] = llm_response.get("predicates", [])
            
            results.append(current_result)

        except Exception as e:
            print(f"Unexpected error processing device {device_number}: {e}")
            current_result["error"] = f"Unexpected error: {str(e)}" # Add error key for this failure type
            errors_occurred += 1
            results.append(current_result)

    # Save all results to JSONL file
    try:
        with open(args.output_file, 'w') as f:
            for entry in results:
                f.write(json.dumps(entry) + '\n')
        print(f"\nExtraction finished. Results saved to {args.output_file}")
    except IOError as e:
        print(f"Error writing output file {args.output_file}: {e}")

    print(f"Processed {len(device_numbers_to_process)} devices.")
    if errors_occurred > 0:
        print(f"Encountered errors during processing for {errors_occurred} devices. Check output file for details.")
    print(f"Total estimated API cost for this run: ${total_api_cost:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract primary predicate from 510(k) summaries using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device_list_file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "aiml_device_numbers.json"),
        help="Path to JSON file containing device numbers."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "validation" / "primary_predicates.jsonl"),
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="OpenAI model to use for extraction."
    )
    
    # Example:
    # python scripts/validation/extract_primary_predicate.py --device_list_file data/raw/sample_device_numbers.json --output_file data/processed/validation/sample_primary_predicates.jsonl

    parsed_args = parser.parse_args()
    main(parsed_args)