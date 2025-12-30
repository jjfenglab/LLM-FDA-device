#!/usr/bin/env python3
"""
LLM Categorization Script
Processes adverse events retrieved from FDA and categorizes them using LLM.
"""
from dotenv import load_dotenv
import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.utils.gpt_utils import calculate_gpt_cost  # noqa: E402
from scripts.common import load_jsonl

# Constants
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LLM_MODEL = "gpt-4.1"

# Parallel processing settings
DEFAULT_MAX_WORKERS = 20
NUM_WORKERS_DEVICE = 4


from pydantic import BaseModel

def format_mdr(mdr_text_list):
    mdr_text = ""
    for text in mdr_text_list:
        if text.startswith('Description of Event or Problem'):
            mdr_text += f"{text}\n\n"
    for text in mdr_text_list:
        if text.startswith('Additional'):
            mdr_text += f"{text}\n\n"
    
    # remaining... shouldnt exist though
    for text in mdr_text_list:
        if not text.startswith('Additional') and not text.startswith('Description of Event or Problem'):
            mdr_text += f"{text}\n\n"
    
    return mdr_text

class MDRClassification(BaseModel):
    reasoning: str
    event_type: str
    fda_device_problem_codes: list[str]

class LLMAnalyzer:
    """LLM analyzer for categorizing adverse events."""
    
    def __init__(self, device_problems_df):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.stats = {'labeled': 0, 'total_cost': 0.0}
        self.device_problems_df = device_problems_df

        self.device_problems_str = device_problems_df[['CDRH Preferred Term', 'IMDRF Code']].to_csv(index=False)

    def categorize_event(self, device_name: str, event: Dict) -> Dict:
        """Categorize adverse event using FDA product problems or LLM."""
        event_text = format_mdr(event.get("mdr_texts", []))
        
        if not event_text.strip():
            logging.info("Event has no text content, categorizing as 'Unclear'")
            return {"fda_device_problem_codes": ["Insufficient Information"], "event_type": "no_text"}
        
        # Always use LLM for main category
        print(f"PRODUCT PROBLEMS EXISTING {event['product_problems']}")
        llm_result, cost = self._llm_categorize(device_name, event_text)
        self.stats['total_cost'] += cost
        return llm_result

    def _llm_categorize(self, device_name: str, event_text: str) -> Tuple[Dict, float]:
        """Use LLM to categorize event."""
        prompt = f"""Carefully review the following Medical Device Report (MDR), which must be filed with the FDA.
        
Medical Device Report Text:
<mdr_text>
Device name: {device_name}
{event_text}
</mdr_text>

Your task is to determine what values to fill out in Form FDA 3500A MedWatch for the entries "Type of Reportable Event" and "Medical Device Problem".

For "Type of Reportable Event", the MDR should be classified into one of the following three categories per FDA 3500A SUPPLEMENT FORM INSTRUCTIONS:
- Death: Check only if the MDR reportable event represents a device-related death.
- Injury: The MDR reportable event represents an adverse event that is life-threatening; results in permanent impairment of a body function or permanent damage to a body structure; or necessitates medical or surgical intervention to preclude permanent impairment of a body function or permanent damage to a body structure.
- Malfunction: Failure of a device to meet its performance specifications or otherwise perform as intended. Performance specifications include all claims made in the labeling for the device. The intended performance of a device refers to the intended use for which the device is labeled or marketed. If neither Death nor Injury are applicable, choose Malfunction.

The FDA guidance document "Medical Device Reporting for Manufacturers" provides the following FAQs to guide the classification of the Event Type:
* What are "MDR Reportable Events"?
  > For manufacturers, "MDR reportable events" are events where the manufacturers become aware of that reasonably suggest that one of their marketed devices may have caused or contributed to a death or serious injury, or has malfunctioned and the malfunction of the device or a similar device that they market would be likely to cause or contribute to a death or serious injury if malfunction were to recur.
* What is meant by "caused or contributed" to a death or serious injury?
  > This means that a death or serious injury was or may have been attributed to a medical device or that a medical device was or may have been a factor in a death or serious injury, including events occurring as a result of [21 CFR 803.3] failure, malfunction, improper or inadequate design, manufacture, labeling, or user error.
* What is device "user error" and why do you want to know about events involving user error?
  > We consider a device "user error" (or "use error") to mean a device-related error or mistake made by the person using the device. The error could be the sole cause of an MDR reportable event, or merely a contributing factor. Such errors often reflect problems with device labeling, the user interface, or other aspects of device design. Thus, FDA believes that these events should be reported in the same way as other adverse events which are caused or contributed to by the device. This is especially important for devices used in non-health care facility settings. If you determine that an event is solely the result of user error with no other performance issue, and there has been no device related death or serious injury, you are not required to submit an MDR report, but you should retain the supporting information in your complaint files.
* What is a "serious injury"?
  > An injury must meet the definition of "serious injury" in 21 CFR 803.3 for an event to be reportable as a serious injury. A "serious injury" is an injury or illness that [21 CFR803.3]: is life threatening, results in permanent impairment of a body function or permanent damage to a body structure, or necessitates medical or surgical intervention to preclude permanent impairment of a body function or permanent damage to a body structure."Permanent" means irreversible impairment or damage to a body structure or function, excluding trivial impairment or damage [21 CFR 803.3]. Note that not all cosmetic damage will be considered trivial. Furthermore, a life-threatening injury meets the definition of serious injury, regardless of whether the threat was "temporary." It should also be noted that a device does not have to malfunction for it to cause or contribute to a serious injury. ven though a device may function properly, it can still cause or contribute to a death or serious injury.

For "Medical Device Problem", assign one or more codes to the following MDR. Instructions from the FDA's MDR Coding Manual:
* The FDA MDR adverse event codes are divided into the following seven categories:
  - "Medical Device Problem": Problems (malfunction, deterioration of function, failure) of medical devices
  - "Medical Device Component": The parts and components which were involved in, or affected by, the medical device adverse event/incident.
  - "Cause Investigation: Type of Investigation": What was investigated and what kind of investigation was conducted to specify the root cause of the adverse event.
  - "Cause Investigation: Investigation Findings": The findings in the specific investigation that are the keys to identify the root cause of the event.
  - "Cause Investigation: Investigation Conclusion": The conclusion regarding the root cause of the reported event.
  - "Health Effects: Clinical Signs and Symptoms or Conditions": The clinical signs and symptoms or conditions of the affected person appearing as a result of the medical device adverse event/incident.
  - "Health Effects: Health Impact": The consequences of the medical device adverse event/incident on the person affected.
* Code Structure: Each set of codes is organized in a tree-like hierarchical structure, where higher-level (closer to the root) codes are more generic, while lower-level (leaf) codes are more specific. For instance, "IMDRF:A01" is higher-level because it has fewer digits while "IMDRF:A010101" is the lowest level with 6 digits. A parent code is often divided into multiple distinct and more-specific child codes, each of which can be considered a member of the set of problems or observations described by the parent code. This allows each set of codes to be intuitively organized in a way that accurately represents the relationship between different but similar codes.
* Reporters should code to the lowest level possible; in other words, they should choose the most specific term(s) available in each category to describe the event or investigation. Reporters may choose more than one code from each set when filing their report, but there is no need to choose both a parent code and one of its children; by definition, the child code is a member or type of the problem or observation represented by its parent, so the child code alone is sufficient.

Your task is to only focus on the category "Medical Device Problem". Carefully go through the following list of medical device problem codes from the FDA and reason through how the codes should be assigned, step-by-step. First consider the highest level codes (IMDRF:AXX with two digits) and then select a more detailed version. The list of FDA codes are:
{self.device_problems_str}

Provide your output in the following JSON format, where "event_type" provides is the assigned label for "Type of Reportable Event" and the "fda_device_problem_codes" is a list of strings with one or more relevant CDRH Preferred Terms (not the IMDRF codes).

Example JSON: {{
    "reasoning": "<DESCRIBE YOUR CHAIN OF THOUGHT HERE>",
    "event_type": "Death", # options: "Death" or "Injury" or "Malfunction"
    "fda_device_problem_codes": ["Lack of Effect"]}}
"""
        try:
            response = self.client.chat.completions.parse(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=MDRClassification
            )
            
            # Calculate cost
            cost = calculate_gpt_cost(model=LLM_MODEL, usage=response.usage)
            
            result = response.choices[0].message.parsed.model_dump()
            print(result)
            self.stats['labeled'] += 1
            
            return result, cost
            
        except Exception as e:
            logging.error(f"LLM categorization failed: {e}")
            return {"fda_device_problem_codes": ['Insufficient Information'], "event_type": "llm_failed"}, 0.0


class AdverseEventsCategorizer:
    """Main class for categorizing adverse events with LLM."""
    
    def __init__(self, device_problems_df, output_file: str):
        self.llm_analyzer = LLMAnalyzer(device_problems_df)
        self.device_problems_df = device_problems_df
        self.output_file = output_file
    
    def _process_events_parallel(self, device_name: str, events: List[Dict], device_number: str) -> List[Dict]:
        """Process events in parallel (for devices with many events)."""
        logging.info(f"Device {device_number}: Processing {len(events)} events in parallel")
        
        # Use a smaller number of workers for event processing to avoid overwhelming APIs
        event_workers = min(NUM_WORKERS_DEVICE, len(events))
        
        def process_single_event(event_data):
            """Process a single event with categorization."""
            device_name, event, event_index = event_data
            
            logging.info(f"Device {device_number}, Event {event_index+1}/{len(events)}: Starting categorization")
            
            event["llm_analysis"] = self.llm_analyzer.categorize_event(device_name, event)
            return event
        
        # Process events in parallel
        with ThreadPoolExecutor(max_workers=event_workers) as executor:
            event_data = [(device_name, event, i) for i, event in enumerate(events)]
            processed_events = list(executor.map(process_single_event, event_data))
        
        return processed_events
    
    def process_device_events(self, device_result: Dict) -> Dict:
        """Process and categorize events for a single device."""
        device_number = device_result.get("device_number")
        device_name = device_result.get("device_name")
        events = device_result.get("adverse_events", [])
        
        if not events:
            logging.info(f"Device {device_number}: No adverse events to process")
            return device_result
        
        logging.info(f"Device {device_number}: Processing {len(events)} adverse events")
        
        # Process events in parallel if there are many events
        device_result["adverse_events"] = self._process_events_parallel(device_name, events, device_number)
        
        return device_result
    
    def categorize_events(self, fda_results: List[Dict], max_workers: int = DEFAULT_MAX_WORKERS) -> List[Dict]:
        """Categorize adverse events for all devices with parallel processing."""
        logging.info(f"Starting LLM categorization for {len(fda_results)} devices")
        
        # Filter devices with adverse events
        devices_with_events = [r for r in fda_results if r.get("has_ae", False)]
        
        if not devices_with_events:
            logging.info("No devices with adverse events to categorize")
            return fda_results
        
        logging.info(f"Found {len(devices_with_events)} devices with adverse events to categorize")
        
        # Define output file path
        # Load existing results if available
        results = []
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                results = json.load(f)
                output_dev_nums = {r['device_number'] for r in results}
                for device in fda_results:
                    if device['device_number'] not in output_dev_nums:
                        results.append(device)
                # Update stats based on existing results
                # Note: Cost information is not stored in existing results, so we only track new costs
                for r in results:
                    if r.get('has_ae'):
                        for event in r.get('adverse_events', []):
                            if 'llm_analysis' in event:
                                self.llm_analyzer.stats['labeled'] += 1
            logging.info(f"Loaded {len(results)} existing results from output file")
            logging.info("Note: Cost tracking only applies to newly processed events in this run")
        else:
            results = fda_results.copy()
        
        # Find devices that need categorization
        categorized_device_numbers = set()
        for r in results:
            if r.get('has_ae', False):
                events = r.get('adverse_events', [])
                has_llm_analyses = ['llm_analysis' in event and 'event_type' in event['llm_analysis'] for event in events]
                if all(has_llm_analyses):
                    categorized_device_numbers.add(r.get('device_number'))
        
        devices_to_categorize = [r for r in results if r.get('has_ae', False) and r.get('device_number') not in categorized_device_numbers]
        print(f"devices_to_categorize {[f['device_number'] for f in devices_to_categorize]}")
        
        if not devices_to_categorize:
            logging.info("All devices with adverse events already categorized")
            logging.info(f"Results saved to {self.output_file}")
            return results
        
        logging.info(f"Need to categorize {len(devices_to_categorize)} devices")
        
        # Process devices in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_device = {
                executor.submit(self.process_device_events, device_result): device_result
                for device_result in devices_to_categorize
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(devices_to_categorize), desc="Categorizing events") as pbar:
                for future in as_completed(future_to_device):
                    device_result = future_to_device[future]
                    try:
                        categorized_device = future.result()
                        
                        # Update the device in results
                        for i, r in enumerate(results):
                            if r.get('device_number') == categorized_device.get('device_number'):
                                results[i] = categorized_device
                                break
                        
                        # Update progress
                        pbar.update(1)
                        
                        # Save progress after every device
                        with open(self.output_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                    except Exception as e:
                        logging.error(f"Error categorizing device {device_result.get('device_number', 'unknown')}: {e}")
                        pbar.update(1)
                        continue
        
        logging.info(f"Results saved to {self.output_file}")
        return results
        
def load_fda_results(file_path: str, device_names_jsonl: str) -> List[Dict]:
    """Load FDA results from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    device_names_results = load_jsonl(device_names_jsonl)
    device_name_dict = {
        res['device_number']: res['metadata']['device_name'] if 'metadata' in res else res['device_number']
        for res in device_names_results
    }
    for data_elem in data:
        if data_elem['device_number'] in device_name_dict:
            data_elem['device_name'] = device_name_dict[data_elem['device_number']]
        else:
            # if device name missing, just put in device number
            data_elem['device_name'] = data_elem['device_number']
    
    logging.info(f"Loaded {len(data)} FDA results from {file_path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="LLM Categorization Script")
    parser.add_argument("--input", required=True,
                       help="Path to JSON file with FDA results")
    parser.add_argument("--device-names-jsonl", required=True,
                       help="Path to JSON file with device names")
    parser.add_argument("--output", required=False, default="output/adverse_events_analysis_results_event_prob_struct.json",
                       help="Path to LLM extraction results")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, 
                       help=f"Maximum number of parallel workers (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--log-file", type=str, default="output/log_categorize_with_llm.txt")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=args.log_file)
    
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Load FDA results
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        logging.error("Please run retrieve_fda_data.py first to generate FDA data")
        sys.exit(1)
    
    fda_results = load_fda_results(args.input, args.device_names_jsonl)

    device_problems_df = pd.read_csv(PROJECT_ROOT / "data/FDA-CDRH_NCIt_Subsets.csv")
    device_problems_df = device_problems_df[device_problems_df['CDRH Subset Name'] == "Medical Device Problem"]
    
    # Run LLM categorization
    categorizer = AdverseEventsCategorizer(device_problems_df, args.output)
    results = categorizer.categorize_events(fda_results, max_workers=args.max_workers)
    
    logging.info(f"LLM categorization completed. Results saved to {OUTPUT_DIR}")
    
    # Print summary
    devices_with_ae = [r for r in results if r.get("has_ae", False)]
    total_events = sum(len(r.get("adverse_events", [])) for r in results)
    
    print(f"Processed {len(results)} devices")
    print(f"Devices with adverse events: {len(devices_with_ae)}")
    print(f"Total adverse events categorized: {total_events}")
    print(f"LLM events labeled: {categorizer.llm_analyzer.stats['labeled']}")
    print(f"Total GPT API cost: ${categorizer.llm_analyzer.stats['total_cost']:.4f}")


if __name__ == "__main__":
    main()
