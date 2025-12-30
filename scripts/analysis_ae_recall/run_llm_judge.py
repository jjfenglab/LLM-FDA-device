#!/usr/bin/env python3
"""
LLM Judge Validation Script
Validates already categorized adverse events using LLM-as-judge.
"""
from dotenv import load_dotenv
import argparse
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from tqdm import tqdm

import numpy as np
import pandas as pd

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Constants
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LLM_MODEL = "gpt-4.1"

# Parallel processing settings
DEFAULT_MAX_WORKERS = 4

from typing import Optional
from pydantic import BaseModel, Field

class MDRJudgeResult(BaseModel):
    event_type_reasoning: Optional[str] = None
    event_type_judge_result: int
    device_problem_reasoning: Optional[str] = None
    device_problem_judge_result: int

    @staticmethod
    def get_json_template():
        return """{
    "event_type_reasoning": <PROVIDE YOUR REASONING HERE>,
    "event_type_judge_result": <1 or 2 or 0>,
    "device_problem_reasoning": <PROVIDE YOUR REASONING HERE>,
    "device_problem_judge_result": <1 or 2 or 0>
}"""

    @staticmethod
    def get_json_simple_template():
        return """{
    "event_type_judge_result": <1 or 2 or 0>,
    "device_problem_judge_result": <1 or 2 or 0>
}"""

class MDRJudgeResultProbOnly(BaseModel):
    device_problem_reasoning: Optional[str] = None
    device_problem_judge_result: int

    @staticmethod
    def get_json_template():
        return """{
    "device_problem_reasoning": <PROVIDE YOUR REASONING HERE>,
    "device_problem_judge_result": <1 or 2 or 0>
}"""

    @staticmethod
    def get_json_simple_template():
        return """{
    "device_problem_judge_result": <1 or 2 or 0>
}"""

class MDRJudgeResultEventOnly(BaseModel):
    event_type_reasoning: Optional[str] = None
    event_type_judge_result: int

    @staticmethod
    def get_json_template():
        return """{
    "event_type_reasoning": <PROVIDE YOUR REASONING HERE>,
    "event_type_judge_result": <1 or 2 or 0>
}"""

    @staticmethod
    def get_json_simple_template():
        return """{
    "event_type_judge_result": <1 or 2 or 0>
}"""

class LLMJudge(ABC):
    """Abstract base class for LLM judge for validating adverse event categorizations."""
    
    def __init__(self, device_problems_df):
        self.stats = {'validated': 0, 'agreements': 0, 'disagreements': 0}
        self.device_problems_str = device_problems_df[['CDRH Preferred Term', 'IMDRF Code']].to_csv(index=False)
    
    @abstractmethod
    def _make_api_call(self, prompt: str, response_format):
        """Make API call to the specific LLM provider."""
        pass
    
    def validate_categorization(self, device_name: str, event: Dict) -> Dict:
        """Validate categorization using LLM-as-judge."""
        event_text = " ".join(event.get("mdr_texts", []))
        gen_event_types = {
            "llm": event['llm_analysis']['event_type'],
            "vendor": event.get('event_type', 'None')
        }
        event_type_same = gen_event_types['llm'] == gen_event_types['vendor']
        gen_codes = {
            "llm": ", ".join([c for c in event['llm_analysis']['fda_device_problem_codes']]) if len(event['llm_analysis']['fda_device_problem_codes']) else "no codes",
            "vendor": ", ".join([v for v in event['product_problems']]) if len(event['product_problems']) else "no codes"
        }
        gen_codes_same = gen_codes['llm'] == gen_codes['vendor']
        rand_idxs = np.random.choice(["llm", "vendor"], size=2, replace=False)
        
        if not event_type_same and not gen_codes_same:
            prompt = f"""You are validating categorizations assigned to Medical Device Reports (MDRs) that are to be submitted to the FDA.

Medical Device Report Text:
<mdr_text>
Device name: {device_name}
{event_text}
</mdr_text>

Review the following instructions from the FDA for selecting values for "Type of Reportable Event" and "Medical Device Problem" in Form FDA 3500A MedWatch.

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

Here are the proposals by two coders for "Type of Reportable Event":
Option 1: {gen_event_types[rand_idxs[0]]}
Option 2: {gen_event_types[rand_idxs[1]]}

Now review the following instructions from the FDA for selecting values for "Medical Device Problem" per the FDA's MDR Coding Manual:
* The FDA MDR adverse event codes are divided into the following seven categories:
  - "Medical Device Problem": Problems (malfunction, deterioration of function, failure) of medical devices
  - "Medical Device Component": The parts and components which were involved in, or affected by, the medical device adverse event/incident.
  - "Cause Investigation: Type of Investigation": What was investigated and what kind of investigation was conducted to specify the root cause of the adverse event.
  - "Cause Investigation: Investigation Findings": The findings in the specific investigation that are the keys to identify the root cause of the event.
  - "Cause Investigation: Investigation Conclusion": The conclusion regarding the root cause of the reported event.
  - "Health Effects: Clinical Signs and Symptoms or Conditions": The clinical signs and symptoms or conditions of the affected person appearing as a result of the medical device adverse event/incident.
  - "Health Effects: Health Impact": The consequences of the medical device adverse event/incident on the person affected.
* Code Structure: Each set of codes is organized in a tree-like hierarchical structure, where higher-level (closer to the root) codes are more generic, while lower-level (leaf) codes are more specific. A parent code is often divided into multiple distinct and more-specific child codes, each of which can be considered a member of the set of problems or observations described by the parent code. This allows each set of codes to be intuitively organized in a way that accurately represents the relationship between different but similar codes.
* Reporters should code to the lowest level possible; in other words, they should choose the most specific term(s) available in each category to describe the event or investigation. Reporters may choose more than one code from each set when filing their report, but there is no need to choose both a parent code and one of its children; by definition, the child code is a member or type of the problem or observation represented by its parent, so the child code alone is sufficient.

Here are the proposals by two coders for the specific category of "Medical Device Problem":
Option 1: {gen_codes[rand_idxs[0]]}
Option 2: {gen_codes[rand_idxs[1]]}

Based on the information above, which assigned values are more appropriate for "Type of Reportable Event" and "Medical Device Problem"? Answer 1 if option 1 is better. Answer 2 if option 2 is better. Answer 0 if both options are equally good (or the two proposed options are the same) and it should be considered a tie. Assign ties minimally.

Please respond with a JSON object:
{{TEMPLATE}}
"""
            response_format = MDRJudgeResult
        elif not gen_codes_same:
            prompt = f"""You are validating categorizations assigned to Medical Device Reports (MDRs) that are to be submitted to the FDA.

Medical Device Report Text:
<mdr_text>
Device name: {device_name}
{event_text}
</mdr_text>

Now review the following instructions from the FDA for selecting values for "Medical Device Problem" per the FDA's MDR Coding Manual:
* The FDA MDR adverse event codes are divided into the following seven categories:
  - "Medical Device Problem": Problems (malfunction, deterioration of function, failure) of medical devices
  - "Medical Device Component": The parts and components which were involved in, or affected by, the medical device adverse event/incident.
  - "Cause Investigation: Type of Investigation": What was investigated and what kind of investigation was conducted to specify the root cause of the adverse event.
  - "Cause Investigation: Investigation Findings": The findings in the specific investigation that are the keys to identify the root cause of the event.
  - "Cause Investigation: Investigation Conclusion": The conclusion regarding the root cause of the reported event.
  - "Health Effects: Clinical Signs and Symptoms or Conditions": The clinical signs and symptoms or conditions of the affected person appearing as a result of the medical device adverse event/incident.
  - "Health Effects: Health Impact": The consequences of the medical device adverse event/incident on the person affected.
* Code Structure: Each set of codes is organized in a tree-like hierarchical structure, where higher-level (closer to the root) codes are more generic, while lower-level (leaf) codes are more specific. A parent code is often divided into multiple distinct and more-specific child codes, each of which can be considered a member of the set of problems or observations described by the parent code. This allows each set of codes to be intuitively organized in a way that accurately represents the relationship between different but similar codes.
* Reporters should code to the lowest level possible; in other words, they should choose the most specific term(s) available in each category to describe the event or investigation. Reporters may choose more than one code from each set when filing their report, but there is no need to choose both a parent code and one of its children; by definition, the child code is a member or type of the problem or observation represented by its parent, so the child code alone is sufficient.

Here are the proposals by two coders for the specific category of "Medical Device Problem":
Option 1: {gen_codes[rand_idxs[0]]}
Option 2: {gen_codes[rand_idxs[1]]}

Based on the information above, which assigned values are more appropriate for "Medical Device Problem"? Answer 1 if option 1 is better. Answer 2 if option 2 is better. Answer 0 if both options are equally good (or the two proposed options are the same) and it should be considered a tie. Assign ties minimally.

Please respond with a JSON object:
{{TEMPLATE}}"""
            response_format = MDRJudgeResultProbOnly
        elif not event_type_same:
            prompt = f"""You are validating categorizations assigned to Medical Device Reports (MDRs) that are to be submitted to the FDA.

Medical Device Report Text:
<mdr_text>
Device name: {device_name}
{event_text}
</mdr_text>

Review the following instructions from the FDA for selecting values for "Type of Reportable Event" and "Medical Device Problem" in Form FDA 3500A MedWatch.

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

Here are the proposals by two coders for "Type of Reportable Event":
Option 1: {gen_event_types[rand_idxs[0]]}
Option 2: {gen_event_types[rand_idxs[1]]}

Based on the information above, which assigned values are more appropriate for "Type of Reportable Event"? Answer 1 if option 1 is better. Answer 2 if option 2 is better. Answer 0 if both options are equally good (or the two proposed options are the same) and it should be considered a tie. Assign ties minimally.

Please respond with a JSON object:
{{TEMPLATE}}
"""
            response_format = MDRJudgeResultEventOnly
        else:
            prompt = None
        
        result = {"device_problem_codes": {}, "event_type": {}}
        if prompt is not None:
            api_response = self._make_api_call(prompt, response_format)
            logging.info(f"api response {event['report_number']} {api_response}")
            result |= api_response

        if gen_codes_same or result["device_problem_judge_result"] == 0:
            result["device_problem_codes"]["judge_result_converted"] = "tie"
        else:
            result["device_problem_codes"]["judge_result_converted"] = rand_idxs[result["device_problem_judge_result"] - 1]
        
        if event_type_same or result["event_type_judge_result"] == 0:
            result["event_type"]["judge_result_converted"] = "tie"
        else:
            result["event_type"]["judge_result_converted"] = rand_idxs[result["event_type_judge_result"] - 1]
        self.stats['validated'] += 1
        print("VALIDATED")
        
        return result


class GPTJudge(LLMJudge):
    """GPT-based judge using OpenAI API."""
    
    def __init__(self, device_problems_df):
        super().__init__(device_problems_df)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def _make_api_call(self, prompt: str, response_format):
        """Make API call to OpenAI GPT."""
        response = self.client.chat.completions.parse(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format
        )
        return response.choices[0].message.parsed.model_dump()


class ClaudeJudge(LLMJudge):
    """Claude-based judge using Anthropic API via LangChain."""
    
    def __init__(self, device_problems_df):
        super().__init__(device_problems_df)
        self.client = ChatAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model="claude-sonnet-4-5-20250929",
            max_retries=2,
            temperature=1,
        )
        self.backup_client = ChatAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model="claude-sonnet-4-20250514",
            max_retries=2,
            temperature=1,
        )
    
    def _make_api_call(self, prompt: str, response_format):
        """Make API call to Claude."""
        prompt_attempt1 = prompt.replace("{TEMPLATE}", response_format.get_json_template())
        logging.info(prompt_attempt1)
        llm_with_tools = self.client.with_structured_output(response_format, include_raw=True)
        response = llm_with_tools.invoke(prompt_attempt1)

        # Check if parsing was successful
        if response.get("parsing_error"):
            logging.error(f"Claude parsing error: {response['parsing_error']}")
            logging.error(f"Raw response: {response.get('raw')}")

            # attempt 2 -- use backup client and simpler prompt
            prompt_attempt2 = prompt.replace("{TEMPLATE}", response_format.get_json_simple_template())
            logging.info(prompt_attempt2)
            llm_with_tools = self.backup_client.with_structured_output(response_format, include_raw=True)
            response = llm_with_tools.invoke(prompt_attempt2)
            if response.get("parsing_error"):
                logging.error(f"Claude parsing error: {response['parsing_error']}")
                logging.error(f"Raw response: {response.get('raw')}")
                raise ValueError("FAILED ATTEMPT 2")
            else:
                return response["parsed"].model_dump()
        else:
            return response["parsed"].model_dump()


class AdverseEventsValidator:
    """Main class for validating categorized adverse events."""
    
    def __init__(self, llm_judge):
        self.llm_judge = llm_judge
    
    def _validate_events_parallel(self, device_name: str, events: List[Dict], device_number: str) -> List[Dict]:
        """Validate events in parallel (for devices with many events)."""
        logging.info(f"Device {device_number}: Validating {len(events)} events in parallel")
        
        # Use a smaller number of workers for event processing to avoid overwhelming APIs
        event_workers = min(4, len(events))
        
        def validate_single_event(event_data):
            """Validate a single event."""
            device_name, event, event_index = event_data
            
            if "llm_analysis" not in event:
                logging.warning(f"Device {device_number}, Event {event_index+1} {event['report_number']}: No categorization found, skipping validation")
                return event
            if "validation" in event:
                return event
            if len(event["mdr_texts"]) == 0:
                logging.warning(f"Device {device_number}, Event {event_index+1} {event['report_number']}: No text found")
                return event
            
            event["validation"] = self.llm_judge.validate_categorization(device_name, event)
            
            return event
        
        # Process events in parallel
        with ThreadPoolExecutor(max_workers=event_workers) as executor:
            event_data = [(device_name, event, i) for i, event in enumerate(events)]
            processed_events = list(executor.map(validate_single_event, event_data))
        
        return processed_events
    
    def validate_device_events(self, device_result: Dict) -> Dict:
        """Validate categorized events for a single device."""
        device_name = device_result.get("device_name")
        device_number = device_result.get("device_number")
        events = device_result.get("adverse_events", [])
        
        if not events:
            logging.info(f"Device {device_number}: No adverse events to validate")
            return device_result
        
        # Check if events have categorization
        categorized_events = [e for e in events if "llm_analysis" in e]
        if not categorized_events:
            logging.info(f"Device {device_number}: No categorized events to validate")
            return device_result
        
        logging.info(f"Device {device_number}: Validating {len(categorized_events)} categorized adverse events")
        
        # Validate events in parallel if there are many events
        device_result["adverse_events"] = self._validate_events_parallel(device_name, events, device_number)
        
        return device_result
    
    def validate_events(self, categorized_results: List[Dict], output_file: str, max_workers: int = DEFAULT_MAX_WORKERS) -> List[Dict]:
        """Validate categorized adverse events for all devices with parallel processing."""
        logging.info(f"Starting LLM validation for {len(categorized_results)} devices")
        
        # Filter devices with categorized adverse events
        devices_with_categorized_events = []
        for r in categorized_results:
            if r.get("has_ae", False):
                events = r.get("adverse_events", [])
                categorized_events = [e for e in events if "llm_analysis" in e]
                if categorized_events:
                    devices_with_categorized_events.append(r)
        
        if not devices_with_categorized_events:
            logging.info("No devices with categorized adverse events to validate")
            return categorized_results
        
        logging.info(f"Found {len(devices_with_categorized_events)} devices with categorized adverse events to validate")
        
        # Load existing results if available
        results = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_results = json.load(f)
                    existing_device_numbers = {d['device_number'] for d in existing_results}
                    results = existing_results
                    for input_device in categorized_results:
                        if input_device['device_number'] not in existing_device_numbers:
                            results.append(input_device)

                logging.info(f"Loaded {len(results)} existing results from output file")
            except Exception as e:
                logging.warning(f"Could not load existing results: {e}")
        
        # If no existing results, start with categorized results
        if not results:
            results = categorized_results.copy()
        
        # Find devices that need validation
        validated_device_numbers = set()
        for r in results:
            if r.get('has_ae', False):
                events = r.get('adverse_events', [])
                if events and all('validation' in ev for ev in events):
                    validated_device_numbers.add(r.get('device_number'))
        
        devices_to_validate = [r for r in results if r.get('has_ae', False) and r.get('device_number') not in validated_device_numbers]
        
        if not devices_to_validate:
            logging.info("All devices with categorized adverse events already validated")
            return results
        
        logging.info(f"Need to validate {len(devices_to_validate)} devices")
        
        # Process devices in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_device = {
                executor.submit(self.validate_device_events, device_result): device_result
                for device_result in devices_to_validate
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(devices_to_validate), desc="Validating events") as pbar:
                for future in as_completed(future_to_device):
                    device_result = future_to_device[future]
                    try:
                        validated_device = future.result()
                        
                        # Update the device in results
                        for i, r in enumerate(results):
                            if r.get('device_number') == validated_device.get('device_number'):
                                results[i] = validated_device
                                break
                        
                        # Update progress
                        pbar.update(1)
                        
                        # Save progress after every device
                        with open(output_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                    except Exception as e:
                        logging.error(f"Error validating device {device_result.get('device_number', 'unknown')}: {e}")
                        pbar.update(1)
                        continue
        
        return results
    
    
def load_categorized_results(file_path: str) -> List[Dict]:
    """Load categorized results from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Loaded {len(data)} categorized results from {file_path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="LLM Judge Validation Script")
    parser.add_argument("--input", required=False, default="output/adverse_events_analysis_results_event_prob_struct.json",
                       help="Path to JSON file with categorized results")
    parser.add_argument("--output-file", required=False, default="output/validated_adverse_events_results_event_prob_struct.json",
                       help="Path to JSON file with LLM judge results")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, 
                       help=f"Maximum number of parallel workers (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--llm-provider", choices=['openai', 'claude'],
                       help="LLM provider to use (default: openai)")
    parser.add_argument("--log-file", type=str, default="output/log_llm_judge.txt")
    
    args = parser.parse_args()
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=args.log_file)
    load_dotenv()
    
    # Validate API keys based on provider
    if args.llm_provider == 'openai':
        if not os.environ.get("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY environment variable not set")
            sys.exit(1)
    elif args.llm_provider == 'claude':
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logging.error("ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)
    
    # Load categorized results
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        logging.error("Please run categorize_with_llm.py first to generate categorized data")
        sys.exit(1)
    
    categorized_results = load_categorized_results(args.input)
    
    device_problems_df = pd.read_csv(PROJECT_ROOT / "data/FDA-CDRH_NCIt_Subsets.csv")
    device_problems_df = device_problems_df[device_problems_df['CDRH Subset Name'] == "Medical Device Problem"]

    # Create appropriate judge based on provider
    if args.llm_provider == 'openai':
        llm_judge = GPTJudge(device_problems_df)
    elif args.llm_provider == 'claude':
        llm_judge = ClaudeJudge(device_problems_df)
    
    validator = AdverseEventsValidator(llm_judge)
    results = validator.validate_events(categorized_results, output_file=args.output_file, max_workers=args.max_workers)
    
    logging.info(f"LLM validation completed. Results saved to {OUTPUT_DIR}")
    
    # Print summary
    total_events = sum(len(r.get("adverse_events", [])) for r in results)
    validated_events = sum(1 for r in results if r.get("has_ae", False) 
                          for event in r.get("adverse_events", []) if "validation" in event)
    for r in results:
        for event in r.get("adverse_events", []):
            if 'validation' not in event:
                print(event)
                
    print(f"Processed {len(results)} devices")
    print(f"Total adverse events: {total_events}")
    print(f"Events validated: {validated_events}")
    

if __name__ == "__main__":
    main()