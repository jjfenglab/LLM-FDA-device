"""
Assigns annotators for MDRs
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.common import *

def format_mdr(mdr_text_list):
    mdr_text = ""
    for text in mdr_text_list:
        if text.startswith('Description of Event or Problem'):
            mdr_text += f"{text}\n\n"
    for text in mdr_text_list:
        if text.startswith('Additional'):
            mdr_text += f"{text}\n\n"
    
    # sanity check
    for text in mdr_text_list:
        if not text.startswith('Additional') and not text.startswith('Description of Event or Problem'):
            raise ValueError()
    
    return mdr_text

def select_extract(extract_dict, order_idx, reverse=False):
    if not reverse:
        if order_idx == 0:
            return extract_dict["llm"]
        else:
            return extract_dict["vendor"]
    else:
        if order_idx == 1:
            return extract_dict["llm"]
        else:
            return extract_dict["vendor"]

def create_annotator_assignments(annotator_dict, mdr_dict, output_file, show_true=True):
    full_list = sorted(annotator_dict["disagree_event_type"] + annotator_dict["disagree_problem_type"] + annotator_dict["death_type"])
    
    rand_order = np.random.choice(2, size=len(full_list), replace=True)
    print(rand_order.tolist())
    annotator_df = pd.DataFrame({
        "report number": full_list,
        "mdr text": [format_mdr(mdr_dict[mdr_num]["text"]) for mdr_num in full_list],
        "event_1": [select_extract(mdr_dict[mdr_num]["event"], order_idx) for order_idx, mdr_num in zip(rand_order, full_list)],
        "event_2": [select_extract(mdr_dict[mdr_num]["event"], order_idx, reverse=True) for order_idx, mdr_num in zip(rand_order, full_list)],
        "problem_1": [select_extract(mdr_dict[mdr_num]["problem"], order_idx) for order_idx, mdr_num in zip(rand_order, full_list)],
        "problem_2": [select_extract(mdr_dict[mdr_num]["problem"], order_idx, reverse=True) for order_idx, mdr_num in zip(rand_order, full_list)],
    })
    annotator_df["event_type"] = ""
    annotator_df["device_problem_codes"] = ""
    if show_true:
        annotator_df["event_1_true"] = ["llm" if order_idx == 0 else "vendor" for order_idx in rand_order]
        annotator_df["event_2_true"] = ["llm" if order_idx == 1 else "vendor" for order_idx in rand_order]
        annotator_df["problem_1_true"] = ["llm" if order_idx == 0 else "vendor" for order_idx in rand_order]
        annotator_df["problem_2_true"] = ["llm" if order_idx == 1 else "vendor" for order_idx in rand_order]
    print(annotator_df.shape)
    annotator_df.to_csv(output_file, index=False)
    print(output_file)

np.random.seed(4)

with open("../analysis_ae_recall/output/adverse_events_analysis_results_event_prob_rebuttal.json", "r") as f:
    llm_extractions = json.load(f)
print("filtered list len", len(llm_extractions))

# get disagreements for event type
disagree_event_dict = {}
disagree_problem_dict = {}
device_count_dict = {}
mdr_to_device_dict = {}
death_dict = {}
tot_counts = 0
tot_aes = 0
for device_dict in llm_extractions:
    device_num = device_dict['device_number']
    for ae_dict in device_dict["adverse_events"]:
        if 'event_type' in ae_dict["llm_analysis"] and len(ae_dict['mdr_texts']) > 0:
            tot_aes += 1
            mdr_to_device_dict[ae_dict['report_number']] = device_num

            llm_event = ae_dict["llm_analysis"]["event_type"]
            vendor_event = ae_dict["event_type"]
            llm_problem = ae_dict["llm_analysis"]["fda_device_problem_codes"]
            vendor_problem = ae_dict["product_problems"]
            contrast_dict = {
                "text": ae_dict['mdr_texts'],
                "problem": {
                    "llm": llm_problem,
                    "vendor": vendor_problem,
                },
                "event": {
                    "llm": llm_event,
                    "vendor": vendor_event,
                }
            }
            if llm_event == "llm_failed":
                raise ValueError("LLM FAILED!")

            if (llm_event == "Death" and vendor_event != "Death") or (llm_event != "Death" and vendor_event == "Death"):
                print("LLM", llm_event, "VENDOR", vendor_event, ae_dict["report_number"])
                device_count_dict[device_num] = 1 + device_count_dict.get(device_num, 0)
                death_dict[ae_dict["report_number"]] = contrast_dict
            elif (llm_event == "Death" and vendor_event == "Death"):
                print("both death")
            else:
                if llm_event != vendor_event:
                    disagree_event_dict[ae_dict["report_number"]] = contrast_dict
                    device_count_dict[device_num] = 1 + device_count_dict.get(device_num, 0)
                    tot_counts += 1
                if len(set(llm_problem) | set(vendor_problem)) > len(set(llm_problem)):
                    if ae_dict["report_number"] not in disagree_problem_dict:
                        disagree_problem_dict[ae_dict["report_number"]] = contrast_dict
                        device_count_dict[device_num] = 1 + device_count_dict.get(device_num, 0)
                        tot_counts += 1

print(f"tot_aes {tot_aes}")
disagree_event_types = list(disagree_event_dict.keys())
event_type_prob = np.array([tot_counts/device_count_dict[mdr_to_device_dict[mdr_num]] for mdr_num in disagree_event_types])
disagree_problem_types = list(disagree_problem_dict.keys())
problem_type_prob = np.array([tot_counts/device_count_dict[mdr_to_device_dict[mdr_num]] for mdr_num in disagree_problem_types])
death_mdr_nums = list(death_dict.keys())
print("death_mdr_nums", len(death_mdr_nums))

event_type_idx = np.random.choice(
    len(disagree_event_types),
    p=event_type_prob/np.sum(event_type_prob),
    size=3 * 15,
    replace=False)
problem_type_idx = np.random.choice(
    len(disagree_problem_types),
    p=problem_type_prob/np.sum(problem_type_prob),
    size=3 * 15,
    replace=False)

disagree_event_types = [disagree_event_types[i] for i in event_type_idx]
disagree_problem_types = [disagree_problem_types[i] for i in problem_type_idx]

mdr_dict = disagree_event_dict | disagree_problem_dict | death_dict

annotator1 = {
    "disagree_event_type": disagree_event_types[:15],
    "disagree_problem_type": disagree_problem_types[:15],
    "death_type": death_mdr_nums,
}

annotator2 = {
    "disagree_event_type": disagree_event_types[15:30],
    "disagree_problem_type": disagree_problem_types[15:30],
    "death_type": death_mdr_nums,
}

annotator3 = {
    "disagree_event_type": disagree_event_types[30:],
    "disagree_problem_type": disagree_problem_types[30:],
    "death_type": death_mdr_nums,
}

annotator4 = {
    "disagree_event_type": disagree_event_types[30:],
    "disagree_problem_type": disagree_problem_types[30:],
    "death_type": death_mdr_nums,
}


create_annotator_assignments(annotator1, mdr_dict, "output/mdr_annotator1.csv")
create_annotator_assignments(annotator2, mdr_dict, "output/mdr_annotator2.csv")
create_annotator_assignments(annotator3, mdr_dict, "output/mdr_annotator3.csv")
create_annotator_assignments(annotator4, mdr_dict, "output/mdr_annotator4.csv")
