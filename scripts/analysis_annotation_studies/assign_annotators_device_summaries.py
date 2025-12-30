"""
Assigns annotators for the device summaries
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.common import *

def create_annotator_assignments(annotator_dict, output_file):
    print(annotator_dict)
    full_list = annotator_dict["prospective"] + annotator_dict["multisite"] + annotator_dict["neither"]
    full_list = pd.DataFrame({
        "device number": sorted(full_list)
    })
    full_list["is_prospective"] = ""
    full_list["num_sites"] = ""
    full_list["human_device_team_testing"] = ""
    full_list["has_clinical_testing"] = ""
    full_list["intended_use_and_clinical_applications"] = ""
    full_list["operational_and_workflow_change"] = ""
    full_list["algorithm_changes"] = ""
    full_list["software_feature_changes"] = ""
    full_list["hardware_changes"] = ""
    full_list["body_part_changes"] = ""
    full_list.to_csv(output_file, index=False)
    print(output_file)

np.random.seed(4)

annot_exists = set(pd.read_csv("../../data/raw/validation/zou_clinical_data.csv")["approval_number"].to_list())
llm_extractions = load_jsonl("../analysis_validation/output/aiml_devices_validation_results.jsonl")
llm_extractions = [device_llm for device_llm in llm_extractions if device_llm["primary_predicate"]]
print("filtered list len", len(llm_extractions))

device_nums = [device_llm["device_number"].lower() for device_llm in llm_extractions]
prospectives = set([device_llm["device_number"].lower() for device_llm in llm_extractions if device_llm["is_prospective"]])
prospective_prevalence = np.mean([device_llm["is_prospective"] for device_llm in llm_extractions])
multisites = set([device_llm["device_number"].lower() for device_llm in llm_extractions if device_llm["is_multisite"]])
multisite_prevalence = np.mean([device_llm["is_multisite"] and not device_llm["is_prospective"] for device_llm in llm_extractions])
neithers = set([device_llm["device_number"].lower() for device_llm in llm_extractions if not device_llm["is_prospective"] and not device_llm["is_multisite"]])
neither_prevalence = np.mean([not device_llm["is_prospective"] and not device_llm["is_multisite"] for device_llm in llm_extractions])

print("num prospective", len(prospectives))
print(f"prevalence prospective {prospective_prevalence:.2f} multisite {multisite_prevalence:.2f} neither {neither_prevalence:.2f}")

all_idxs = np.arange(len(llm_extractions))
mapping_arr = np.array(["n"] * len(llm_extractions), dtype=str)
for i, llm_device_dict in enumerate(llm_extractions):
    if llm_device_dict["is_prospective"]:
        mapping_arr[i] = "p"
    elif llm_device_dict["is_multisite"]:
        mapping_arr[i] = "m"

prospective_idxs = [device_nums[i] for i in np.random.choice(all_idxs[mapping_arr == "p"], size=30, replace=False)]
multisite_idxs = [device_nums[i] for i in np.random.choice(all_idxs[mapping_arr == "m"], size=30, replace=False)]
neither_idxs = [device_nums[i] for i in np.random.choice(all_idxs[mapping_arr == "n"], size=21, replace=False)]

assert all([idx in prospectives for idx in prospective_idxs])
assert all([idx in multisites for idx in multisite_idxs])
assert all([idx in neithers for idx in neither_idxs])

print("prospective intersection", set(prospective_idxs).intersection(annot_exists))
print("multisite intersection", set(multisite_idxs).intersection(annot_exists))

annotator1 = {
    "prospective": prospective_idxs[:9],
    "multisite": multisite_idxs[:8],
    "neither": neither_idxs[:7],
}

annotator2 = {
    "prospective": prospective_idxs[9:17],
    "multisite": multisite_idxs[8:17],
    "neither": neither_idxs[7:14],
}

annotator3 = {
    "prospective": prospective_idxs[17:25],
    "multisite": multisite_idxs[17:24],
    "neither": neither_idxs[14:],
}

annotator4 = {
    "prospective": prospective_idxs[24:],
    "multisite": multisite_idxs[21:],
    "neither": neither_idxs[14:],
}

create_annotator_assignments(annotator1, "output/annotator1.csv")
create_annotator_assignments(annotator2, "output/annotator2.csv")
create_annotator_assignments(annotator3, "output/annotator3.csv")
create_annotator_assignments(annotator4, "output/annotator4.csv")
