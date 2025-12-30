import sys
import argparse
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from scipy import stats

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.common import load_jsonl, wald_confidence_interval, wald_confidence_interval_cts, format_ci_string

def load_llm_extractions(llm_annotations_paths):
    """Load LLM extractions"""
    extraction_dict = {}
    for llm_annotations_path in llm_annotations_paths:
        llm_extractions = load_jsonl(llm_annotations_path)
        
        for llm_device in llm_extractions:
            device_num = llm_device["device_number"].lower()
            if 'num_sites' in llm_device and llm_device['num_sites'] is None:
                llm_device['num_sites'] = 1
            key_in_dict = device_num in extraction_dict.keys()
            if key_in_dict:
                extraction_dict[device_num] |= llm_device
            else:
                extraction_dict[device_num] = llm_device
        
            
    return extraction_dict

def load_annotations(annotation_files):
    """Load all annotator CSV files and merge annotations"""
    all_annotations = []
    for file in annotation_files:
        df = pd.read_csv(file)
        df["annotator"] = file.split("/")[-1].split("_")[1].split(".")[0]  # Extract annotator name
        # change definition of annotations for num_sites to "min num sites"
        df.loc[df.num_sites == 0, "num_sites"] = 1
        df["is_multisite"] = (df["num_sites"] > 1).astype(int)
        df['algorithm_or_software_feature_changes'] = df['algorithm_changes'] | df['software_feature_changes']
        all_annotations.append(df)
    
    return pd.concat(all_annotations, ignore_index=True)

def calculate_sampling_weights(llm_extractions, annotated_device_nums):
    """Calculate sampling weights for reweighting to original distribution"""
    llm_extractions = {k:v for k,v in llm_extractions.items()} # if v["primary_predicate"]}
    
    total_devices = len(llm_extractions)
    num_annotated_prospective = 0
    num_annotated_multisite = 0
    num_annotated_neither = 0
    for device_num in annotated_device_nums:
        is_prospective = llm_extractions[device_num]['is_prospective']
        is_multisite = llm_extractions[device_num]['is_multisite']
        if is_prospective:
            num_annotated_prospective += is_prospective
        elif is_multisite:
            num_annotated_multisite += is_multisite
        else:
            num_annotated_neither += 1
    
    prospective_prevalence = np.mean([device["is_prospective"] for device in llm_extractions.values() if 'is_prospective' in device])
    multisite_prevalence = np.mean([device["is_multisite"] and not device["is_prospective"] for device in llm_extractions.values() if 'is_prospective' in device and 'is_multisite' in device])
    neither_prevalence = np.mean([not device["is_prospective"] and not device["is_multisite"] for device in llm_extractions.values() if 'is_prospective' in device and 'is_multisite' in device])
    
    print(f"num prospect {num_annotated_prospective}, num multisite {num_annotated_multisite}, num neither {num_annotated_neither}")
    return {
        "prospective": (total_devices / num_annotated_prospective) * prospective_prevalence,
        "multisite": (total_devices / num_annotated_multisite) * multisite_prevalence,
        "neither": (total_devices / num_annotated_neither) * neither_prevalence,
        "total_devices": total_devices
    }


def calculate_attribute_prevalence(llm_extractions, all_attrs):
    """Calculate prevalence of each attribute across all LLM extractions"""
    results = []

    for attr in all_attrs:
        # Count devices with this attribute
        count_positive = 0
        total_devices = 0

        for device_data in llm_extractions.values():
            if attr in device_data:
                total_devices += 1
                if device_data[attr]:
                    count_positive += 1

        if total_devices > 0:
            prevalence_pct = count_positive / total_devices
            results.append({
                'Attribute': attr,
                'Prevalence': f"{prevalence_pct:.2f}",
                'Count': f"{count_positive}/{total_devices}"
            })

    return pd.DataFrame(results)

def get_device_strata(matched_df):
    """Assign each device to its stratum and corresponding weight"""
    matched_df["stratum"] = "neither"
    matched_df.loc[matched_df["pred_is_prospective"] == 1, "stratum"] = "prospective"
    matched_df.loc[(matched_df["pred_is_multisite"] == 1) & (matched_df["pred_is_prospective"] == 0), "stratum"] = "multisite"
    return matched_df

def calculate_metrics(y_true, y_pred, weights=None, is_numeric=False):
    """Calculate is_same, TPV, FPR, PPV, and NPV with optional weighting. For numeric attributes, also calculate MAE."""
    if weights is None:
        weights = np.ones(len(y_true))

    is_same = np.sum(weights * (y_true == y_pred))
    tp = np.sum(weights * (y_true == 1) * (y_pred == 1))
    tn = np.sum(weights * (y_true == 0) * (y_pred == 0))
    fp = np.sum(weights * (y_true == 0) * (y_pred == 1))
    fn = np.sum(weights * (y_true == 1) * (y_pred == 0))

    total = tp + tn + fp + fn
    pos_true = tp + fn
    neg_true = tn + fp
    pos_pred = tp + fp
    neg_pred = tn + fn

    # Calculate effective sample size for confidence intervals
    effective_n = (np.sum(weights) ** 2) / np.sum(weights ** 2)

    is_same = is_same/np.sum(weights)
    tpr = tp / pos_true if pos_true > 0 else 0  # sensitivity
    fpr = fp / neg_true if neg_true > 0 else 0  # 1 - specificity
    ppv = tp / pos_pred if pos_pred > 0 else 0  # positive predictive value
    npv = tn / neg_pred if neg_pred > 0 else 0  # negative predictive value
    # print("ppv", tp, pos_pred)
    # print("npv", tn, neg_pred)

    # Calculate confidence intervals
    _, is_same_lower, is_same_upper = wald_confidence_interval(is_same * effective_n, effective_n)
    _, tpr_lower, tpr_upper = wald_confidence_interval(tp, pos_true) if pos_true > 0 else (0, 0, 0)
    _, fpr_lower, fpr_upper = wald_confidence_interval(fp, neg_true) if neg_true > 0 else (0, 0, 0)
    _, ppv_lower, ppv_upper = wald_confidence_interval(tp, pos_pred) if pos_pred > 0 else (0, 0, 0)
    _, npv_lower, npv_upper = wald_confidence_interval(tn, neg_pred) if neg_pred > 0 else (0, 0, 0)

    result = {
        "is_same": is_same,
        "is_same_ci": format_ci_string(is_same, is_same_lower, is_same_upper, is_same * effective_n, effective_n),
        "tpr_ci": format_ci_string(tpr, tpr_lower, tpr_upper, tp, pos_true),
        "fpr_ci": format_ci_string(fpr, fpr_lower, fpr_upper, fp, neg_true),
        "ppv_ci": format_ci_string(ppv, ppv_lower, ppv_upper, tp, pos_pred),
        "npv_ci": format_ci_string(npv, npv_lower, npv_upper, tn, neg_pred),
    }

    # Add MAE for numeric attributes
    if is_numeric:
        abs_errors = weights * np.abs(y_true - y_pred)
        result["mae_ci"] = format_ci_string(*wald_confidence_interval_cts(abs_errors))

    return result

def evaluate_all_attributes(matched_df, sampling_weights, all_attrs):
    """Evaluate all boolean attributes with both weighted and unweighted metrics"""
    
    # Map stratum to weights
    weight_mapping = {
        "prospective": sampling_weights["prospective"],
        "multisite": sampling_weights["multisite"], 
        "neither": sampling_weights["neither"]
    }
    matched_df["weight"] = matched_df["stratum"].map(weight_mapping)
    print(matched_df.columns)
    
    weighted_results = []
    all_true_vals = []
    all_pred_vals = []
    all_weights = []
    for attr in all_attrs:
        print("ATTRIBUTE", attr)
        if f"true_{attr}" in matched_df.columns and f"pred_{attr}" in matched_df.columns:
            y_true = matched_df[f"true_{attr}"].values
            y_pred = matched_df[f"pred_{attr}"].values
            weights = matched_df["weight"].values
            all_true_vals.append(y_true)
            all_pred_vals.append(y_pred)
            all_weights.append(weights)
            print(matched_df[["device_number", f"true_{attr}", f"pred_{attr}"]])

            is_numeric = (attr == "num_sites")
            weighted_metrics = calculate_metrics(y_true, y_pred, weights, is_numeric=is_numeric)
            weighted_results.append({
                "attribute": attr,
                **weighted_metrics
            })
        
    weighted_metrics = calculate_metrics(
        np.concatenate(all_true_vals),
        np.concatenate(all_pred_vals),
        np.concatenate(all_weights))
    weighted_results.append({
        "attribute": "all",
        **weighted_metrics
    })
    
    return pd.DataFrame(weighted_results)

def get_agreement_rates(df, all_attrs):
    df3 = df[df.annotator == 'jean']
    df4 = df[df.annotator == 'adarsh']
    inter_df = df3.merge(df4, on="device number")
    all_attr_agreements = []
    agree_df = []
    for attr in all_attrs:
        print(inter_df[[f'{attr}_x', f'{attr}_y']])
        attr_agreements = inter_df[f'{attr}_x'] == inter_df[f'{attr}_y']
        all_attr_agreements.append(attr_agreements)
        attr_agreement_mean = np.mean(attr_agreements)
        agree_df.append({"attr": attr, "agreement": attr_agreement_mean})
    all_attr_agreements = np.concatenate(all_attr_agreements)
    all_agreement_ci = wald_confidence_interval(all_attr_agreements.sum(), all_attr_agreements.size)
    print(f"all_agreement_ci {format_ci_string(all_agreement_ci[0], all_agreement_ci[1], all_agreement_ci[2], sig_figs=3)}")
    return pd.DataFrame(agree_df)

def main():
    parser = argparse.ArgumentParser(description="Summarize annotation results and calculate performance metrics")
    parser.add_argument("--all-attrs", nargs="+", default=[
            "num_sites", "is_prospective", "is_multisite", "human_device_team_testing", 
            "has_clinical_testing", "intended_use_and_clinical_applications",
            "operational_and_workflow_change",
            "algorithm_or_software_feature_changes",
            "hardware_changes", "body_part_changes"
        ])
    parser.add_argument("--llm_extractions", nargs="+",
                       default=[
                        "../analysis_validation/output/aiml_devices_validation_results.jsonl",
                        "../analysis_pre_post_associations/output/aiml_device_results.jsonl",
                       ],
                       help="Path to ground truth JSONL file")
    parser.add_argument("--annotation_files", nargs="+", 
                       default=[
                        "annotated_data/annotator1_xiao.csv",
                        "annotated_data/annotator2_patrick.csv", 
                        "annotated_data/annotator3_jean.csv",
                        "annotated_data/annotator4_adarsh.csv",
                        ],
                       help="Paths to annotator CSV files")
    parser.add_argument("--agreement_files", nargs="+", 
                       default=[
                        "annotated_data/annotator3_jean.csv",
                        "annotated_data/annotator4_adarsh.csv",
                        ],
                       help="Paths to annotator CSV files")
    parser.add_argument("--results_output", 
                       default="output/annotation_validation_results.csv",
                       help="Path to save results CSV file")
    
    args = parser.parse_args()
    
    # Load and process data
    llm_extractions = load_llm_extractions(args.llm_extractions)
    device_annotations = load_annotations(args.annotation_files)
    device_inter_annotations = load_annotations(args.agreement_files)

    # Check agreement rates
    agreement_df = get_agreement_rates(device_inter_annotations, args.all_attrs)
    print(f"AGREEMENT, avg {agreement_df['agreement'].mean()}")
    print(agreement_df)
    
    # Aggregate annotations (majority vote for boolean, average for numeric)
    agg_dict = {col: lambda x: round(x.mean()) for col in args.all_attrs}
    agg_dict["num_sites"] = lambda x: round(x.mean())
    
    device_annotations = device_annotations.groupby("device number").agg(agg_dict).reset_index()
    
    # Create matched dataset with all attributes
    matched_data = []
    for _, row in device_annotations.iterrows():
        device_num = row["device number"].lower()
        data_row = {"device_number": device_num}
        
        # Add ground truth and llm extractions for all boolean attributes
        for attr in args.all_attrs:
            # print("llm_extractions[device_num][attr]", attr, llm_extractions[device_num][attr])
            data_row[f"pred_{attr}"] = int(llm_extractions[device_num][attr])
            data_row[f"true_{attr}"] = row[attr]
        
        matched_data.append(data_row)
    
    matched_df = pd.DataFrame(matched_data)

    matched_df = get_device_strata(matched_df)
    
    # Get prevalence of attributes
    prevalence_df = calculate_attribute_prevalence(llm_extractions, args.all_attrs)
    print("\n" + "="*50)
    print("ATTRIBUTE PREVALENCE ACROSS ALL LLM EXTRACTIONS")
    print("="*50)
    print(prevalence_df.to_string(index=False))
    print()

    # Evaluation
    sampling_weights = calculate_sampling_weights(llm_extractions, device_annotations['device number'])
    results_weighted = evaluate_all_attributes(matched_df, sampling_weights, args.all_attrs)
    print("results_weighted")
    print(results_weighted)
    results_weighted.to_csv(args.results_output)
    results_unweighted = evaluate_all_attributes(matched_df, {"prospective": 1, "multisite": 1, "neither": 1}, args.all_attrs)
    print("results_unweighted")
    print(results_unweighted)
    
if __name__ == "__main__":
    main()