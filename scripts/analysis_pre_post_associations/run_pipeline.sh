python extract_all_device_features.py --output output/aiml_device_results.jsonl
python filter_predicate_and_add_metadata.py --device output/aiml_device_results.jsonl --output output/aiml_device_results_with_metadata_names.jsonl
python filter_predicate_and_add_metadata.py --device output/aiml_device_results.jsonl --output output/aiml_device_results_with_metadata.jsonl
python merge_pre_post_market_data.py --device_results_file output/aiml_device_results_with_metadata.jsonl --ae_result ../analysis_ae_recall/output/adverse_events_analysis_results_event_prob_struct.json --output output/merged_pre_post_market_data_mapped.json
python prepare_statistical_model_input_data.py --output output/ae_survival_data_mapped.csv --input output/merged_pre_post_market_data_mapped.json
# Then run survival.Rmd
