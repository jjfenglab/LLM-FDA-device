# This is just to compare LLM extractions against Wu
python compare_llm_results_with_previous_paper.py --output output/llm_validation_comparison.jsonl
python summarize_validation_comparison.py --input output/llm_validation_comparison.jsonl --output output/validation_comparison_detailed.csv
# This is to get LLM extractions for all devices
python survey_validation_trends_all_devices.py --input ../../data/aiml_device_numbers_071025.json --output output/aiml_devices_validation_results.jsonl
python plot_validation_study_trends.py --input output/aiml_devices_validation_results.jsonl --metadata ../analysis_pre_post_associations/output/aiml_device_results_with_metadata.jsonl --output output/validation_trends.png
# Rule-based comparator
python run_rule_based_comparator.py --input ../../data/aiml_device_numbers_071025.json --llm-output output/aiml_devices_validation_results.jsonl --output output/aiml_device_rule_extractions.jsonl
python summarize_validation_comparison.py --input output/aiml_device_rule_extractions.jsonl --output output/rule_based_validation_wu.csv