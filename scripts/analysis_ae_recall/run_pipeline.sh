# Run after scripts/analysis_pre_post_associations
python retrieve_fda_data.py --input-all ../../data/aiml_device_numbers_071025.json --input-510 ../analysis_pre_post_associations/output/aiml_device_results_with_metadata.jsonl --output output/fda_data_retrieval_results.json
python categorize_with_llm.py --input output/fda_data_retrieval_results.json --device-names ../analysis_pre_post_associations/output/aiml_device_results_with_metadata_names.jsonl --output output/adverse_events_analysis_results_event_prob_rebuttal_names.json
python run_llm_judge.py --llm-provider openai --input output/adverse_events_analysis_results_event_prob_rebuttal_names.json --output output/adverse_events_analysis_results_event_prob_rebuttal_names_gpt.json
python run_llm_judge.py --llm-provider claude --input output/adverse_events_analysis_results_event_prob_rebuttal_names.json --output output/adverse_events_analysis_results_event_prob_rebuttal_names_claude.json

python plot_ae_recall_analysis.py --results-file output/adverse_events_analysis_results_event_prob_rebuttal_names.json --judge output/adverse_events_analysis_results_event_prob_rebuttal_names_gpt.json output/adverse_events_analysis_results_event_prob_rebuttal_names_claude.json --output-fig-event output/adverse_events_by_event_type_rebuttal.png --output-fig-problem output/adverse_events_by_product_probs_rebuttal.png --log-file output/log_mdr_rebuttal.txt

# rule-based analysis
python categorize_with_rules.py --input output/fda_data_retrieval_results.json --device-names ../analysis_pre_post_associations/output/aiml_device_results_with_metadata_names.jsonl --output output/adverse_events_analysis_results_event_rules.json