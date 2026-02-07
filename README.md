# Scaling Medical Device Regulatory Science using Large Language Models

This repository contains the code and analysis pipeline for

<img width="1017" height="307" alt="LLM pipeline" src="https://github.com/jjfenglab/LLM-FDA-device/figure1.pdf" />


**Hanyang Li, Xiao He, Adarsh Subbaswamy, Patrick Vossler, Alexej Gossmann, Karandeep Singh & Jean Feng. Scaling medical device regulatory science using large language models. npj Digital Medicine (2026).** https://doi.org/10.1038/s41746-026-02353-7

## Overview

This work develops and validates an LLM-based pipeline for scaling data analyses in medical device regulatory science. We demonstrate how LLMs can accurately extract structured information from complex, unstructured FDA regulatory documents in three case studies:
1. **Device Validation Practices**: What validation practices are reported for FDA-cleared/approved AI/ML medical devices?
2. **Medical Device Report (MDR) Coding**: Can LLMs assist/improve the accuracy of codes assigned to MDRs?
3. **Pre-Market Risk Factors**: Can we identify device characteristics during FDA clearance that are associated with post-market MDRs?

## Repository Structure

### Main Components

- **`data/`**: FDA reference datasets
  - `FDA-CDRH_NCIt_Subsets.csv` - FDA medical device terminology/classification data
  - Other relevant datasets can be downloaded using the provided scripts

- **`scripts/`**: Core analysis pipeline
  - `common.py` - Shared utilities for data processing and analysis
  - `download_device_pdfs.py` - FDA API integration for downloading device summaries
  - Each case study corresponds to a folder:
     1. Device Validation Practices: `scripts/analysis_validation/`
     2. MDR coding: `scripts/analysis_ae_recall/`
     3. Pre-Market Risk Factors: `scripts/analysis_pre_post_associations/`
- **`scripts/utils/`**: Utility modules
  - `gpt_utils.py` - OpenAI API integration with cost tracking for multiple GPT models
  - `pdf_utils.py` - PDF text extraction with OCR fallback for poor quality documents
  - `extract_primary_predicate.py` - Regex-based extraction of device predicate information

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis Pipelines

Each analysis workflow can be executed by running `run_pipeline.sh` within its respective folder

### Configuration

Create a `.env` file with your API credentials:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{li2026scaling,
  title={Scaling medical device regulatory science using large language models},
  author={Li, Hanyang and He, Xiao and Subbaswamy, Adarsh and Vossler, Patrick and Gossmann, Alexej and Singh, Karandeep and Feng, Jean},
  journal={npj Digital Medicine},
  year={2026},
  doi={10.1038/s41746-026-02353-7},
  url={https://doi.org/10.1038/s41746-026-02353-7}
}
```
