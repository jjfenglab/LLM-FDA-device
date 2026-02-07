# Scaling Medical Device Regulatory Science using Large Language Models

This repository contains the code and analysis pipeline for

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

- **`scripts/`**: Core analysis pipeline (~6,489 lines of Python code)
  - `common.py` - Shared utilities for data processing and analysis
  - `download_device_pdfs.py` - FDA API integration for downloading device summaries

- **`scripts/utils/`**: Utility modules
  - `gpt_utils.py` - OpenAI API integration with cost tracking for multiple GPT models
  - `pdf_utils.py` - PDF text extraction with OCR fallback for poor quality documents
  - `extract_primary_predicate.py` - Regex-based extraction of device predicate information

### Analysis Workflows

Each analysis module contains a complete pipeline with shell script orchestration:

#### 1. **Validation Analysis** (`scripts/analysis_validation/`)
- Surveys device validation practices across all AI/ML devices (1995-2025)
- Compares LLM extractions with previous manual studies
- Identifies trends in prospective vs. retrospective studies and multi-site evaluations

#### 2. **Annotation Studies** (`scripts/analysis_annotation_studies/`)
- Validates LLM extractions against expert human annotations
- Measures inter-rater agreement between human annotators
- Compares LLM vs. human performance on device summaries and MDRs

#### 3. **Pre/Post-Market Associations** (`scripts/analysis_pre_post_associations/`)
- Links pre-market device characteristics to post-market adverse events
- Uses Cox proportional hazards models for survival analysis
- Identifies risk factors associated with higher rates of adverse events

#### 4. **Adverse Event & Recall Analysis** (`scripts/analysis_ae_recall/`)
- Categorizes adverse events using LLM-based Medical Device Problem codes
- Compares LLM vs. vendor-assigned event classifications
- Uses "LLM-as-a-judge" validation with multiple models (GPT-4.1, Claude Sonnet)

## Technical Architecture

### LLM Pipeline
- **Primary Model**: GPT-4.1 from OpenAI for information extraction
- **Validation Models**: Claude Sonnet 4.5 for LLM-as-a-judge validation
- **Structured Prompts**: In-context learning with regulatory examples
- **Temperature**: Set to 0 for reproducible results
- **Error Handling**: OCR fallback for poor quality PDF documents

### Data Processing Flow
1. **Data Collection**: FDA APIs (MAUDE database, Recall API, Decision Summaries)
2. **Preprocessing**: PDF text extraction with OCR backup using tesseract
3. **LLM Extraction**: Structured information extraction with GPT-4.1
4. **Validation**: Human expert annotation and LLM-as-a-judge comparison
5. **Statistical Analysis**: Cox models, survival analysis, trend visualization

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis Pipelines

Each analysis workflow can be executed via its pipeline script:

```bash
# Validation analysis
cd scripts/analysis_validation && ./run_pipeline.sh

# Annotation studies
cd scripts/analysis_annotation_studies && ./run_pipeline.sh

# Pre/post-market associations
cd scripts/analysis_pre_post_associations && ./run_pipeline.sh

# Adverse event analysis
cd scripts/analysis_ae_recall && ./run_pipeline.sh
```

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
