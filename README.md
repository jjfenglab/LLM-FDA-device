# Scaling Medical Device Regulatory Science using Large Language Models

This repository contains the code and analysis pipeline for our research paper published in *npj Digital Medicine*:

**Li, H., He, X., Subbaswamy, A. et al. Scaling medical device regulatory science using large language models. npj Digit. Med. (2026).** https://doi.org/10.1038/s41746-026-02353-7

## Overview

This work presents the first wide-ranging validation study of Large Language Models (LLMs) for scaling data analyses in medical device regulatory science. We demonstrate how LLMs can accurately extract structured information from complex, unstructured FDA regulatory documents, enabling rapid analysis of AI/ML-enabled medical devices at unprecedented scale.

### Key Findings

- **Dataset Scale**: Analysis of 1,247 FDA-authorized AI/ML-enabled medical devices and 1,852 Medical Device Reports (MDRs)
- **High Accuracy**: LLM extractions achieve 80%+ accuracy rates across multiple regulatory attributes
- **Time Savings**: Analyses that previously took months/years can now be completed in days
- **Three Major Applications**: Device validation practice monitoring, MDR coding improvement, and pre/post-market risk factor identification

## Research Questions Addressed

1. **Device Validation Practices**: How have validation studies for AI/ML medical devices evolved over time?
2. **Medical Device Report (MDR) Coding**: Can LLMs improve the accuracy of adverse event classification?
3. **Pre-Market Risk Factors**: What device characteristics during FDA clearance predict post-market adverse events?

## Repository Structure

### Main Components

- **`data/`**: FDA reference datasets
  - `FDA-CDRH_NCIt_Subsets.csv` - FDA medical device terminology/classification data

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

### Key Technologies
- **LLM APIs**: OpenAI GPT-4.1, Anthropic Claude Sonnet 4.5
- **Data Processing**: pandas, numpy, matplotlib, seaborn
- **PDF Processing**: pymupdf, pytesseract (OCR)
- **Statistical Analysis**: R/RMarkdown for survival analysis
- **Async Operations**: aiohttp, asyncio for concurrent API calls
- **Configuration**: python-dotenv for environment management

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

## Dataset

- **AI/ML-enabled Devices**: 1,247 devices (1,227 with available decision summaries)
- **Medical Device Reports**: 1,852 MDRs (1,841 after filtering)
- **Time Period**: 1995-2025
- **Regulatory Pathways**: 510(k), De Novo, PMA clearances
- **Data Sources**: FDA APIs, MAUDE database, FDA AI/ML device listings

## Key Results

### Validation Performance
- **Device Attributes**: 80-90% accuracy for device validation characteristics
- **MDR Classification**: 88% human preference for LLM vs. vendor Event Type codes
- **Predicate Extraction**: 99% agreement with prior manual studies

### Regulatory Insights
- **Validation Trends**: Multi-site studies increased from 20% to 50+ over time
- **Coding Improvements**: LLM identifies systematic misclassifications in vendor-coded MDRs
- **Risk Factors**: Hardware changes and recall history associated with higher adverse event rates

## Impact and Applications

This work demonstrates how LLMs can:
- **Scale Regulatory Analysis**: Analyze 1,200+ devices in days vs. months manually
- **Improve Data Quality**: Identify and correct systematic coding errors in surveillance data
- **Enable Rapid Insights**: Support real-time policy development and regulatory decision-making
- **Enhance Transparency**: Make regulatory data analysis accessible to broader research community

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
