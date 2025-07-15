This repository contains the implementation of the methods described in our paper: "When automated fact-checking meets argumentation: unveiling fake news through argumentative evidence".

## Overview

This codebase implements a deep learning framework for argument mining, fake news classification and analysis. It includes:

- Data preprocessing and feature extraction
- Transformer-based models for joint argument mining and fake news classification
- Evaluation metrics and visualization tools

## Data

In the data folder you will find LIARArg.csv

The names of the columns are quite self-explanatory, here are some explanations for the less straightforward columns:

- `claim_id`: Each claim has a unique id to be identified during the relation prediction task.
- `premise_id`: Each premise has a unique id to be identified during the relation prediction task.
- `claim_position`: The start and end position of a claim vs. the text in the column`whole_text`.
- `premise_position`: The start and end position of a premise vs. the text in the column`whole_text`.
- `summary`: The summary of the whole fact-checking article (see the paper for more details).
- `whole_text`: The whole text of the fact-checking article.
- `fullText_based_content`: The full-length content of the fact-checking article, useful for summary generation.
- `support_relation` and similar columns: the dependant and governor of each relation. For instance, [1838, 1828] in the row with id 8249 means that the component with id 1838 supports the component with id 1828. In this specific example, 1838 is a premise and 1828 is a claim. 



## Installation

To install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

```bash
python src/data/prepare_dataset.py --input_path data/raw --output_path data/processed
```

### Model Training

```bash
python src/train.py --config configs/transformer_config.yaml
```

### Evaluation

```bash
python src/evaluate.py --model_path models/best_model.pt --test_data data/processed/test.json
```