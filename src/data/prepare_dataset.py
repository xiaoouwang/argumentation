#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare datasets for argumentation mining tasks.
This script processes raw data into the format required for training and evaluation.
"""

import os
import json
import argparse
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import logging
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
nltk.download('punkt', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare data for argumentation mining')

    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the directory containing raw input data')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save processed data')
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Train/validation/test split ratio (e.g., 0.8 0.1 0.1)')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of examples to sample (for debugging)')

    return parser.parse_args()


def process_raw_files(input_path: str) -> List[Dict[str, Any]]:
    """
    Process raw input files.

    Args:
        input_path: Path to the directory containing raw input files

    Returns:
        List of processed examples
    """
    logger.info(f"Processing raw files from {input_path}")

    # This function should be adapted to your specific data format
    # Here's a placeholder implementation that assumes each file contains:
    # 1. One document per file
    # 2. Each document has text and annotations

    examples = []

    # Get all files in the input directory
    files = [f for f in os.listdir(input_path) if f.endswith('.txt') or f.endswith('.json')]

    for file in tqdm(files, desc="Processing files"):
        file_path = os.path.join(input_path, file)

        # Determine file type and process accordingly
        if file.endswith('.json'):
            # Assume JSON files contain pre-annotated data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Extract text and annotations based on your data format
                # This is just a placeholder - adapt to your specific format
                document_id = data.get('id', file)
                text = data.get('text', '')

                annotations = data.get('annotations', [])
                components = []

                for annotation in annotations:
                    component_type = annotation.get('type')
                    start = annotation.get('start')
                    end = annotation.get('end')
                    text_span = text[start:end] if start is not None and end is not None else ''

                    components.append({
                        'type': component_type,
                        'start': start,
                        'end': end,
                        'text': text_span
                    })

                relations = []
                for relation in data.get('relations', []):
                    source = relation.get('source')
                    target = relation.get('target')
                    rel_type = relation.get('type')

                    relations.append({
                        'source': source,
                        'target': target,
                        'type': rel_type
                    })

                examples.append({
                    'id': document_id,
                    'text': text,
                    'components': components,
                    'relations': relations
                })

        elif file.endswith('.txt'):
            # For plain text files, we'll use a simple heuristic approach
            # to identify potential argument components
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            document_id = file

            # Split into sentences
            sentences = sent_tokenize(text)

            # Process each sentence with spaCy
            components = []
            offset = 0

            for sentence in sentences:
                # Find the start and end positions in the original text
                start = text.find(sentence, offset)
                end = start + len(sentence)
                offset = end

                # Use SpaCy to analyze the sentence
                doc = nlp(sentence)

                # Heuristic: sentences with modal verbs or specific phrases
                # are more likely to be claims
                modal_verbs = ['must', 'should', 'could', 'would', 'might', 'may', 'can']
                claim_markers = ['believe', 'think', 'argue', 'claim', 'suggest', 'conclude']
                premise_markers = ['because', 'since', 'therefore', 'thus', 'as a result']

                has_modal = any(token.text.lower() in modal_verbs for token in doc)
                has_claim_marker = any(token.lemma_.lower() in claim_markers for token in doc)
                has_premise_marker = any(token.text.lower() in premise_markers for token in doc)

                # Assign component type based on heuristics
                if has_claim_marker or has_modal:
                    component_type = "Claim"
                elif has_premise_marker:
                    component_type = "Premise"
                else:
                    component_type = "Non_argumentative"

                components.append({
                    'type': component_type,
                    'start': start,
                    'end': end,
                    'text': sentence
                })

            examples.append({
                'id': document_id,
                'text': text,
                'components': components,
                'relations': []  # No relations for plain text files
            })

    logger.info(f"Processed {len(examples)} examples")
    return examples


def split_data(examples: List[Dict[str, Any]],
               split_ratio: List[float],
               random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train, validation, and test sets.

    Args:
        examples: List of processed examples
        split_ratio: List of three floats for train/val/test split
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    import numpy as np
    np.random.seed(random_seed)

    # Shuffle the examples
    indices = np.random.permutation(len(examples))

    # Calculate split indices
    train_end = int(split_ratio[0] * len(examples))
    val_end = train_end + int(split_ratio[1] * len(examples))

    # Split the data
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_examples = [examples[i] for i in train_indices]
    val_examples = [examples[i] for i in val_indices]
    test_examples = [examples[i] for i in test_indices]

    logger.info(f"Split data into {len(train_examples)} train, "
                f"{len(val_examples)} validation, and {len(test_examples)} test examples")

    return train_examples, val_examples, test_examples


def save_split(examples: List[Dict[str, Any]], output_path: str, split_name: str):
    """
    Save a data split to disk.

    Args:
        examples: List of examples
        output_path: Directory to save data
        split_name: Name of the split (train, validation, test)
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{split_name}.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(examples)} examples to {output_file}")


def main():
    """Main function to prepare the dataset."""
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Process raw data
    examples = process_raw_files(args.input_path)

    # Sample data for debugging if requested
    if args.sample_size and args.sample_size < len(examples):
        import random
        random.seed(args.random_seed)
        examples = random.sample(examples, args.sample_size)
        logger.info(f"Sampled {len(examples)} examples for debugging")

    # Split data
    train_examples, val_examples, test_examples = split_data(
        examples, args.split_ratio, args.random_seed
    )

    # Save splits
    save_split(train_examples, args.output_path, "train")
    save_split(val_examples, args.output_path, "validation")
    save_split(test_examples, args.output_path, "test")

    logger.info("Data preparation completed successfully")


if __name__ == "__main__":
    main()