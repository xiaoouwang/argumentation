#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the argumentation mining model.
This script handles model training, validation, and checkpointing.
"""

import os
import json
import yaml
import logging
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from models.transformer_model import ArgumentationTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArgumentationDataset(Dataset):
    """Dataset for argumentation mining tasks."""

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer,
        max_seq_length: int = 512,
        label2id: Optional[Dict[str, int]] = None,
        relation_types: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            examples: List of processed examples
            tokenizer: Tokenizer for encoding text
            max_seq_length: Maximum sequence length
            label2id: Mapping from label names to IDs
            relation_types: Mapping from relation types to IDs
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Set default label mapping if not provided
        if label2id is None:
            self.label2id = {
                "Claim": 0,
                "Premise": 1,
                "Major_Claim": 2,
                "Non_argumentative": 3,
                "Backing": 4,
            }
        else:
            self.label2id = label2id

        # Set default relation type mapping if not provided
        if relation_types is None:
            self.relation_types = {
                "Support": 0,
                "Attack": 1,
                "None": 2,
            }
        else:
            self.relation_types = relation_types

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Get a single example from the dataset."""
        example = self.examples[idx]

        # Encode the text with the tokenizer
        encoding = self.tokenizer(
            example['text'],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Remove the batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Get the component label (for now, we'll use the majority label in the document)
        component_counts = {}
        for component in example['components']:
            component_type = component['type']
            if component_type in self.label2id:
                component_counts[component_type] = component_counts.get(component_type, 0) + 1

        # Find the majority component type
        if component_counts:
            majority_type = max(component_counts, key=component_counts.get)
            component_label = self.label2id[majority_type]
        else:
            # Default to Non_argumentative if no components found
            component_label = self.label2id.get("Non_argumentative", 3)

        # Add the component label to the encoding
        encoding['component_labels'] = torch.tensor(component_label, dtype=torch.long)

        # Process relations if any
        if example['relations']:
            relation_pairs = []
            relation_labels = []

            # Get token IDs for component starts
            component_token_ids = {}
            for i, component in enumerate(example['components']):
                start_char = component['start']
                # Find the token ID for this character position
                # This is a simplification - in practice, you'd need to handle alignment better
                token_id = encoding['input_ids'].tolist().index(
                    self.tokenizer.encode(example['text'][start_char:start_char+1], add_special_tokens=False)[0]
                ) if start_char < len(example['text']) else 0
                component_token_ids[i] = token_id

            # Process each relation
            for relation in example['relations']:
                source_idx = relation['source']
                target_idx = relation['target']
                rel_type = relation['type']

                # Get token IDs for source and target components
                if source_idx in component_token_ids and target_idx in component_token_ids:
                    source_token_id = component_token_ids[source_idx]
                    target_token_id = component_token_ids[target_idx]

                    relation_pairs.append([source_token_id, target_token_id])
                    relation_labels.append(self.relation_types.get(rel_type, 2))  # Default to "None"

            # Pad relations to a fixed size
            max_relations = 10  # Maximum number of relations per document

            if relation_pairs:
                relation_pairs = relation_pairs[:max_relations]
                relation_labels = relation_labels[:max_relations]

                # Pad with -1 for missing relations
                while len(relation_pairs) < max_relations:
                    relation_pairs.append([-1, -1])
                    relation_labels.append(-1)  # Ignore index for padding
            else:
                # No relations, pad with -1
                relation_pairs = [[-1, -1]] * max_relations
                relation_labels = [-1] * max_relations

            encoding['relation_pairs'] = torch.tensor(relation_pairs, dtype=torch.long)
            encoding['relation_labels'] = torch.tensor(relation_labels, dtype=torch.long)
        else:
            # No relations in this example
            max_relations = 10
            relation_pairs = [[-1, -1]] * max_relations
            relation_labels = [-1] * max_relations

            encoding['relation_pairs'] = torch.tensor(relation_pairs, dtype=torch.long)
            encoding['relation_labels'] = torch.tensor(relation_labels, dtype=torch.long)

        return encoding


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train argumentation mining model')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the trained model')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to run training')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to run evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initialization')

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def load_examples(data_file: str) -> List[Dict[str, Any]]:
    """
    Load examples from a JSON file.

    Args:
        data_file: Path to the data file

    Returns:
        List of examples
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    return examples


def compute_metrics(preds, labels):
    """
    Compute evaluation metrics.

    Args:
        preds: Model predictions
        labels: Ground truth labels

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Component classification metrics
    component_preds = np.argmax(preds['component_logits'], axis=1)
    component_labels = labels['component_labels']

    component_accuracy = accuracy_score(component_labels, component_preds)
    component_precision, component_recall, component_f1, _ = precision_recall_fscore_support(
        component_labels, component_preds, average='weighted'
    )

    metrics = {
        'component_accuracy': component_accuracy,
        'component_precision': component_precision,
        'component_recall': component_recall,
        'component_f1': component_f1,
    }

    # Relation classification metrics (if available)
    if 'relation_logits' in preds and 'relation_labels' in labels:
        relation_preds = np.argmax(preds['relation_logits'].reshape(-1, preds['relation_logits'].shape[-1]), axis=1)
        relation_labels = labels['relation_labels'].reshape(-1)

        # Filter out padding (-1)
        mask = relation_labels != -1
        filtered_relation_preds = relation_preds[mask]
        filtered_relation_labels = relation_labels[mask]

        if len(filtered_relation_labels) > 0:
            relation_accuracy = accuracy_score(filtered_relation_labels, filtered_relation_preds)
            relation_precision, relation_recall, relation_f1, _ = precision_recall_fscore_support(
                filtered_relation_labels, filtered_relation_preds, average='weighted'
            )

            metrics.update({
                'relation_accuracy': relation_accuracy,
                'relation_precision': relation_precision,
                'relation_recall': relation_recall,
                'relation_f1': relation_f1,
            })

    return metrics


def train(args, config):
    """
    Train the model.

    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])

    # Load and prepare the datasets
    train_examples = load_examples(config['data']['train_file'])
    val_examples = load_examples(config['data']['validation_file'])

    # Create label mappings
    label2id = config['task']['label2id']
    id2label = config['task']['id2label']

    # Create datasets
    train_dataset = ArgumentationDataset(
        train_examples,
        tokenizer,
        max_seq_length=config['model']['max_seq_length'],
        label2id=label2id,
        relation_types={rel: i for i, rel in enumerate(config['relation']['relation_types'])}
            if config['relation']['enabled'] else None,
    )

    val_dataset = ArgumentationDataset(
        val_examples,
        tokenizer,
        max_seq_length=config['model']['max_seq_length'],
        label2id=label2id,
        relation_types={rel: i for i, rel in enumerate(config['relation']['relation_types'])}
            if config['relation']['enabled'] else None,
    )

    # Create data loaders
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config['training']['batch_size'],
        collate_fn=lambda x: {key: torch.stack([example[key] for example in x])
                               for key in x[0].keys()},
    )

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=config['training']['batch_size'],
        collate_fn=lambda x: {key: torch.stack([example[key] for example in x])
                               for key in x[0].keys()},
    )

    # Initialize the model
    model = ArgumentationTransformer(
        model_name_or_path=config['model']['base_model'],
        num_labels=config['task']['num_labels'],
        dropout_prob=config['model']['hidden_dropout_prob'],
        relation_types=len(config['relation']['relation_types']) if config['relation']['enabled'] else 0,
        enable_relation_classification=config['relation']['enabled'],
    )

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        eps=config['training']['adam_epsilon'],
        weight_decay=config['training']['weight_decay'],
    )

    # Calculate total training steps
    num_training_steps = len(train_dataloader) * config['training']['num_train_epochs']
    num_warmup_steps = int(num_training_steps * config['training']['warmup_ratio'])

    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    global_step = 0
    best_val_f1 = 0.0

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config['training']['num_train_epochs']}")
    logger.info(f"  Batch size = {config['training']['batch_size']}")
    logger.info(f"  Total optimization steps = {num_training_steps}")

    model.zero_grad()

    for epoch in range(int(config['training']['num_train_epochs'])):
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_train_epochs']}")

        # Training
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids', None),
                component_labels=batch['component_labels'],
                relation_labels=batch.get('relation_labels', None),
                relation_pairs=batch.get('relation_pairs', None),
            )

            loss = outputs['loss']
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])

            # Update parameters
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1

            # Log training progress
            if global_step % config['training']['logging_steps'] == 0:
                logger.info(f"Step {global_step}: loss = {loss.item():.4f}")

            # Save checkpoint
            if global_step % config['training']['save_steps'] == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # Log epoch results
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} - Average loss: {avg_epoch_loss:.4f}")

        # Validation
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(val_dataset)}")
        logger.info(f"  Batch size = {config['training']['batch_size']}")

        model.eval()

        val_component_logits = []
        val_component_labels = []
        val_relation_logits = []
        val_relation_labels = []

        for batch in tqdm(val_dataloader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids', None),
                    component_labels=batch['component_labels'],
                    relation_labels=batch.get('relation_labels', None),
                    relation_pairs=batch.get('relation_pairs', None),
                )

            # Collect logits and labels for metric computation
            val_component_logits.append(outputs['component_logits'].detach().cpu().numpy())
            val_component_labels.append(batch['component_labels'].detach().cpu().numpy())

            if 'relation_logits' in outputs:
                val_relation_logits.append(outputs['relation_logits'].detach().cpu().numpy())
                val_relation_labels.append(batch['relation_labels'].detach().cpu().numpy())

        # Concatenate logits and labels
        val_component_logits = np.concatenate(val_component_logits, axis=0)
        val_component_labels = np.concatenate(val_component_labels, axis=0)

        val_preds = {'component_logits': val_component_logits}
        val_labels = {'component_labels': val_component_labels}

        if val_relation_logits and val_relation_labels:
            val_relation_logits = np.concatenate(val_relation_logits, axis=0)
            val_relation_labels = np.concatenate(val_relation_labels, axis=0)
            val_preds['relation_logits'] = val_relation_logits
            val_labels['relation_labels'] = val_relation_labels

        # Compute metrics
        metrics = compute_metrics(val_preds, val_labels)

        # Log metrics
        for key, value in metrics.items():
            logger.info(f"Validation {key}: {value:.4f}")

        # Save best model
        if metrics['component_f1'] > best_val_f1:
            best_val_f1 = metrics['component_f1']
            best_model_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            logger.info(f"New best model with F1 = {best_val_f1:.4f}")

    # Save final model
    final_model_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Saved final model to {final_model_dir}")

    # Save training configuration
    with open(os.path.join(args.output_dir, "training_config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    logger.info("Training completed successfully")


def main():
    """Main function."""
    args = parse_arguments()
    config = load_config(args.config)

    if args.do_train:
        train(args, config)

    # TODO: Implement evaluation logic if needed


if __name__ == "__main__":
    main()