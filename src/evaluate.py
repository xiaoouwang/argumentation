#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for the argumentation mining model.
This script handles model evaluation on test data and reports metrics.
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
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, set_seed

from models.transformer_model import ArgumentationTransformer
from train import ArgumentationDataset, compute_metrics, load_examples

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate argumentation mining model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to the test data')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initialization')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Whether to save model predictions')
    parser.add_argument('--visualize', action='store_true',
                        help='Whether to generate visualizations')

    return parser.parse_args()


def evaluate_model(args):
    """
    Evaluate the model on test data.

    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model configuration
    if os.path.exists(os.path.join(args.model_path, "training_config.yaml")):
        with open(os.path.join(args.model_path, "training_config.yaml"), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Configuration file not found. Using default configuration.")
        config = {
            'model': {
                'max_seq_length': args.max_seq_length,
            },
            'task': {
                'num_labels': 5,
                'label2id': {
                    "Claim": 0,
                    "Premise": 1,
                    "Major_Claim": 2,
                    "Non_argumentative": 3,
                    "Backing": 4,
                },
                'id2label': {
                    "0": "Claim",
                    "1": "Premise",
                    "2": "Major_Claim",
                    "3": "Non_argumentative",
                    "4": "Backing",
                }
            },
            'relation': {
                'enabled': True,
                'relation_types': ["Support", "Attack", "None"],
            }
        }

    # Override with command line arguments
    config['model']['max_seq_length'] = args.max_seq_length

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = ArgumentationTransformer.from_pretrained(
        args.model_path,
        num_labels=config['task']['num_labels'],
        relation_types=len(config['relation']['relation_types']) if config['relation']['enabled'] else 0,
        enable_relation_classification=config['relation']['enabled'],
    )

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load test data
    test_examples = load_examples(args.test_data)

    # Create dataset and dataloader
    test_dataset = ArgumentationDataset(
        test_examples,
        tokenizer,
        max_seq_length=config['model']['max_seq_length'],
        label2id=config['task']['label2id'],
        relation_types={rel: i for i, rel in enumerate(config['relation']['relation_types'])}
            if config['relation']['enabled'] else None,
    )

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size,
        collate_fn=lambda x: {key: torch.stack([example[key] for example in x])
                               for key in x[0].keys()},
    )

    # Evaluation
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Batch size = {args.batch_size}")

    test_component_logits = []
    test_component_labels = []
    test_relation_logits = []
    test_relation_labels = []

    # Store all predictions if requested
    all_predictions = []

    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluation")):
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
        test_component_logits.append(outputs['component_logits'].detach().cpu().numpy())
        test_component_labels.append(batch['component_labels'].detach().cpu().numpy())

        if 'relation_logits' in outputs:
            test_relation_logits.append(outputs['relation_logits'].detach().cpu().numpy())
            test_relation_labels.append(batch['relation_labels'].detach().cpu().numpy())

        # Save predictions if requested
        if args.save_predictions:
            # Get predictions for component classification
            component_probs = torch.softmax(outputs['component_logits'], dim=-1)
            component_preds = torch.argmax(component_probs, dim=-1)

            # Get predictions for relation classification if enabled
            relation_preds = None
            relation_probs = None
            if 'relation_logits' in outputs:
                relation_probs = torch.softmax(outputs['relation_logits'], dim=-1)
                relation_preds = torch.argmax(relation_probs, dim=-1)

            # Get the original examples for this batch
            batch_examples = test_examples[batch_idx * args.batch_size:
                                           min((batch_idx + 1) * args.batch_size, len(test_examples))]

            # Create prediction data for each example
            for i, example in enumerate(batch_examples):
                example_id = example.get('id', f"example_{batch_idx}_{i}")

                # Component prediction
                component_pred = component_preds[i].item()
                component_prob = component_probs[i][component_pred].item()
                component_label = config['task']['id2label'][str(component_pred)]

                prediction = {
                    'id': example_id,
                    'text': example['text'],
                    'predicted_component': component_label,
                    'component_confidence': component_prob,
                    'component_probabilities': {
                        config['task']['id2label'][str(j)]: prob.item()
                        for j, prob in enumerate(component_probs[i])
                    },
                    'true_component': config['task']['id2label'].get(
                        str(batch['component_labels'][i].item()), "Unknown"
                    ),
                }

                # Relation predictions if available
                if relation_preds is not None:
                    relation_predictions = []

                    for j in range(relation_preds.size(1)):
                        # Skip padding relations
                        if batch['relation_pairs'][i, j, 0].item() == -1:
                            continue

                        rel_pred = relation_preds[i, j].item()
                        rel_prob = relation_probs[i, j, rel_pred].item()
                        rel_type = config['relation']['relation_types'][rel_pred]

                        # Get the true relation type
                        true_rel_idx = batch['relation_labels'][i, j].item()
                        true_rel_type = (
                            config['relation']['relation_types'][true_rel_idx]
                            if true_rel_idx != -1 and true_rel_idx < len(config['relation']['relation_types'])
                            else "Unknown"
                        )

                        source_idx = batch['relation_pairs'][i, j, 0].item()
                        target_idx = batch['relation_pairs'][i, j, 1].item()

                        relation_predictions.append({
                            'source_token': source_idx,
                            'target_token': target_idx,
                            'predicted_relation': rel_type,
                            'relation_confidence': rel_prob,
                            'true_relation': true_rel_type,
                        })

                    prediction['relation_predictions'] = relation_predictions

                all_predictions.append(prediction)

    # Concatenate logits and labels
    test_component_logits = np.concatenate(test_component_logits, axis=0)
    test_component_labels = np.concatenate(test_component_labels, axis=0)

    test_preds = {'component_logits': test_component_logits}
    test_labels = {'component_labels': test_component_labels}

    if test_relation_logits and test_relation_labels:
        test_relation_logits = np.concatenate(test_relation_logits, axis=0)
        test_relation_labels = np.concatenate(test_relation_labels, axis=0)
        test_preds['relation_logits'] = test_relation_logits
        test_labels['relation_labels'] = test_relation_labels

    # Compute metrics
    metrics = compute_metrics(test_preds, test_labels)

    # Log metrics
    logger.info("***** Evaluation Results *****")
    for key, value in metrics.items():
        logger.info(f"  {key} = {value:.4f}")

    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_file}")

    # Save predictions if requested
    if args.save_predictions:
        predictions_file = os.path.join(args.output_dir, "predictions.json")
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(all_predictions, f, indent=2, ensure_ascii=False)

        logger.info(f"Predictions saved to {predictions_file}")

    # Generate visualizations if requested
    if args.visualize:
        try:
            generate_visualizations(test_examples, all_predictions, config, args.output_dir)
            logger.info(f"Visualizations saved to {args.output_dir}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

    return metrics


def generate_visualizations(examples, predictions, config, output_dir):
    """
    Generate visualizations of model predictions.

    Args:
        examples: List of test examples
        predictions: List of model predictions
        config: Model configuration
        output_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Extract true and predicted labels
    y_true = []
    y_pred = []

    for pred in predictions:
        true_label = pred['true_component']
        pred_label = pred['predicted_component']

        # Convert to label index
        true_idx = config['task']['label2id'].get(true_label, 0)
        pred_idx = config['task']['label2id'].get(pred_label, 0)

        y_true.append(true_idx)
        y_pred.append(pred_idx)

    # Create confusion matrix
    labels = list(config['task']['label2id'].keys())
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    ax.set_title("Confusion Matrix - Argument Component Classification")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Plot class distribution
    class_counts = {label: 0 for label in labels}
    for label in [pred['predicted_component'] for pred in predictions]:
        class_counts[label] = class_counts.get(label, 0) + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(class_counts.keys(), class_counts.values())
    ax.set_xlabel("Argument Component Type")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predicted Argument Components")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "class_distribution.png"), dpi=300)
    plt.close()

    # Plot confidence distribution
    confidences = [pred['component_confidence'] for pred in predictions]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidences, bins=20)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Model Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "confidence_distribution.png"), dpi=300)
    plt.close()

    # If relation predictions are available, visualize them too
    if any('relation_predictions' in pred for pred in predictions):
        # Count relation types
        relation_counts = {}
        for pred in predictions:
            if 'relation_predictions' in pred:
                for rel in pred['relation_predictions']:
                    rel_type = rel['predicted_relation']
                    relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1

        if relation_counts:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(relation_counts.keys(), relation_counts.values())
            ax.set_xlabel("Relation Type")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Predicted Relations")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "relation_distribution.png"), dpi=300)
            plt.close()


def main():
    """Main function."""
    args = parse_arguments()
    evaluate_model(args)


if __name__ == "__main__":
    main()