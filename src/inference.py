#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for the argumentation mining model.
This script demonstrates how to use the trained model to analyze new texts.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional

import torch
from models.transformer_model import ArgumentComponentDetector
from utils.visualization import visualize_text_with_components, create_argument_graph, visualize_argument_graph

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with argumentation mining model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to input file with texts to analyze (JSON)')
    parser.add_argument('--input_text', type=str, default=None,
                        help='Text to analyze (alternative to input_file)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--visualize', action='store_true',
                        help='Whether to generate visualizations')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for inference (cpu or cuda, default: auto-detect)')

    return parser.parse_args()


def load_input_texts(input_file: str) -> List[str]:
    """
    Load input texts from a file.

    Args:
        input_file: Path to input file (JSON or text)

    Returns:
        List of texts to analyze
    """
    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                # List of strings
                texts = data
            elif all(isinstance(item, dict) for item in data):
                # List of dictionaries, extract 'text' field
                texts = [item.get('text', '') for item in data if 'text' in item]
            else:
                raise ValueError("JSON file must contain a list of strings or objects with 'text' field")
        elif isinstance(data, dict) and 'texts' in data:
            # Dictionary with 'texts' field
            texts = data['texts']
        else:
            raise ValueError("JSON file must contain a list of strings or objects with 'text' field")
    else:
        # Assume plain text file, one text per line
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

    return texts


def analyze_text(
    model: ArgumentComponentDetector,
    text: str,
    output_dir: Optional[str] = None,
    visualize: bool = False,
    save_prefix: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a single text.

    Args:
        model: ArgumentComponentDetector model
        text: Text to analyze
        output_dir: Directory to save results
        visualize: Whether to generate visualizations
        save_prefix: Prefix for saved files

    Returns:
        Dictionary with analysis results
    """
    # Predict argument components
    predictions = model.predict([text])[0]

    # Extract the component type and confidence
    component_type = predictions['predicted_label']
    confidence = predictions['confidence']

    logger.info(f"Detected component: {component_type} (confidence: {confidence:.4f})")

    # Create result dictionary
    result = {
        'text': text,
        'component_type': component_type,
        'confidence': confidence,
        'probabilities': predictions['probabilities']
    }

    # Generate visualizations if requested
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Create a component for visualization
        component = {
            'type': component_type,
            'start': 0,
            'end': len(text),
            'text': text
        }

        # Define file names
        file_prefix = f"{save_prefix}_" if save_prefix else ""
        text_vis_path = os.path.join(output_dir, f"{file_prefix}text_visualization.png")

        # Visualize text with components
        visualize_text_with_components(
            text=text,
            components=[component],
            output_path=text_vis_path,
            title=f"Detected Component: {component_type}"
        )

        logger.info(f"Text visualization saved to {text_vis_path}")

        # Add visualization paths to result
        result['visualizations'] = {
            'text': text_vis_path
        }

    return result


def analyze_texts(
    model: ArgumentComponentDetector,
    texts: List[str],
    output_dir: Optional[str] = None,
    visualize: bool = False,
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """
    Analyze multiple texts.

    Args:
        model: ArgumentComponentDetector model
        texts: List of texts to analyze
        output_dir: Directory to save results
        visualize: Whether to generate visualizations
        batch_size: Batch size for inference

    Returns:
        List of dictionaries with analysis results
    """
    results = []

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Predict argument components
        batch_predictions = model.predict(batch)

        # Process each prediction
        for j, predictions in enumerate(batch_predictions):
            idx = i + j
            text = texts[idx]

            # Extract the component type and confidence
            component_type = predictions['predicted_label']
            confidence = predictions['confidence']

            logger.info(f"Text {idx+1}/{len(texts)}: {component_type} (confidence: {confidence:.4f})")

            # Create result dictionary
            result = {
                'id': idx + 1,
                'text': text,
                'component_type': component_type,
                'confidence': confidence,
                'probabilities': predictions['probabilities']
            }

            # Generate visualizations if requested
            if visualize and output_dir:
                # Create a component for visualization
                component = {
                    'type': component_type,
                    'start': 0,
                    'end': len(text),
                    'text': text
                }

                # Define file names
                text_vis_path = os.path.join(output_dir, f"text_{idx+1}_visualization.png")

                # Visualize text with components
                visualize_text_with_components(
                    text=text,
                    components=[component],
                    output_path=text_vis_path,
                    title=f"Text {idx+1} - Detected Component: {component_type}"
                )

                # Add visualization paths to result
                result['visualizations'] = {
                    'text': text_vis_path
                }

            results.append(result)

    return results


def main():
    """Main function."""
    args = parse_arguments()

    # Check if either input_file or input_text is provided
    if args.input_file is None and args.input_text is None:
        raise ValueError("Either --input_file or --input_text must be provided")

    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Using device: {device}")

    # Load the model
    logger.info(f"Loading model from {args.model_path}")
    model = ArgumentComponentDetector(
        model_path=args.model_path,
        device=device
    )

    # Process input
    if args.input_file:
        # Load texts from file
        logger.info(f"Loading texts from {args.input_file}")
        texts = load_input_texts(args.input_file)
        logger.info(f"Loaded {len(texts)} texts for analysis")

        # Analyze texts
        results = analyze_texts(
            model=model,
            texts=texts,
            output_dir=args.output_dir,
            visualize=args.visualize,
            batch_size=args.batch_size
        )

        # Save results
        if args.output_dir:
            results_file = os.path.join(args.output_dir, "results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {results_file}")

    elif args.input_text:
        # Analyze single text
        logger.info("Analyzing input text")
        result = analyze_text(
            model=model,
            text=args.input_text,
            output_dir=args.output_dir,
            visualize=args.visualize,
            save_prefix="input"
        )

        # Save result
        if args.output_dir:
            result_file = os.path.join(args.output_dir, "result.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Result saved to {result_file}")

    logger.info("Inference completed successfully")


if __name__ == "__main__":
    main()