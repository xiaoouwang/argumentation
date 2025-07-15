#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for argumentation structures.
This module provides functions to visualize argument components and relations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx
from matplotlib.colors import to_rgba


def create_argument_graph(
    components: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    title: str = "Argumentation Structure"
) -> nx.DiGraph:
    """
    Create a directed graph representing an argument structure.

    Args:
        components: List of argument components
        relations: List of relations between components
        title: Title of the graph

    Returns:
        A NetworkX directed graph
    """
    # Create a directed graph
    G = nx.DiGraph(title=title)

    # Add nodes (components)
    for i, component in enumerate(components):
        component_type = component.get('type', 'Unknown')
        text = component.get('text', '')

        # Use first 30 characters of text as node label
        label = text[:30] + '...' if len(text) > 30 else text

        # Add node to graph
        G.add_node(
            i,
            label=label,
            type=component_type,
            text=text,
            start=component.get('start'),
            end=component.get('end')
        )

    # Add edges (relations)
    for relation in relations:
        source = relation.get('source')
        target = relation.get('target')
        rel_type = relation.get('type', 'Unknown')

        if source is not None and target is not None:
            G.add_edge(source, target, type=rel_type)

    return G


def visualize_argument_graph(
    G: nx.DiGraph,
    output_path: str,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300,
    show_labels: bool = True,
    node_size_factor: float = 2000,
    edge_width: float = 2.0
) -> None:
    """
    Visualize an argument graph.

    Args:
        G: NetworkX directed graph
        output_path: Path to save the visualization
        figsize: Figure size (width, height)
        dpi: Resolution
        show_labels: Whether to show node labels
        node_size_factor: Factor to scale node sizes
        edge_width: Width of edges
    """
    # Create the figure
    plt.figure(figsize=figsize)

    # Define node colors based on component types
    type_colors = {
        'Claim': '#e41a1c',        # Red
        'Premise': '#377eb8',      # Blue
        'Major_Claim': '#4daf4a',  # Green
        'Backing': '#984ea3',      # Purple
        'Non_argumentative': '#999999',  # Gray
        'Unknown': '#ffa500'       # Orange
    }

    # Define edge colors based on relation types
    edge_colors = {
        'Support': '#66c2a5',      # Teal
        'Attack': '#fc8d62',       # Orange
        'None': '#8da0cb',         # Light blue
        'Unknown': '#cccccc'       # Light gray
    }

    # Get node types
    node_types = nx.get_node_attributes(G, 'type')

    # Define node colors
    node_colors = [type_colors.get(node_types.get(node, 'Unknown'), '#ffa500') for node in G.nodes()]

    # Get node sizes based on text length
    node_texts = nx.get_node_attributes(G, 'text')
    node_sizes = [len(node_texts.get(node, '')) * node_size_factor / 100 for node in G.nodes()]
    node_sizes = [max(size, node_size_factor / 2) for size in node_sizes]  # Minimum size

    # Get edge types
    edge_types = nx.get_edge_attributes(G, 'type')

    # Define edge colors
    edge_colors_list = [edge_colors.get(edge_types.get(edge, 'Unknown'), '#cccccc') for edge in G.edges()]

    # Use a spring layout for the graph
    pos = nx.spring_layout(G, seed=42, k=0.3)

    # Draw the graph
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8
    )

    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors_list,
        width=edge_width,
        alpha=0.7,
        arrowsize=20,
        arrowstyle='->'
    )

    # Draw labels if requested
    if show_labels:
        # Get node labels
        node_labels = nx.get_node_attributes(G, 'label')

        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=10,
            font_weight='bold'
        )

    # Set title
    plt.title(G.graph.get('title', 'Argumentation Structure'), fontsize=16)

    # Add a legend for node types
    node_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=node_type)
        for node_type, color in type_colors.items()
        if node_type in node_types.values()
    ]

    # Add a legend for edge types
    edge_legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=edge_type)
        for edge_type, color in edge_colors.items()
        if edge_type in edge_types.values()
    ]

    # Add the legends
    if node_legend_elements:
        plt.legend(
            handles=node_legend_elements,
            title="Component Types",
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.
        )

    if edge_legend_elements:
        plt.legend(
            handles=edge_legend_elements,
            title="Relation Types",
            loc='upper left',
            bbox_to_anchor=(1.05, 0.5),
            borderaxespad=0.
        )

    # Remove axis
    plt.axis('off')

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def visualize_text_with_components(
    text: str,
    components: List[Dict[str, Any]],
    output_path: str,
    highlight_colors: Optional[Dict[str, str]] = None,
    title: str = "Text with Argument Components",
    figsize: Tuple[int, int] = (12, len(text) // 80 + 5),  # Adjust height based on text length
    dpi: int = 300
) -> None:
    """
    Visualize text with highlighted argument components.

    Args:
        text: Full text
        components: List of components with start, end, and type information
        output_path: Path to save the visualization
        highlight_colors: Dictionary mapping component types to colors
        title: Title of the visualization
        figsize: Figure size (width, height)
        dpi: Resolution
    """
    # Default colors for component types
    if highlight_colors is None:
        highlight_colors = {
            'Claim': '#e41a1c',        # Red
            'Premise': '#377eb8',      # Blue
            'Major_Claim': '#4daf4a',  # Green
            'Backing': '#984ea3',      # Purple
            'Non_argumentative': '#999999',  # Gray
            'Unknown': '#ffa500'       # Orange
        }

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set the title
    ax.set_title(title, fontsize=16)

    # Display the text
    ax.text(
        0.05, 0.95,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        wrap=True
    )

    # Highlight components
    for component in components:
        component_type = component.get('type', 'Unknown')
        start = component.get('start')
        end = component.get('end')

        if start is not None and end is not None:
            # Calculate text position
            # This is a simplification - in practice, you'd need to handle line wrapping
            # and other complexities for accurate highlighting
            pos_x = 0.05
            pos_y = 0.95 - (start / len(text)) * 0.9

            # Calculate width and height
            width = (end - start) / len(text) * 0.9
            height = 0.03

            # Get color for component type
            color = highlight_colors.get(component_type, highlight_colors['Unknown'])

            # Add a rectangle to highlight the component
            rect = plt.Rectangle(
                (pos_x, pos_y),
                width, height,
                facecolor=to_rgba(color, 0.2),
                edgecolor=color,
                linewidth=2,
                transform=ax.transAxes
            )
            ax.add_patch(rect)

    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=component_type)
        for component_type, color in highlight_colors.items()
        if component_type in [comp.get('type', 'Unknown') for comp in components]
    ]

    ax.legend(
        handles=legend_elements,
        title="Component Types",
        loc='upper right',
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.
    )

    # Remove axis
    ax.axis('off')

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def visualize_predictions(
    predictions: List[Dict[str, Any]],
    output_dir: str,
    max_examples: int = 5
) -> None:
    """
    Visualize model predictions.

    Args:
        predictions: List of predictions
        output_dir: Directory to save visualizations
        max_examples: Maximum number of examples to visualize
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, "example_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Select a subset of examples to visualize
    examples_to_visualize = predictions[:max_examples]

    # Visualize each example
    for i, example in enumerate(examples_to_visualize):
        # Extract example ID and text
        example_id = example.get('id', f"example_{i}")
        text = example.get('text', '')

        # Skip examples with no text
        if not text:
            continue

        # Create components from predictions
        components = []

        # For simplicity, we'll treat the entire text as a single component
        # In practice, you would have more granular information about component boundaries
        component_type = example.get('predicted_component', 'Unknown')
        components.append({
            'type': component_type,
            'start': 0,
            'end': len(text),
            'text': text
        })

        # Extract relation predictions if available
        relations = []
        if 'relation_predictions' in example:
            for j, rel in enumerate(example['relation_predictions']):
                # Create relation
                relations.append({
                    'source': rel.get('source_token', 0),
                    'target': rel.get('target_token', 0),
                    'type': rel.get('predicted_relation', 'Unknown')
                })

        # Create a file name based on the example ID
        filename_base = f"{example_id}".replace(" ", "_").replace("/", "_")

        # Visualize text with components
        text_vis_path = os.path.join(vis_dir, f"{filename_base}_text.png")
        visualize_text_with_components(
            text=text,
            components=components,
            output_path=text_vis_path,
            title=f"Example {example_id}: {component_type}"
        )

        # Visualize argument graph if there are relations
        if relations:
            graph_vis_path = os.path.join(vis_dir, f"{filename_base}_graph.png")
            G = create_argument_graph(
                components=components,
                relations=relations,
                title=f"Example {example_id}: Argumentation Structure"
            )
            visualize_argument_graph(
                G=G,
                output_path=graph_vis_path
            )


if __name__ == "__main__":
    # Example usage
    # This code will only run if this module is executed directly
    # It demonstrates how to use the visualization functions

    # Create a simple example
    text = """
    Global warming is a serious threat to our planet.
    The Earth's temperature has risen by 1 degree Celsius since pre-industrial times.
    This is causing sea levels to rise and more extreme weather events.
    Some argue that natural climate cycles are the main cause, but scientific evidence strongly supports human activities as the primary driver.
    Therefore, we must take immediate action to reduce greenhouse gas emissions.
    """

    components = [
        {
            'type': 'Major_Claim',
            'start': 5,
            'end': 48,
            'text': 'Global warming is a serious threat to our planet.'
        },
        {
            'type': 'Premise',
            'start': 54,
            'end': 126,
            'text': 'The Earth\'s temperature has risen by 1 degree Celsius since pre-industrial times.'
        },
        {
            'type': 'Premise',
            'start': 132,
            'end': 194,
            'text': 'This is causing sea levels to rise and more extreme weather events.'
        },
        {
            'type': 'Premise',
            'start': 200,
            'end': 311,
            'text': 'Some argue that natural climate cycles are the main cause, but scientific evidence strongly supports human activities as the primary driver.'
        },
        {
            'type': 'Claim',
            'start': 317,
            'end': 384,
            'text': 'Therefore, we must take immediate action to reduce greenhouse gas emissions.'
        }
    ]

    relations = [
        {'source': 1, 'target': 0, 'type': 'Support'},
        {'source': 2, 'target': 0, 'type': 'Support'},
        {'source': 3, 'target': 0, 'type': 'Support'},
        {'source': 0, 'target': 4, 'type': 'Support'}
    ]

    # Create output directory
    os.makedirs("example_output", exist_ok=True)

    # Visualize text with components
    visualize_text_with_components(
        text=text,
        components=components,
        output_path="example_output/example_text.png",
        title="Example Text with Argument Components"
    )

    # Visualize argument graph
    G = create_argument_graph(
        components=components,
        relations=relations,
        title="Example Argumentation Structure"
    )
    visualize_argument_graph(
        G=G,
        output_path="example_output/example_graph.png"
    )

    print("Example visualizations created in 'example_output' directory.")