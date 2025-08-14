"""
NetworkVisualizer class for creating network visualizations with matplotlib backend.

This module provides functionality to visualize social networks with node sizes
based on centrality metrics, color coding for different node importance levels,
and various layout algorithms including spring, circular, and hierarchical layouts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from collections import defaultdict
import warnings


class NetworkVisualizer:
    """
    Creates network visualizations using matplotlib backend.
    
    Provides methods for visualizing networks with node sizes based on centrality,
    color coding for different node importance levels, and multiple layout algorithms.
    Supports both static and customizable network visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (50, 50), dpi: int = 100):
        """
        Initialize NetworkVisualizer.
        
        Args:
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        # Default styling parameters
        self.default_node_size = 300
        self.default_node_color = '#1f77b4'
        self.default_edge_color = '#666666'
        self.default_edge_width = 1.0
        self.default_font_size = 8
        
        # Color schemes for importance levels
        self.importance_colors = {
            'very_high': '#d62728',    # Red
            'high': '#ff7f0e',         # Orange  
            'medium': '#2ca02c',       # Green
            'low': '#1f77b4',          # Blue
            'very_low': '#9467bd'      # Purple
        }
        
    def visualize_network(self, 
                         graph: nx.Graph,
                         layout: str = 'spring',
                         node_size: Union[int, Dict[str, float], str] = None,
                         node_color: Union[str, Dict[str, str], str] = None,
                         edge_width: Union[float, Dict[Tuple[str, str], float]] = None,
                         show_labels: bool = False,
                         title: str = "Social Network Visualization",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a basic network visualization.
        
        Args:
            graph: NetworkX graph to visualize
            layout: Layout algorithm ('spring', 'circular', 'hierarchical', 'random')
            node_size: Node sizes (int for uniform, dict for per-node, or centrality metric name)
            node_color: Node colors (str for uniform, dict for per-node, or centrality metric name)
            edge_width: Edge widths (float for uniform, dict for per-edge)
            show_labels: Whether to show node labels
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object
            
        Raises:
            ValueError: If layout algorithm is not supported
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph")
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate layout positions
        pos = self._calculate_layout(graph, layout)
        
        # Process node sizes
        node_sizes = self._process_node_sizes(graph, node_size)
        
        # Process node colors
        node_colors = self._process_node_colors(graph, node_color)
        
        # Process edge widths
        edge_widths = self._process_edge_widths(graph, edge_width)
        
        # Draw edges first (so they appear behind nodes)
        nx.draw_networkx_edges(
            graph, pos, ax=ax,
            width=edge_widths,
            edge_color=self.default_edge_color,
            alpha=0.6
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8
        )
        
        # Draw labels if requested
        if show_labels:
            # Use username if available, otherwise use node ID
            labels = {}
            for node in graph.nodes():
                if 'username' in graph.nodes[node] and graph.nodes[node]['username']:
                    labels[node] = graph.nodes[node]['username']
                else:
                    labels[node] = str(node)
                    
            nx.draw_networkx_labels(
                graph, pos, labels, ax=ax,
                font_size=self.default_font_size,
                font_weight='bold'
            )
        
        # Set title and remove axes
        ax.set_title(title, fontsize=30, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add network statistics as text
        stats_text = self._generate_stats_text(graph)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved network visualization to {save_path}")
            
        self.logger.info(f"Created network visualization with {graph.number_of_nodes()} nodes")
        return fig
        
    def visualize_centrality(self, 
                           graph: nx.Graph,
                           centrality_scores: Dict[str, float],
                           centrality_type: str,
                           layout: str = 'spring',
                           fixed_node_size: int = 300,
                           show_top_k: Optional[int] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize network with fixed node sizes and centrality-based colors.
        
        Args:
            graph: NetworkX graph to visualize
            centrality_scores: Dictionary mapping node IDs to centrality scores
            centrality_type: Type of centrality being visualized
            layout: Layout algorithm to use
            fixed_node_size: Fixed size for all nodes (default: 300)
            show_top_k: Only label top k nodes by centrality (optional)
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate layout positions
        pos = self._calculate_layout(graph, layout)
        
        # Use fixed node size for all nodes
        node_sizes = [fixed_node_size] * graph.number_of_nodes()
        
        # Create color map based on centrality importance levels
        node_colors = self._create_centrality_colors(centrality_scores)
        
        # Draw edges with better visibility
        nx.draw_networkx_edges(
            graph, pos, ax=ax,
            width=1.5,  # Increased edge width for better visibility
            edge_color=self.default_edge_color,
            alpha=0.7,  # Increased alpha for better visibility
            edge_cmap=plt.cm.Greys  # Use colormap for edge weights if available
        )
        
        # Draw nodes with fixed size
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9  # Increased alpha for better visibility
        )
        
        # Draw labels for top nodes if requested
        if show_top_k:
            top_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:show_top_k]
            labels = {}
            for node_id, _ in top_nodes:
                if 'username' in graph.nodes[node_id] and graph.nodes[node_id]['username']:
                    labels[node_id] = graph.nodes[node_id]['username']
                else:
                    labels[node_id] = str(node_id)
                    
            nx.draw_networkx_labels(
                graph, pos, labels, ax=ax,
                font_size=self.default_font_size,
                font_weight='bold'
            )
        
        # Set title
        title = f"{centrality_type.title()} Centrality Visualization"
        ax.set_title(title, fontsize=30, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add centrality statistics
        stats_text = self._generate_centrality_stats_text(centrality_scores, centrality_type)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add color legend
        self._add_centrality_legend(ax)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved centrality visualization to {save_path}")
            
        self.logger.info(f"Created {centrality_type} centrality visualization")
        return fig

    def visualize_propagation(self, 
                            graph: nx.Graph,
                            propagation_result: Dict[str, Any],
                            layout: str = 'spring',
                            show_labels: bool = False,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize influence propagation through the network.
        
        Args:
            graph: NetworkX graph to visualize
            propagation_result: Result from influence propagation simulation
            layout: Layout algorithm to use
            show_labels: Whether to show node labels
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        if 'propagation_history' not in propagation_result:
            raise ValueError("Propagation result must include history for visualization")
        
        history = propagation_result['propagation_history']
        seed_nodes = propagation_result.get('seed_nodes', [])
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate layout positions
        pos = self._calculate_layout(graph, layout)
        
        # Get final state for visualization
        final_step = history[-1] if history else {}
        activated_nodes = final_step.get('activated', [])
        
        # Color scheme for propagation states
        colors = {
            'seed': '#d62728',        # Red for seed nodes
            'activated': '#2ca02c',   # Green for activated nodes
            'inactive': '#cccccc'     # Gray for inactive nodes
        }
        
        # Determine node colors
        node_colors = []
        for node in graph.nodes():
            if node in seed_nodes:
                node_colors.append(colors['seed'])
            elif node in activated_nodes:
                node_colors.append(colors['activated'])
            else:
                node_colors.append(colors['inactive'])
        
        # Draw edges with good visibility
        nx.draw_networkx_edges(
            graph, pos, ax=ax,
            width=1.0,
            edge_color=self.default_edge_color,
            alpha=0.6
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_size=self.default_node_size,
            node_color=node_colors,
            alpha=0.8
        )
        
        # Draw labels if requested
        if show_labels:
            labels = {}
            for node in graph.nodes():
                if 'username' in graph.nodes[node] and graph.nodes[node]['username']:
                    labels[node] = graph.nodes[node]['username']
                else:
                    labels[node] = str(node)
                    
            nx.draw_networkx_labels(
                graph, pos, labels, ax=ax,
                font_size=self.default_font_size,
                font_weight='bold'
            )
        
        # Set title
        title = f"Influence Propagation Visualization\n{len(activated_nodes)}/{graph.number_of_nodes()} nodes activated"
        ax.set_title(title, fontsize=30, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add propagation statistics
        # Add graph info to propagation result for stats generation
        propagation_result['total_nodes'] = graph.number_of_nodes()
        stats_text = self._generate_propagation_stats_text(propagation_result)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add propagation legend
        self._add_propagation_legend(ax, colors)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved propagation visualization to {save_path}")
            
        self.logger.info(f"Created propagation visualization")
        return fig

    def visualize_propagation_timeline(self, 
                                     graph: nx.Graph,
                                     propagation_result: Dict[str, Any],
                                     layout: str = 'spring',
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create timeline visualization showing propagation process over time.
        
        Args:
            graph: NetworkX graph to visualize
            propagation_result: Result from influence propagation simulation
            layout: Layout algorithm to use
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object with subplots for each time step
        """
        if 'propagation_history' not in propagation_result:
            raise ValueError("Propagation result must include history for timeline visualization")
        
        history = propagation_result['propagation_history']
        seed_nodes = propagation_result.get('seed_nodes', [])
        
        # Calculate layout positions
        pos = self._calculate_layout(graph, layout)
        
        # Create subplots for different time steps
        num_steps = len(history)
        cols = min(4, num_steps)
        rows = (num_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(self.figsize[0] * cols, self.figsize[1] * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Color scheme for propagation states
        colors = {
            'seed': '#d62728',           # Red for seed nodes
            'newly_activated': '#ff7f0e', # Orange for newly activated
            'activated': '#2ca02c',       # Green for previously activated
            'inactive': '#cccccc'         # Gray for inactive
        }
        
        for i, step_data in enumerate(history):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Determine node colors for this step
            node_colors = []
            for node in graph.nodes():
                if node in seed_nodes:
                    if step_data['step'] == 0:
                        node_colors.append(colors['seed'])
                    else:
                        node_colors.append(colors['activated'])
                elif node in step_data.get('newly_activated', []):
                    node_colors.append(colors['newly_activated'])
                elif node in step_data.get('activated', []):
                    node_colors.append(colors['activated'])
                else:
                    node_colors.append(colors['inactive'])
            
            # Draw network for this step
            nx.draw_networkx_edges(
                graph, pos, ax=ax,
                width=0.8,
                edge_color=self.default_edge_color,
                alpha=0.5
            )
            
            nx.draw_networkx_nodes(
                graph, pos, ax=ax,
                node_size=200,
                node_color=node_colors,
                alpha=0.8
            )
            
            ax.set_title(f"Step {step_data['step']}\n"
                        f"Activated: {step_data.get('total_activated', 0)}")
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(history), len(axes)):
            axes[i].axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['seed'], 
                      markersize=10, label='Seed nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['newly_activated'], 
                      markersize=10, label='Newly activated'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['activated'], 
                      markersize=10, label='Previously activated'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['inactive'], 
                      markersize=10, label='Inactive')
        ]
        
        # Add legend to the last subplot
        if len(history) > 0:
            axes[min(len(history) - 1, len(axes) - 1)].legend(handles=legend_elements, 
                                                              loc='upper right', fontsize=8)
        
        plt.suptitle("Influence Propagation Timeline", fontsize=30, fontweight='bold')
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved propagation timeline to {save_path}")
            
        self.logger.info(f"Created propagation timeline visualization with {len(history)} steps")
        return fig
        
    def visualize_communities(self, 
                            graph: nx.Graph,
                            communities: Dict[int, List[str]],
                            layout: str = 'spring',
                            show_community_labels: bool = True,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize network with communities highlighted using distinct colors.
        
        Args:
            graph: NetworkX graph to visualize
            communities: Dictionary mapping community IDs to lists of node IDs
            layout: Layout algorithm to use
            show_community_labels: Whether to show community ID labels
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate layout positions
        pos = self._calculate_layout(graph, layout)
        
        # Generate distinct colors for communities
        community_colors = self._generate_community_colors(len(communities))
        
        # Create node color mapping
        node_colors = []
        node_to_community = {}
        
        for community_id, node_list in communities.items():
            for node in node_list:
                node_to_community[node] = community_id
                
        for node in graph.nodes():
            community_id = node_to_community.get(node, -1)  # -1 for uncategorized
            if community_id >= 0:
                node_colors.append(community_colors[community_id % len(community_colors)])
            else:
                node_colors.append('#cccccc')  # Gray for uncategorized
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, pos, ax=ax,
            width=0.5,
            edge_color=self.default_edge_color,
            alpha=0.3
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_size=self.default_node_size,
            node_color=node_colors,
            alpha=0.8
        )
        
        # Draw community boundaries
        self._draw_community_boundaries(ax, pos, communities, node_to_community)
        
        # Add community labels if requested
        if show_community_labels:
            self._add_community_labels(ax, pos, communities)
        
        # Set title
        title = f"Community Structure Visualization ({len(communities)} communities)"
        ax.set_title(title, fontsize=30, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add community statistics
        stats_text = self._generate_community_stats_text(communities, graph)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved community visualization to {save_path}")
            
        self.logger.info(f"Created community visualization with {len(communities)} communities")
        return fig
        
    def create_multi_layout_comparison(self, 
                                     graph: nx.Graph,
                                     layouts: List[str] = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparison visualization showing the same network with different layouts.
        
        Args:
            graph: NetworkX graph to visualize
            layouts: List of layout algorithms to compare
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object with subplots for each layout
        """
        if layouts is None:
            layouts = ['spring', 'circular', 'random', 'shell']
            
        # Create subplots
        n_layouts = len(layouts)
        cols = 2
        rows = (n_layouts + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(self.figsize[0] * cols, self.figsize[1] * rows))
        if n_layouts == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Create visualization for each layout
        for i, layout in enumerate(layouts):
            ax = axes[i]
            
            try:
                # Calculate layout positions
                pos = self._calculate_layout(graph, layout)
                
                # Draw network
                nx.draw_networkx_edges(
                    graph, pos, ax=ax,
                    width=0.5,
                    edge_color=self.default_edge_color,
                    alpha=0.6
                )
                
                nx.draw_networkx_nodes(
                    graph, pos, ax=ax,
                    node_size=100,
                    node_color=self.default_node_color,
                    alpha=0.8
                )
                
                ax.set_title(f"{layout.title()} Layout", fontsize=14, fontweight='bold')
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error with {layout} layout:\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{layout.title()} Layout (Error)", fontsize=14)
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_layouts, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Layout Algorithm Comparison", fontsize=30, fontweight='bold')
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved layout comparison to {save_path}")
            
        self.logger.info(f"Created layout comparison with {len(layouts)} layouts")
        return fig        

    def _calculate_layout(self, graph: nx.Graph, layout: str) -> Dict[str, Tuple[float, float]]:
        """
        Calculate node positions using specified layout algorithm.
        
        Args:
            graph: NetworkX graph
            layout: Layout algorithm name
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
            
        Raises:
            ValueError: If layout algorithm is not supported
        """
        layout = layout.lower()
        
        if layout == 'spring':
            # Spring layout with improved parameters for better visualization
            pos = nx.spring_layout(graph, k=1/np.sqrt(graph.number_of_nodes()), 
                                 iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'hierarchical':
            # Use shell layout as hierarchical approximation
            try:
                # Try to create shells based on degree
                degree_dict = dict(graph.degree())
                max_degree = max(degree_dict.values()) if degree_dict else 1
                
                shells = []
                for i in range(max_degree + 1):
                    shell = [node for node, degree in degree_dict.items() if degree == i]
                    if shell:
                        shells.append(shell)
                        
                if len(shells) > 1:
                    pos = nx.shell_layout(graph, nlist=shells)
                else:
                    # Fallback to spring layout
                    pos = nx.spring_layout(graph, seed=42)
            except:
                # Fallback to spring layout if shell layout fails
                pos = nx.spring_layout(graph, seed=42)
        elif layout == 'random':
            pos = nx.random_layout(graph, seed=42)
        elif layout == 'shell':
            pos = nx.shell_layout(graph)
        elif layout == 'kamada_kawai':
            try:
                pos = nx.kamada_kawai_layout(graph)
            except:
                # Fallback to spring layout for disconnected graphs
                pos = nx.spring_layout(graph, seed=42)
        else:
            raise ValueError(f"Unsupported layout algorithm: {layout}. "
                           f"Supported layouts: spring, circular, hierarchical, random, shell, kamada_kawai")
        
        return pos
        
    def _process_node_sizes(self, graph: nx.Graph, node_size: Union[int, Dict[str, float], str]) -> List[float]:
        """
        Process node size parameter into list of sizes for each node.
        
        Args:
            graph: NetworkX graph
            node_size: Size specification (int, dict, or centrality metric name)
            
        Returns:
            List of node sizes in graph node order
        """
        if node_size is None:
            return [self.default_node_size] * graph.number_of_nodes()
        elif isinstance(node_size, int):
            return [node_size] * graph.number_of_nodes()
        elif isinstance(node_size, dict):
            return [node_size.get(node, self.default_node_size) for node in graph.nodes()]
        elif isinstance(node_size, str):
            # Assume it's a centrality metric name stored in node attributes
            sizes = []
            for node in graph.nodes():
                centrality = graph.nodes[node].get(node_size, 0)
                size = max(50, centrality * 1000)  # Scale and set minimum
                sizes.append(size)
            return sizes
        else:
            return [self.default_node_size] * graph.number_of_nodes()
            
    def _process_node_colors(self, graph: nx.Graph, node_color: Union[str, Dict[str, str], str]) -> List[str]:
        """
        Process node color parameter into list of colors for each node.
        
        Args:
            graph: NetworkX graph
            node_color: Color specification (str, dict, or centrality metric name)
            
        Returns:
            List of node colors in graph node order
        """
        if node_color is None:
            return [self.default_node_color] * graph.number_of_nodes()
        elif isinstance(node_color, str) and node_color.startswith('#'):
            # Single color for all nodes
            return [node_color] * graph.number_of_nodes()
        elif isinstance(node_color, dict):
            return [node_color.get(node, self.default_node_color) for node in graph.nodes()]
        elif isinstance(node_color, str):
            # Assume it's a centrality metric name stored in node attributes
            colors = []
            centrality_values = []
            
            # Collect centrality values
            for node in graph.nodes():
                centrality = graph.nodes[node].get(node_color, 0)
                centrality_values.append(centrality)
                
            # Create color mapping based on centrality percentiles
            if centrality_values:
                colors = self._create_centrality_colors_from_values(centrality_values)
            else:
                colors = [self.default_node_color] * graph.number_of_nodes()
                
            return colors
        else:
            return [self.default_node_color] * graph.number_of_nodes()
            
    def _process_edge_widths(self, graph: nx.Graph, edge_width: Union[float, Dict[Tuple[str, str], float]]) -> List[float]:
        """
        Process edge width parameter into list of widths for each edge.
        
        Args:
            graph: NetworkX graph
            edge_width: Width specification (float or dict)
            
        Returns:
            List of edge widths in graph edge order
        """
        if edge_width is None:
            # Use edge weights if available
            widths = []
            for u, v, data in graph.edges(data=True):
                weight = data.get('weight', 1.0)
                width = max(0.5, min(5.0, weight))  # Clamp between 0.5 and 5.0
                widths.append(width)
            return widths
        elif isinstance(edge_width, (int, float)):
            return [edge_width] * graph.number_of_edges()
        elif isinstance(edge_width, dict):
            widths = []
            for u, v in graph.edges():
                # Try both edge directions
                width = edge_width.get((u, v), edge_width.get((v, u), self.default_edge_width))
                widths.append(width)
            return widths
        else:
            return [self.default_edge_width] * graph.number_of_edges()
            
    def _create_centrality_colors(self, centrality_scores: Dict[str, float]) -> List[str]:
        """
        Create color mapping based on centrality importance levels.
        
        Args:
            centrality_scores: Dictionary mapping node IDs to centrality scores
            
        Returns:
            List of colors corresponding to importance levels
        """
        if not centrality_scores:
            return []
            
        values = list(centrality_scores.values())
        return self._create_centrality_colors_from_values(values)
        
    def _create_centrality_colors_from_values(self, values: List[float]) -> List[str]:
        """
        Create color mapping from centrality values based on percentiles.
        
        Args:
            values: List of centrality values
            
        Returns:
            List of colors corresponding to importance levels
        """
        if not values:
            return []
            
        # Calculate percentile thresholds
        p20 = 0.00001
        p40 = 0.0001
        p60 = 0.001
        p80 = 0.01
        
        colors = []
        for value in values:
            if value >= p80:
                colors.append(self.importance_colors['very_high'])
            elif value >= p60:
                colors.append(self.importance_colors['high'])
            elif value >= p40:
                colors.append(self.importance_colors['medium'])
            elif value >= p20:
                colors.append(self.importance_colors['low'])
            else:
                colors.append(self.importance_colors['very_low'])
                
        return colors
        
    def _generate_community_colors(self, n_communities: int) -> List[str]:
        """
        Generate distinct colors for communities.
        
        Args:
            n_communities: Number of communities
            
        Returns:
            List of distinct colors
        """
        # Use matplotlib's tab10 colormap for up to 10 communities
        if n_communities <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_communities]
            return [mcolors.rgb2hex(color) for color in colors]
        else:
            # Use hsv colormap for more communities
            colors = plt.cm.hsv(np.linspace(0, 1, n_communities))
            return [mcolors.rgb2hex(color) for color in colors]
            
    def _draw_community_boundaries(self, ax: plt.Axes, pos: Dict, communities: Dict, node_to_community: Dict):
        """
        Draw boundaries around communities.
        
        Args:
            ax: Matplotlib axes
            pos: Node positions
            communities: Community mapping
            node_to_community: Node to community mapping
        """
        for community_id, nodes in communities.items():
            if len(nodes) < 3:  # Skip communities with too few nodes
                continue
                
            # Get positions of nodes in this community
            community_pos = [pos[node] for node in nodes if node in pos]
            
            if len(community_pos) < 3:
                continue
                
            # Calculate convex hull
            try:
                from scipy.spatial import ConvexHull
                points = np.array(community_pos)
                hull = ConvexHull(points)
                
                # Create polygon patch
                hull_points = points[hull.vertices]
                polygon = patches.Polygon(hull_points, fill=False, edgecolor='black', 
                                        linewidth=2, linestyle='--', alpha=0.5)
                ax.add_patch(polygon)
            except:
                # Fallback: draw circle around community center
                x_coords = [p[0] for p in community_pos]
                y_coords = [p[1] for p in community_pos]
                center_x, center_y = np.mean(x_coords), np.mean(y_coords)
                
                # Calculate radius as max distance from center
                distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) 
                           for x, y in community_pos]
                radius = max(distances) * 1.2  # Add some padding
                
                circle = patches.Circle((center_x, center_y), radius, fill=False, 
                                      edgecolor='black', linewidth=2, linestyle='--', alpha=0.5)
                ax.add_patch(circle)
                
    def _add_community_labels(self, ax: plt.Axes, pos: Dict, communities: Dict):
        """
        Add community ID labels to the visualization.
        
        Args:
            ax: Matplotlib axes
            pos: Node positions
            communities: Community mapping
        """
        for community_id, nodes in communities.items():
            if not nodes:
                continue
                
            # Calculate community center
            community_pos = [pos[node] for node in nodes if node in pos]
            if community_pos:
                center_x = np.mean([p[0] for p in community_pos])
                center_y = np.mean([p[1] for p in community_pos])
                
                # Add label
                ax.text(center_x, center_y, f"C{community_id}", 
                       fontsize=12, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                       
    def _add_centrality_legend(self, ax: plt.Axes):
        """
        Add color legend for centrality importance levels.
        
        Args:
            ax: Matplotlib axes
        """
        legend_elements = []
        for level, color in self.importance_colors.items():
            label = level.replace('_', ' ').title()
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, label=label))
        
        ax.legend(handles=legend_elements, loc='upper right', title='Importance Level')
        
    def _generate_stats_text(self, graph: nx.Graph) -> str:
        """
        Generate network statistics text for display.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Formatted statistics string
        """
        stats = [
            f"Nodes: {graph.number_of_nodes()}",
            f"Edges: {graph.number_of_edges()}",
            f"Density: {nx.density(graph):.3f}",
            f"Components: {nx.number_connected_components(graph)}"
        ]
        
        if graph.number_of_edges() > 0:
            weights = [data.get('weight', 1) for _, _, data in graph.edges(data=True)]
            stats.append(f"Avg Edge Weight: {np.mean(weights):.2f}")
            
        return "\n".join(stats)
        
    def _generate_centrality_stats_text(self, centrality_scores: Dict[str, float], centrality_type: str) -> str:
        """
        Generate centrality statistics text for display.
        
        Args:
            centrality_scores: Centrality scores
            centrality_type: Type of centrality
            
        Returns:
            Formatted statistics string
        """
        if not centrality_scores:
            return f"{centrality_type.title()} Centrality\nNo data available"
            
        values = list(centrality_scores.values())
        stats = [
            f"{centrality_type.title()} Centrality",
            f"Mean: {np.mean(values):.4f}",
            f"Std: {np.std(values):.4f}",
            f"Max: {np.max(values):.4f}",
            f"Min: {np.min(values):.4f}"
        ]
        
        return "\n".join(stats)
        
    def _generate_community_stats_text(self, communities: Dict[int, List[str]], graph: nx.Graph) -> str:
        """
        Generate community statistics text for display.
        
        Args:
            communities: Community mapping
            graph: NetworkX graph
            
        Returns:
            Formatted statistics string
        """
        if not communities:
            return "Communities\nNo data available"
            
        community_sizes = [len(nodes) for nodes in communities.values()]
        stats = [
            f"Communities: {len(communities)}",
            f"Avg Size: {np.mean(community_sizes):.1f}",
            f"Largest: {max(community_sizes)}",
            f"Smallest: {min(community_sizes)}"
        ]
        
        # Calculate modularity if possible
        try:
            # Create community assignment for modularity calculation
            community_assignment = {}
            for comm_id, nodes in communities.items():
                for node in nodes:
                    community_assignment[node] = comm_id
                    
            modularity = nx.algorithms.community.modularity(graph, communities.values())
            stats.append(f"Modularity: {modularity:.3f}")
        except:
            pass
            
        return "\n".join(stats)

    def _generate_propagation_stats_text(self, propagation_result: Dict[str, Any]) -> str:
        """
        Generate influence propagation statistics text for display.
        
        Args:
            propagation_result: Result from influence propagation simulation
            
        Returns:
            Formatted statistics string
        """
        history = propagation_result['propagation_history']
        final_step = history[-1] if history else {}
        total_activated = final_step.get('total_activated', 0)
        final_activated = final_step.get('activated', [])
        
        # Get total nodes from the propagation result if available
        total_nodes = propagation_result.get('total_nodes', 0)
        if not total_nodes and 'graph_info' in propagation_result:
            total_nodes = propagation_result['graph_info'].get('num_nodes', 0)
        
        stats = [
            f"Total Nodes: {total_nodes}",
            f"Total Activated: {total_activated}",
            f"Final Activated: {len(final_activated)}"
        ]
        
        return "\n".join(stats)

    def _add_propagation_legend(self, ax: plt.Axes, colors: Dict[str, str]):
        """
        Add legend for influence propagation visualization.
        
        Args:
            ax: Matplotlib axes
            colors: Dictionary of colors for different states
        """
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['seed'], 
                      markersize=10, label='Seed nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['activated'], 
                      markersize=10, label='Activated nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)    

    def visualize_influence_propagation_with_timeline(self, 
                                                    graph: nx.Graph,
                                                    influence_propagator,
                                                    seed_nodes: List[str],
                                                    max_steps: int = 10,
                                                    layout: str = 'spring',
                                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive influence propagation visualization with timeline.
        
        Args:
            graph: NetworkX graph
            influence_propagator: InfluencePropagator instance
            seed_nodes: List of seed nodes for propagation
            max_steps: Maximum propagation steps
            layout: Network layout algorithm
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info(f"Creating influence propagation timeline with {len(seed_nodes)} seed nodes")
        
        try:
            # Run propagation simulation
            propagation_result = influence_propagator.simulate_propagation(
                graph=graph,
                seed_nodes=seed_nodes,
                max_steps=max_steps,
                track_history=True
            )
            
            # Use the influence propagator's timeline visualization
            fig = influence_propagator.visualize_propagation_timeline(
                graph=graph,
                propagation_result=propagation_result,
                layout=layout,
                save_path=save_path,
                figsize=self.figsize
            )
            
            self.logger.info("Influence propagation timeline visualization created successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create influence propagation timeline: {e}")
            # Fallback to basic propagation visualization
            return self.visualize_propagation(graph, {'seed_nodes': seed_nodes}, layout, save_path)
    
    def visualize_influence_comparison_strategies(self,
                                                graph: nx.Graph,
                                                influence_propagator,
                                                centrality_scores: Dict[str, Dict[str, float]],
                                                seed_counts: List[int] = [1, 3, 5, 10],
                                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization comparing different seed selection strategies.
        
        Args:
            graph: NetworkX graph
            influence_propagator: InfluencePropagator instance
            centrality_scores: Dictionary of centrality metrics
            seed_counts: List of seed counts to test
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating influence strategy comparison visualization")
        
        try:
            # Evaluate different seed selection strategies
            influence_evaluation = influence_propagator.evaluate_seed_influence(
                graph=graph,
                centrality_scores=centrality_scores,
                seed_counts=seed_counts,
                num_simulations=10
            )
            
            # Use the influence propagator's comparison visualization
            fig = influence_propagator.visualize_influence_comparison(
                influence_evaluation=influence_evaluation,
                save_path=save_path,
                figsize=self.figsize
            )
            
            self.logger.info("Influence strategy comparison visualization created successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create influence strategy comparison: {e}")
            # Create a simple fallback visualization
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f"Influence comparison failed: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def create_influence_dashboard(self,
                                 graph: nx.Graph,
                                 influence_propagator,
                                 centrality_scores: Dict[str, Dict[str, float]],
                                 communities: Optional[Dict[int, List[str]]] = None,
                                 seed_nodes: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive influence analysis dashboard.
        
        Args:
            graph: NetworkX graph
            influence_propagator: InfluencePropagator instance
            centrality_scores: Dictionary of centrality metrics
            communities: Community detection results
            seed_nodes: Specific seed nodes for propagation simulation
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object with multiple subplots
        """
        self.logger.info("Creating influence analysis dashboard")
        
        try:
            # Create a large figure with multiple subplots
            fig = plt.figure(figsize=(20, 16))
            
            # Subplot 1: Network with centrality-based node sizes
            ax1 = plt.subplot(2, 3, 1)
            degree_centrality = centrality_scores.get('degree', {})
            if degree_centrality:
                self._plot_network_on_axis(ax1, graph, 
                                         node_size=degree_centrality,
                                         title="Network with Degree Centrality")
            
            # Subplot 2: Network with communities
            ax2 = plt.subplot(2, 3, 2)
            if communities:
                self._plot_communities_on_axis(ax2, graph, communities,
                                             title="Community Structure")
            else:
                self._plot_network_on_axis(ax2, graph, title="Network Structure")
            
            # Subplot 3: Influence propagation simulation
            ax3 = plt.subplot(2, 3, 3)
            if seed_nodes:
                try:
                    propagation_result = influence_propagator.simulate_propagation(
                        graph=graph, seed_nodes=seed_nodes, max_steps=5
                    )
                    self._plot_propagation_on_axis(ax3, graph, propagation_result,
                                                 title="Influence Propagation")
                except Exception as e:
                    ax3.text(0.5, 0.5, f"Propagation failed: {str(e)}", 
                           ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title("Influence Propagation (Failed)")
            else:
                ax3.text(0.5, 0.5, "No seed nodes specified", 
                       ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title("Influence Propagation")
            
            # Subplot 4: Centrality comparison
            ax4 = plt.subplot(2, 3, 4)
            self._plot_centrality_comparison(ax4, centrality_scores)
            
            # Subplot 5: Influence metrics
            ax5 = plt.subplot(2, 3, 5)
            try:
                influence_evaluation = influence_propagator.evaluate_seed_influence(
                    graph=graph, centrality_scores=centrality_scores,
                    seed_counts=[1, 3, 5], num_simulations=5
                )
                self._plot_influence_metrics(ax5, influence_evaluation)
            except Exception as e:
                ax5.text(0.5, 0.5, f"Influence evaluation failed: {str(e)}", 
                       ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title("Influence Metrics (Failed)")
            
            # Subplot 6: Network statistics
            ax6 = plt.subplot(2, 3, 6)
            self._plot_network_statistics(ax6, graph, centrality_scores, communities)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Influence dashboard saved to {save_path}")
            
            self.logger.info("Influence analysis dashboard created successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create influence dashboard: {e}")
            # Create a simple fallback
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f"Dashboard creation failed: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def _plot_network_on_axis(self, ax: plt.Axes, graph: nx.Graph, 
                            node_size: Union[int, Dict[str, float]] = None,
                            node_color: Union[str, Dict[str, str]] = None,
                            title: str = "Network"):
        """Helper method to plot network on a specific axis."""
        pos = self._calculate_layout(graph, 'spring')
        
        # Process node attributes
        sizes = self._process_node_sizes(graph, node_size or self.default_node_size)
        colors = self._process_node_colors(graph, node_color or self.default_node_color)
        
        # Draw network
        nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color=colors, ax=ax)
        nx.draw_networkx_edges(graph, pos, edge_color=self.default_edge_color, ax=ax)
        
        ax.set_title(title)
        ax.axis('off')
    
    def _plot_communities_on_axis(self, ax: plt.Axes, graph: nx.Graph, 
                                communities: Dict[int, List[str]], title: str = "Communities"):
        """Helper method to plot communities on a specific axis."""
        pos = self._calculate_layout(graph, 'spring')
        
        # Generate community colors
        community_colors = self._generate_community_colors(len(communities))
        
        # Create node color mapping
        node_colors = {}
        for i, (comm_id, nodes) in enumerate(communities.items()):
            color = community_colors[i % len(community_colors)]
            for node in nodes:
                if node in graph.nodes():
                    node_colors[node] = color
        
        colors = self._process_node_colors(graph, node_colors)
        
        # Draw network
        nx.draw_networkx_nodes(graph, pos, node_color=colors, ax=ax)
        nx.draw_networkx_edges(graph, pos, edge_color=self.default_edge_color, ax=ax)
        
        ax.set_title(title)
        ax.axis('off')
    
    def _plot_propagation_on_axis(self, ax: plt.Axes, graph: nx.Graph, 
                                propagation_result: Dict[str, Any], title: str = "Propagation"):
        """Helper method to plot propagation result on a specific axis."""
        pos = self._calculate_layout(graph, 'spring')
        
        # Create node colors based on propagation state
        node_colors = {}
        seed_nodes = propagation_result.get('seed_nodes', [])
        influenced_nodes = propagation_result.get('influenced_nodes', [])
        
        for node in graph.nodes():
            if node in seed_nodes:
                node_colors[node] = '#ff4444'  # Red for seed nodes
            elif node in influenced_nodes:
                node_colors[node] = '#ffaa44'  # Orange for influenced nodes
            else:
                node_colors[node] = '#cccccc'  # Gray for uninfluenced nodes
        
        colors = self._process_node_colors(graph, node_colors)
        
        # Draw network
        nx.draw_networkx_nodes(graph, pos, node_color=colors, ax=ax)
        nx.draw_networkx_edges(graph, pos, edge_color=self.default_edge_color, ax=ax)
        
        ax.set_title(title)
        ax.axis('off')
    
    def _plot_centrality_comparison(self, ax: plt.Axes, centrality_scores: Dict[str, Dict[str, float]]):
        """Helper method to plot centrality comparison."""
        if not centrality_scores:
            ax.text(0.5, 0.5, "No centrality data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Centrality Comparison")
            return
        
        # Get top nodes for each centrality measure
        centrality_names = list(centrality_scores.keys())
        
        # Create bar plot
        x_pos = np.arange(len(centrality_names))
        means = []
        
        for centrality_name in centrality_names:
            scores = list(centrality_scores[centrality_name].values())
            if scores:
                means.append(np.mean(scores))
            else:
                means.append(0)
        
        bars = ax.bar(x_pos, means, alpha=0.7)
        ax.set_xlabel('Centrality Measures')
        ax.set_ylabel('Average Score')
        ax.set_title('Centrality Measures Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name.replace('_', ' ').title() for name in centrality_names], rotation=45)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{mean:.3f}', ha='center', va='bottom')
    
    def _plot_influence_metrics(self, ax: plt.Axes, influence_evaluation: Dict[str, Any]):
        """Helper method to plot influence metrics."""
        if not influence_evaluation:
            ax.text(0.5, 0.5, "No influence data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Influence Metrics")
            return
        
        # Extract data for plotting
        strategies = list(influence_evaluation.keys())
        seed_counts = set()
        for strategy_data in influence_evaluation.values():
            seed_counts.update(strategy_data.keys())
        seed_counts = sorted(seed_counts)
        
        # Plot influence vs seed count for each strategy
        for strategy in strategies:
            means = []
            x_vals = []
            
            for num_seeds in seed_counts:
                if num_seeds in influence_evaluation[strategy]:
                    data = influence_evaluation[strategy][num_seeds]
                    means.append(data['mean_influence'])
                    x_vals.append(num_seeds)
            
            if means:
                ax.plot(x_vals, means, marker='o', label=strategy.replace('_', ' ').title())
        
        ax.set_xlabel('Number of Seed Nodes')
        ax.set_ylabel('Mean Influence')
        ax.set_title('Influence vs Seed Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_network_statistics(self, ax: plt.Axes, graph: nx.Graph, 
                               centrality_scores: Dict[str, Dict[str, float]],
                               communities: Optional[Dict[int, List[str]]]):
        """Helper method to plot network statistics."""
        ax.axis('off')
        
        # Gather statistics
        stats = []
        stats.append(f"Nodes: {graph.number_of_nodes()}")
        stats.append(f"Edges: {graph.number_of_edges()}")
        stats.append(f"Density: {nx.density(graph):.4f}")
        
        if nx.is_connected(graph):
            stats.append(f"Avg Path Length: {nx.average_shortest_path_length(graph):.2f}")
            stats.append(f"Diameter: {nx.diameter(graph)}")
        else:
            stats.append("Graph is disconnected")
        
        stats.append(f"Clustering: {nx.average_clustering(graph):.4f}")
        
        if communities:
            stats.append(f"Communities: {len(communities)}")
            avg_comm_size = np.mean([len(nodes) for nodes in communities.values()])
            stats.append(f"Avg Community Size: {avg_comm_size:.1f}")
        
        if centrality_scores:
            stats.append(f"Centrality Measures: {len(centrality_scores)}")
        
        # Display statistics
        stats_text = "\n".join(stats)
        ax.text(0.1, 0.9, "Network Statistics:", transform=ax.transAxes, 
               fontsize=12, fontweight='bold')
        ax.text(0.1, 0.8, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))