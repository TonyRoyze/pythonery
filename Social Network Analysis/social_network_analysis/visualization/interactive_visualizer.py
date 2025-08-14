"""
InteractiveVisualizer class for creating interactive network visualizations using Plotly.

This module provides functionality to create dynamic, interactive visualizations
with zoom and pan capabilities for large networks, and interactive centrality
and community exploration features.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from collections import defaultdict
import json


class InteractiveVisualizer:
    """
    Creates interactive network visualizations using Plotly backend.
    
    Provides methods for creating dynamic visualizations with zoom and pan
    capabilities, interactive centrality exploration, and community analysis
    features for large networks.
    """
    
    def __init__(self, width: int = 1200, height: int = 800):
        """
        Initialize InteractiveVisualizer.
        
        Args:
            width: Figure width in pixels
            height: Figure height in pixels
        """
        self.width = width
        self.height = height
        self.logger = logging.getLogger(__name__)
        
        # Default styling parameters
        self.default_node_size = 10
        self.default_node_color = '#1f77b4'
        self.default_edge_color = '#666666'
        self.default_edge_width = 1.0
        
        # Color schemes for importance levels
        self.importance_colors = {
            'very_high': '#d62728',    # Red
            'high': '#ff7f0e',         # Orange  
            'medium': '#2ca02c',       # Green
            'low': '#1f77b4',          # Blue
            'very_low': '#9467bd'      # Purple
        }
        
        # Community color palette
        self.community_colors = px.colors.qualitative.Set3
        
    def create_interactive_network(self, 
                                 graph: nx.Graph,
                                 layout: str = 'spring',
                                 node_size: Union[int, Dict[str, float], str] = None,
                                 node_color: Union[str, Dict[str, str], str] = None,
                                 edge_width: Union[float, Dict[Tuple[str, str], float]] = None,
                                 show_labels: bool = True,
                                 title: str = "Interactive Social Network Visualization",
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive network visualization with zoom and pan capabilities.
        
        Args:
            graph: NetworkX graph to visualize
            layout: Layout algorithm ('spring', 'circular', 'hierarchical', 'random')
            node_size: Node sizes (int for uniform, dict for per-node, or centrality metric name)
            node_color: Node colors (str for uniform, dict for per-node, or centrality metric name)
            edge_width: Edge widths (float for uniform, dict for per-edge)
            show_labels: Whether to show node labels on hover
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Plotly figure object with interactive features
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph")
            
        # Calculate layout positions
        pos = self._calculate_layout(graph, layout)
        
        # Process node attributes
        node_sizes = self._process_node_sizes(graph, node_size)
        node_colors = self._process_node_colors(graph, node_color)
        edge_widths = self._process_edge_widths(graph, edge_width)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        self._add_edges_to_figure(fig, graph, pos, edge_widths)
        
        # Add nodes
        self._add_nodes_to_figure(fig, graph, pos, node_sizes, node_colors, show_labels)
        
        # Update layout for interactivity
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text=self._generate_stats_text(graph),
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=0.995,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=self.width,
            height=self.height,
            plot_bgcolor='white'
        )
        
         # Save figure if path provided
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Saved interactive network visualization to {save_path}")
            
        self.logger.info(f"Created interactive network visualization with {graph.number_of_nodes()} nodes")
        return fig
        
    def create_centrality_dashboard(self, 
                                  graph: nx.Graph,
                                  centrality_metrics: Dict[str, Dict[str, float]],
                                  layout: str = 'spring') -> go.Figure:
        """
        Create an interactive dashboard for exploring centrality metrics.
        
        Args:
            graph: NetworkX graph to visualize
            centrality_metrics: Dictionary with centrality types as keys and node scores as values
            layout: Layout algorithm to use
            
        Returns:
            Plotly figure with interactive centrality exploration
        """
        if not centrality_metrics:
            raise ValueError("No centrality metrics provided")
            
        # Calculate layout positions
        pos = self._calculate_layout(graph, layout)
        
        # Create subplots for different centrality metrics
        n_metrics = len(centrality_metrics)
        cols = min(2, n_metrics)
        rows = (n_metrics + 1) // 2
        
        subplot_titles = [f"{metric.replace('_', ' ').title()} Centrality" 
                         for metric in centrality_metrics.keys()]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Add each centrality visualization
        for i, (metric_name, scores) in enumerate(centrality_metrics.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Ensure scores is a dictionary of node->float mappings
            if not isinstance(scores, dict):
                self.logger.warning(f"Invalid scores format for {metric_name}, skipping")
                continue
                
            # Process node attributes for this metric
            node_sizes = self._scale_node_sizes_for_centrality(scores)
            node_colors = self._create_centrality_colors(scores)
            
            # Add edges for this subplot
            self._add_edges_to_subplot(fig, graph, pos, row, col)
            
            # Add nodes for this subplot
            self._add_nodes_to_subplot(fig, graph, pos, node_sizes, node_colors, 
                                     scores, metric_name, row, col)
        
        # Update layout
        fig.update_layout(
            title=dict(text="Interactive Centrality Analysis Dashboard", x=0.5, font=dict(size=24)),
            showlegend=True,
            hovermode='closest',
            width=self.width * 1.5,
            height=self.height * rows / 2,
            plot_bgcolor='white'
        )
        
        # Update all subplot axes
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
        
        self.logger.info(f"Created centrality dashboard with {len(centrality_metrics)} metrics")
        return fig
        
    def create_community_explorer(self, 
                                graph: nx.Graph,
                                communities: Dict[int, List[str]],
                                centrality_scores: Optional[Dict[str, float]] = None,
                                layout: str = 'spring') -> go.Figure:
        """
        Create an interactive community exploration visualization.
        
        Args:
            graph: NetworkX graph to visualize
            communities: Dictionary mapping community IDs to lists of node IDs
            centrality_scores: Optional centrality scores for node sizing
            layout: Layout algorithm to use
            
        Returns:
            Plotly figure with interactive community exploration
        """
        if not communities:
            raise ValueError("No communities provided")
            
        # Calculate layout positions
        pos = self._calculate_layout(graph, layout)
        
        # Create node to community mapping
        node_to_community = {}
        for community_id, node_list in communities.items():
            for node in node_list:
                node_to_community[node] = community_id
        
        # Process node attributes
        if centrality_scores:
            node_sizes = self._scale_node_sizes_for_centrality(centrality_scores)
        else:
            node_sizes = [self.default_node_size] * graph.number_of_nodes()
            
        # Create figure
        fig = go.Figure()
        
        # Add edges
        self._add_edges_to_figure(fig, graph, pos)
        
        # Add nodes by community for better interactivity
        for community_id, node_list in communities.items():
            # Validate that node_list is iterable and not a single value
            if not isinstance(node_list, (list, tuple, set)):
                self.logger.warning(f"Invalid node_list format for community {community_id}: {type(node_list)}")
                continue
                
            community_color = self.community_colors[community_id % len(self.community_colors)]
            
            # Get positions and attributes for this community
            community_nodes = [node for node in node_list if node in graph.nodes()]
            if not community_nodes:
                continue
                
            x_coords = [pos[node][0] for node in community_nodes]
            y_coords = [pos[node][1] for node in community_nodes]
            
            # Get node sizes for this community
            node_indices = [list(graph.nodes()).index(node) for node in community_nodes]
            community_sizes = [node_sizes[i] for i in node_indices]
            
            # Create hover text
            hover_text = []
            for node in community_nodes:
                text = f"Node: {node}<br>"
                text += f"Community: {community_id}<br>"
                
                # Add username if available
                if 'username' in graph.nodes[node] and graph.nodes[node]['username']:
                    text += f"Username: {graph.nodes[node]['username']}<br>"
                    
                # Add centrality if available
                if centrality_scores and node in centrality_scores:
                    text += f"Centrality: {centrality_scores[node]:.4f}<br>"
                    
                # Add degree
                text += f"Degree: {graph.degree(node)}"
                hover_text.append(text)
            
            # Add community nodes
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=community_sizes,
                    color=community_color,
                    line=dict(width=1, color='black'),
                    opacity=0.8
                ),
                text=hover_text,
                hoverinfo='text',
                name=f'Community {community_id} ({len(community_nodes)} nodes)',
                showlegend=True
            ))
        
        # Add community statistics annotation
        stats_text = self._generate_community_stats_text(communities, graph)
        
        # Update layout
        fig.update_layout(
            title=dict(text="Interactive Community Explorer", x=0.5, font=dict(size=20)),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text=stats_text,
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=0.995,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=self.width,
            height=self.height,
            plot_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        )
        
        self.logger.info(f"Created community explorer with {len(communities)} communities")
        return fig
        
    def create_multi_view_dashboard(self, 
                                  graph: nx.Graph,
                                  centrality_metrics: Dict[str, Dict[str, float]],
                                  communities: Dict[int, List[str]],
                                  layout: str = 'spring') -> go.Figure:
        """
        Create a comprehensive multi-view dashboard combining network, centrality, and community views.
        
        Args:
            graph: NetworkX graph to visualize
            centrality_metrics: Dictionary with centrality types as keys and node scores as values
            communities: Dictionary mapping community IDs to lists of node IDs
            layout: Layout algorithm to use
            
        Returns:
            Plotly figure with multiple interactive views
        """
        # Calculate layout positions
        pos = self._calculate_layout(graph, layout)
        
        # Create subplots: 2x2 grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Network Overview",
                "Community Structure", 
                "Degree Centrality",
                "Betweenness Centrality"
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Network Overview (top-left)
        self._add_network_overview_subplot(fig, graph, pos, 1, 1)
        
        # 2. Community Structure (top-right)
        self._add_community_subplot(fig, graph, pos, communities, 1, 2)
        
        # 3. Degree Centrality (bottom-left)
        if 'degree' in centrality_metrics:
            self._add_centrality_subplot(fig, graph, pos, centrality_metrics['degree'], 
                                       'Degree', 2, 1)
        
        # 4. Betweenness Centrality (bottom-right)
        if 'betweenness' in centrality_metrics:
            self._add_centrality_subplot(fig, graph, pos, centrality_metrics['betweenness'], 
                                       'Betweenness', 2, 2)
        
        # Update layout
        fig.update_layout(
            title=dict(text="Social Network Analysis Dashboard", x=0.5, font=dict(size=24)),
            showlegend=False,
            hovermode='closest',
            width=self.width * 1.5,
            height=self.height * 1.2,
            plot_bgcolor='white'
        )
        
        # Update all subplot axes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
        
        self.logger.info("Created multi-view dashboard")
        return fig
        
    def save_interactive_html(self, fig: go.Figure, filepath: str, include_plotlyjs: str = 'cdn'):
        """
        Save interactive visualization as HTML file.
        
        Args:
            fig: Plotly figure to save
            filepath: Path to save HTML file
            include_plotlyjs: How to include Plotly.js ('cdn', 'inline', 'directory')
        """
        fig.write_html(filepath, include_plotlyjs=include_plotlyjs)
        self.logger.info(f"Saved interactive visualization to {filepath}")
        
    def _calculate_layout(self, graph: nx.Graph, layout: str) -> Dict[str, Tuple[float, float]]:
        """
        Calculate node positions using specified layout algorithm.
        
        Args:
            graph: NetworkX graph
            layout: Layout algorithm name
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        layout = layout.lower()
        
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=1/np.sqrt(graph.number_of_nodes()), 
                                 iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'hierarchical':
            try:
                # Use shell layout based on degree
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
                    pos = nx.spring_layout(graph, seed=42)
            except:
                pos = nx.spring_layout(graph, seed=42)
        elif layout == 'random':
            pos = nx.random_layout(graph, seed=42)
        else:
            pos = nx.spring_layout(graph, seed=42)
        
        return pos
        
    def _process_node_sizes(self, graph: nx.Graph, node_size: Union[int, Dict[str, float], str]) -> List[float]:
        """Process node size parameter into list of sizes for each node."""
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
                size = max(5, centrality * 50)  # Scale and set minimum
                sizes.append(size)
            return sizes
        else:
            return [self.default_node_size] * graph.number_of_nodes()
            
    def _process_node_colors(self, graph: nx.Graph, node_color: Union[str, Dict[str, str], str]) -> List[str]:
        """Process node color parameter into list of colors for each node."""
        if node_color is None:
            return [self.default_node_color] * graph.number_of_nodes()
        elif isinstance(node_color, str) and node_color.startswith('#'):
            return [node_color] * graph.number_of_nodes()
        elif isinstance(node_color, dict):
            return [node_color.get(node, self.default_node_color) for node in graph.nodes()]
        elif isinstance(node_color, str):
            # Assume it's a centrality metric name
            colors = []
            centrality_values = []
            
            for node in graph.nodes():
                centrality = graph.nodes[node].get(node_color, 0)
                centrality_values.append(centrality)
                
            if centrality_values:
                colors = self._create_centrality_colors_from_values(centrality_values)
            else:
                colors = [self.default_node_color] * graph.number_of_nodes()
                
            return colors
        else:
            return [self.default_node_color] * graph.number_of_nodes()
            
    def _process_edge_widths(self, graph: nx.Graph, edge_width: Union[float, Dict[Tuple[str, str], float]]) -> List[float]:
        """Process edge width parameter into list of widths for each edge."""
        if edge_width is None:
            widths = []
            for u, v, data in graph.edges(data=True):
                weight = data.get('weight', 1.0)
                width = max(0.5, min(5.0, weight))
                widths.append(width)
            return widths
        elif isinstance(edge_width, (int, float)):
            return [edge_width] * graph.number_of_edges()
        elif isinstance(edge_width, dict):
            widths = []
            for u, v in graph.edges():
                width = edge_width.get((u, v), edge_width.get((v, u), self.default_edge_width))
                widths.append(width)
            return widths
        else:
            return [self.default_edge_width] * graph.number_of_edges()
            
    def _add_edges_to_figure(self, fig: go.Figure, graph: nx.Graph, pos: Dict, 
                           edge_widths: Optional[List[float]] = None):
        """Add edges to Plotly figure."""
        if edge_widths is None:
            edge_widths = [self.default_edge_width] * graph.number_of_edges()
            
        # Create edge traces
        edge_x = []
        edge_y = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color=self.default_edge_color),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
    def _add_nodes_to_figure(self, fig: go.Figure, graph: nx.Graph, pos: Dict,
                           node_sizes: List[float], node_colors: List[str], show_labels: bool):
        """Add nodes to Plotly figure."""
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        
        # Create hover text
        hover_text = []
        for node in graph.nodes():
            text = f"Node: {node}<br>"
            if 'username' in graph.nodes[node] and graph.nodes[node]['username']:
                text += f"Username: {graph.nodes[node]['username']}<br>"
            text += f"Degree: {graph.degree(node)}<br>"
            
            # Add centrality metrics if available
            for metric in ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']:
                if metric in graph.nodes[node]:
                    text += f"{metric.replace('_', ' ').title()}: {graph.nodes[node][metric]:.4f}<br>"
                    
            hover_text.append(text)
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=hover_text if show_labels else None,
            hoverinfo='text' if show_labels else 'none',
            showlegend=False
        ))
        
    def _add_edges_to_subplot(self, fig: go.Figure, graph: nx.Graph, pos: Dict, row: int, col: int):
        """Add edges to a specific subplot."""
        edge_x = []
        edge_y = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color=self.default_edge_color),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ), row=row, col=col)
        
    def _add_nodes_to_subplot(self, fig: go.Figure, graph: nx.Graph, pos: Dict,
                            node_sizes: List[float], node_colors: List[str],
                            scores: Dict[str, float], metric_name: str, row: int, col: int):
        """Add nodes to a specific subplot."""
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        
        # Create hover text
        hover_text = []
        for node in graph.nodes():
            text = f"Node: {node}<br>"
            if 'username' in graph.nodes[node] and graph.nodes[node]['username']:
                text += f"Username: {graph.nodes[node]['username']}<br>"
            text += f"{metric_name.title()}: {scores.get(node, 0):.4f}<br>"
            text += f"Degree: {graph.degree(node)}"
            hover_text.append(text)
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=hover_text,
            hoverinfo='text',
            showlegend=False
        ), row=row, col=col)
        
    def _scale_node_sizes_for_centrality(self, centrality_scores: Dict[str, float]) -> List[float]:
        """Scale node sizes based on centrality scores."""
        if not centrality_scores:
            return []
            
        # Extract numeric values, handling potential non-numeric entries
        values = []
        for val in centrality_scores.values():
            if isinstance(val, (int, float)):
                values.append(val)
            else:
                values.append(0.0)  # Default for non-numeric values
                
        if not values:
            return [self.default_node_size] * len(centrality_scores)
            
        min_val, max_val = min(values), max(values)
        
        if max_val == min_val:
            return [self.default_node_size] * len(values)
            
        # Scale to range [5, 30] for better visibility
        scaled_sizes = []
        for val in values:
            normalized = (val - min_val) / (max_val - min_val)
            size = 5 + normalized * 25
            scaled_sizes.append(size)
            
        return scaled_sizes
        
    def _create_centrality_colors(self, centrality_scores: Dict[str, float]) -> List[str]:
        """Create color mapping based on centrality importance levels."""
        if not centrality_scores:
            return []
            
        values = list(centrality_scores.values())
        return self._create_centrality_colors_from_values(values)
        
    def _create_centrality_colors_from_values(self, values: List[float]) -> List[str]:
        """Create color mapping from centrality values based on percentiles."""
        if not values:
            return []
            
        # Ensure all values are numeric
        numeric_values = []
        for val in values:
            if isinstance(val, (int, float)):
                numeric_values.append(val)
            else:
                numeric_values.append(0.0)
                
        if not numeric_values:
            return [self.default_node_color] * len(values)
            
        # Calculate percentile thresholds
        p20 = np.percentile(numeric_values, 20)
        p40 = np.percentile(numeric_values, 40)
        p60 = np.percentile(numeric_values, 60)
        p80 = np.percentile(numeric_values, 80)
        
        colors = []
        for value in numeric_values:
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
        
    def _add_network_overview_subplot(self, fig: go.Figure, graph: nx.Graph, pos: Dict, row: int, col: int):
        """Add network overview to subplot."""
        self._add_edges_to_subplot(fig, graph, pos, row, col)
        
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=8,
                color=self.default_node_color,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            hoverinfo='none',
            showlegend=False
        ), row=row, col=col)
        
    def _add_community_subplot(self, fig: go.Figure, graph: nx.Graph, pos: Dict, 
                             communities: Dict[int, List[str]], row: int, col: int):
        """Add community visualization to subplot."""
        self._add_edges_to_subplot(fig, graph, pos, row, col)
        
        # Create node to community mapping
        node_to_community = {}
        for community_id, node_list in communities.items():
            # Validate that node_list is iterable
            if not isinstance(node_list, (list, tuple, set)):
                continue
            for node in node_list:
                node_to_community[node] = community_id
        
        # Add nodes colored by community
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        
        node_colors = []
        for node in graph.nodes():
            community_id = node_to_community.get(node, -1)
            if community_id >= 0:
                color = self.community_colors[community_id % len(self.community_colors)]
            else:
                color = '#cccccc'
            node_colors.append(color)
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=8,
                color=node_colors,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            hoverinfo='none',
            showlegend=False
        ), row=row, col=col)
        
    def _add_centrality_subplot(self, fig: go.Figure, graph: nx.Graph, pos: Dict,
                              centrality_scores: Dict[str, float], metric_name: str, row: int, col: int):
        """Add centrality visualization to subplot."""
        self._add_edges_to_subplot(fig, graph, pos, row, col)
        
        node_sizes = self._scale_node_sizes_for_centrality(centrality_scores)
        node_colors = self._create_centrality_colors(centrality_scores)
        
        self._add_nodes_to_subplot(fig, graph, pos, node_sizes, node_colors, 
                                 centrality_scores, metric_name, row, col)
        
    def _generate_stats_text(self, graph: nx.Graph) -> str:
        """Generate network statistics text for display."""
        stats = [
            f"Nodes: {graph.number_of_nodes()}",
            f"Edges: {graph.number_of_edges()}",
            f"Density: {nx.density(graph):.3f}",
            f"Components: {nx.number_connected_components(graph)}"
        ]
        
        if graph.number_of_edges() > 0:
            weights = [data.get('weight', 1) for _, _, data in graph.edges(data=True)]
            stats.append(f"Avg Edge Weight: {np.mean(weights):.2f}")
            
        return "<br>".join(stats)
        
    def _generate_community_stats_text(self, communities: Dict[int, List[str]], graph: nx.Graph) -> str:
        """Generate community statistics text for display."""
        stats = [
            f"Communities: {len(communities)}",
            f"Nodes: {graph.number_of_nodes()}",
            f"Edges: {graph.number_of_edges()}"
        ]
        
        if communities:
            sizes = [len(nodes) for nodes in communities.values()]
            stats.extend([
                f"Avg Community Size: {np.mean(sizes):.1f}",
                f"Largest Community: {max(sizes)}",
                f"Smallest Community: {min(sizes)}"
            ])
            
        return "<br>".join(stats)