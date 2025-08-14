"""
GraphBuilder class for constructing NetworkX graphs from social network data.

This module provides functionality to convert relationship data into NetworkX graphs,
assign node attributes from comments and reply count data, and calculate edge weights
based on interaction frequency.
"""

import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from collections import defaultdict
import numpy as np


class GraphBuilder:
    """
    Constructs NetworkX graphs from social network relationship data.
    
    Handles conversion of relationship data into graph format, assignment of node
    attributes from various data sources, and calculation of edge weights based
    on interaction patterns.
    """
    
    def __init__(self):
        """Initialize GraphBuilder with logging."""
        self.logger = logging.getLogger(__name__)
        
    def build_graph(self, 
                   relationships: pd.DataFrame, 
                   comments: Optional[pd.DataFrame] = None) -> nx.Graph:
        """
        Build NetworkX graph from relationship data.
        
        Args:
            relationships: DataFrame with columns: Source, Destination, Weight, 
                          Source_Name, Destination_Name
            comments: Optional DataFrame with comment data for additional context
            
        Returns:
            NetworkX Graph with nodes and edges from relationship data
            
        Raises:
            ValueError: If required columns are missing from relationships data
        """
        # Validate required columns
        required_columns = ['Source', 'Destination', 'Weight', 'Source_Name', 'Destination_Name']
        missing_columns = [col for col in required_columns if col not in relationships.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in relationships data: {missing_columns}")
            
        if relationships.empty:
            self.logger.warning("Empty relationships DataFrame provided")
            return nx.Graph()
            
        # Create undirected graph
        graph = nx.Graph()
        
        # Add edges with weights
        for _, row in relationships.iterrows():
            source = row['Source']
            destination = row['Destination']
            weight = row['Weight']
            source_name = row['Source_Name']
            dest_name = row['Destination_Name']
            
            # Skip self-loops
            if source == destination:
                continue
                
            # Add edge with weight and metadata
            graph.add_edge(source, destination, 
                          weight=weight,
                          source_name=source_name,
                          dest_name=dest_name)
                          
        # Add node names as attributes
        node_names = {}
        for _, row in relationships.iterrows():
            node_names[row['Source']] = row['Source_Name']
            node_names[row['Destination']] = row['Destination_Name']
            
        nx.set_node_attributes(graph, node_names, 'username')
        
        self.logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
        
    def add_node_attributes(self, 
                           graph: nx.Graph, 
                           attributes: Dict[str, Dict]) -> nx.Graph:
        """
        Add node attributes to existing graph.
        
        Args:
            graph: NetworkX graph to modify
            attributes: Dictionary mapping attribute names to node-value dictionaries
                       e.g., {'reply_count': {'node1': 10, 'node2': 5}}
            
        Returns:
            Modified NetworkX graph with added attributes
        """
        for attr_name, attr_dict in attributes.items():
            # Only add attributes for nodes that exist in the graph
            filtered_attrs = {node: value for node, value in attr_dict.items() 
                            if node in graph.nodes()}
            
            nx.set_node_attributes(graph, filtered_attrs, attr_name)
            
            nodes_updated = len(filtered_attrs)
            total_nodes = len(attr_dict)
            self.logger.info(f"Added {attr_name} attribute to {nodes_updated}/{total_nodes} nodes")
            
        return graph
        
    def add_reply_count_attributes(self, 
                                  graph: nx.Graph, 
                                  reply_counts: pd.DataFrame) -> nx.Graph:
        """
        Add reply count attributes to graph nodes.
        
        Args:
            graph: NetworkX graph to modify
            reply_counts: DataFrame with columns: author_id, author_name, reply_count
            
        Returns:
            Modified NetworkX graph with reply count attributes
        """
        if reply_counts.empty:
            self.logger.warning("Empty reply_counts DataFrame provided")
            return graph
            
        # Create reply count dictionary
        reply_count_dict = dict(zip(reply_counts['author_id'], reply_counts['reply_count']))
        
        # Add as node attributes
        attributes = {'reply_count': reply_count_dict}
        return self.add_node_attributes(graph, attributes)
        
    def add_comment_attributes(self, 
                              graph: nx.Graph, 
                              comments: pd.DataFrame) -> nx.Graph:
        """
        Add comment-based attributes to graph nodes.
        
        Calculates aggregate statistics from comments data including:
        - Total comments count
        - Average likes per comment
        - Total likes received
        
        Args:
            graph: NetworkX graph to modify
            comments: DataFrame with columns: author_id, author, like_count, timestamp
            
        Returns:
            Modified NetworkX graph with comment-based attributes
        """
        if comments.empty:
            self.logger.warning("Empty comments DataFrame provided")
            return graph
            
        # Calculate comment statistics per author
        comment_stats = comments.groupby('author_id').agg({
            'like_count': ['count', 'mean', 'sum'],
            'timestamp': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        comment_stats.columns = ['comment_count', 'avg_likes_per_comment', 'total_likes', 
                               'first_comment_date', 'last_comment_date']
        
        # Convert to dictionaries for node attributes
        attributes = {
            'comment_count': comment_stats['comment_count'].to_dict(),
            'avg_likes_per_comment': comment_stats['avg_likes_per_comment'].to_dict(),
            'total_likes': comment_stats['total_likes'].to_dict(),
            'first_comment_date': comment_stats['first_comment_date'].to_dict(),
            'last_comment_date': comment_stats['last_comment_date'].to_dict()
        }
        
        return self.add_node_attributes(graph, attributes)
        
    def calculate_edge_weights(self, interactions: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """
        Calculate edge weights based on interaction frequency.
        
        Args:
            interactions: DataFrame with interaction data containing Source, Destination, Weight
            
        Returns:
            Dictionary mapping (source, destination) tuples to calculated weights
        """
        if interactions.empty:
            return {}
            
        edge_weights = {}
        
        # Group by source-destination pairs and sum weights
        grouped = interactions.groupby(['Source', 'Destination'])['Weight'].sum()
        
        for (source, dest), weight in grouped.items():
            # Create undirected edge key (smaller node first for consistency)
            edge_key = tuple(sorted([source, dest]))
            
            # If edge already exists, add to existing weight
            if edge_key in edge_weights:
                edge_weights[edge_key] += weight
            else:
                edge_weights[edge_key] = weight
                
        self.logger.info(f"Calculated weights for {len(edge_weights)} edges")
        return edge_weights
        
    def update_edge_weights(self, 
                           graph: nx.Graph, 
                           interactions: pd.DataFrame) -> nx.Graph:
        """
        Update edge weights in graph based on interaction data.
        
        Args:
            graph: NetworkX graph to modify
            interactions: DataFrame with interaction data
            
        Returns:
            Modified NetworkX graph with updated edge weights
        """
        edge_weights = self.calculate_edge_weights(interactions)
        
        # Update weights for existing edges
        updated_count = 0
        for (node1, node2), weight in edge_weights.items():
            if graph.has_edge(node1, node2):
                graph[node1][node2]['weight'] = weight
                updated_count += 1
                
        self.logger.info(f"Updated weights for {updated_count} edges")
        return graph
        
    def build_complete_graph(self, 
                            relationships: pd.DataFrame,
                            comments: Optional[pd.DataFrame] = None,
                            reply_counts: Optional[pd.DataFrame] = None) -> nx.Graph:
        """
        Build complete graph with all available data integrated.
        
        Args:
            relationships: DataFrame with relationship data
            comments: Optional DataFrame with comment data
            reply_counts: Optional DataFrame with reply count data
            
        Returns:
            Complete NetworkX graph with all attributes assigned
        """
        # Build base graph from relationships
        graph = self.build_graph(relationships, comments)
        
        # Add reply count attributes if available
        if reply_counts is not None and not reply_counts.empty:
            graph = self.add_reply_count_attributes(graph, reply_counts)
            
        # Add comment-based attributes if available
        if comments is not None and not comments.empty:
            graph = self.add_comment_attributes(graph, comments)
            
        # Initialize missing attributes with default values
        self._initialize_missing_attributes(graph)
        
        self.logger.info(f"Built complete graph with {graph.number_of_nodes()} nodes, "
                        f"{graph.number_of_edges()} edges")
        return graph
        
    def _initialize_missing_attributes(self, graph: nx.Graph) -> None:
        """
        Initialize missing node attributes with default values.
        
        Args:
            graph: NetworkX graph to modify in-place
        """
        default_attributes = {
            'reply_count': 0,
            'comment_count': 0,
            'avg_likes_per_comment': 0.0,
            'total_likes': 0,
            'first_comment_date': None,
            'last_comment_date': None
        }
        
        for node in graph.nodes():
            for attr_name, default_value in default_attributes.items():
                if attr_name not in graph.nodes[node]:
                    graph.nodes[node][attr_name] = default_value
                    
    def get_graph_statistics(self, graph: nx.Graph) -> Dict[str, Union[int, float]]:
        """
        Calculate basic graph statistics.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            Dictionary containing graph statistics
        """
        if graph.number_of_nodes() == 0:
            return {'nodes': 0, 'edges': 0, 'density': 0.0, 'avg_degree': 0.0}
            
        stats = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
            'connected_components': nx.number_connected_components(graph),
            'largest_component_size': len(max(nx.connected_components(graph), key=len))
        }
        
        # Add weight statistics if edges have weights
        if graph.number_of_edges() > 0:
            weights = [data.get('weight', 1) for _, _, data in graph.edges(data=True)]
            stats.update({
                'avg_edge_weight': np.mean(weights),
                'min_edge_weight': np.min(weights),
                'max_edge_weight': np.max(weights)
            })
            
        return stats