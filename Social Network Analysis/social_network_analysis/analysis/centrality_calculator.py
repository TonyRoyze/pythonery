"""
CentralityCalculator class for comprehensive centrality analysis of social networks.

This module provides functionality to calculate various centrality metrics including
degree, betweenness, closeness centrality, and clustering coefficients. It also
includes ranking and comparison utilities for centrality analysis.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict
import warnings


class CentralityCalculator:
    """
    Calculates comprehensive centrality metrics for network analysis.
    
    Provides methods for computing degree, betweenness, closeness centrality,
    clustering coefficients, and utilities for ranking and comparing centrality
    measures across network nodes.
    """
    
    def __init__(self):
        """Initialize CentralityCalculator with logging."""
        self.logger = logging.getLogger(__name__)
        self._centrality_cache = {}
        
    def calculate_degree_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Calculate degree centrality for all nodes in the graph.
        
        Degree centrality measures the fraction of nodes a node is connected to.
        For undirected graphs, this is the number of neighbors divided by the
        maximum possible number of neighbors.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            Dictionary mapping node IDs to degree centrality scores
            
        Raises:
            ValueError: If graph is empty
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot calculate centrality for empty graph")
            
        cache_key = ('degree', id(graph))
        if cache_key in self._centrality_cache:
            return self._centrality_cache[cache_key]
            
        degree_centrality = nx.degree_centrality(graph)
        self._centrality_cache[cache_key] = degree_centrality
        
        self.logger.info(f"Calculated degree centrality for {len(degree_centrality)} nodes")
        return degree_centrality
        
    def calculate_betweenness_centrality(self, 
                                       graph: nx.Graph, 
                                       normalized: bool = True,
                                       weight: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate betweenness centrality for all nodes in the graph.
        
        Betweenness centrality measures the extent to which a node lies on paths
        between other nodes. Nodes with high betweenness centrality act as bridges
        in the network.
        
        Args:
            graph: NetworkX graph to analyze
            normalized: Whether to normalize centrality values
            weight: Edge attribute name to use as weight (optional)
            
        Returns:
            Dictionary mapping node IDs to betweenness centrality scores
            
        Raises:
            ValueError: If graph is empty
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot calculate centrality for empty graph")
            
        cache_key = ('betweenness', id(graph), normalized, weight)
        if cache_key in self._centrality_cache:
            return self._centrality_cache[cache_key]
            
        # For large graphs, use approximation to improve performance
        if graph.number_of_nodes() > 1000:
            self.logger.warning("Large graph detected. Using approximation for betweenness centrality.")
            k = min(100, graph.number_of_nodes() // 10)  # Sample size for approximation
            betweenness_centrality = nx.betweenness_centrality(
                graph, normalized=normalized, weight=weight, k=k
            )
        else:
            betweenness_centrality = nx.betweenness_centrality(
                graph, normalized=normalized, weight=weight
            )
            
        self._centrality_cache[cache_key] = betweenness_centrality
        
        self.logger.info(f"Calculated betweenness centrality for {len(betweenness_centrality)} nodes")
        return betweenness_centrality
        
    def calculate_closeness_centrality(self, 
                                     graph: nx.Graph,
                                     distance: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate closeness centrality for all nodes in the graph.
        
        Closeness centrality measures how close a node is to all other nodes
        in the network. Nodes with high closeness centrality can reach other
        nodes through shorter paths.
        
        Args:
            graph: NetworkX graph to analyze
            distance: Edge attribute name to use as distance (optional)
            
        Returns:
            Dictionary mapping node IDs to closeness centrality scores
            
        Raises:
            ValueError: If graph is empty
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot calculate centrality for empty graph")
            
        cache_key = ('closeness', id(graph), distance)
        if cache_key in self._centrality_cache:
            return self._centrality_cache[cache_key]
            
        # Handle disconnected graphs by calculating for each component
        if not nx.is_connected(graph):
            self.logger.warning("Graph is disconnected. Calculating closeness centrality per component.")
            closeness_centrality = {}
            
            for component in nx.connected_components(graph):
                subgraph = graph.subgraph(component)
                component_closeness = nx.closeness_centrality(subgraph, distance=distance)
                closeness_centrality.update(component_closeness)
        else:
            closeness_centrality = nx.closeness_centrality(graph, distance=distance)
            
        self._centrality_cache[cache_key] = closeness_centrality
        
        self.logger.info(f"Calculated closeness centrality for {len(closeness_centrality)} nodes")
        return closeness_centrality
        
    def calculate_clustering_coefficient(self, 
                                       graph: nx.Graph,
                                       weight: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate clustering coefficient for all nodes in the graph.
        
        Clustering coefficient measures the degree to which nodes in a graph
        tend to cluster together. It quantifies how close a node's neighbors
        are to being a complete graph (clique).
        
        Args:
            graph: NetworkX graph to analyze
            weight: Edge attribute name to use as weight (optional)
            
        Returns:
            Dictionary mapping node IDs to clustering coefficient scores
            
        Raises:
            ValueError: If graph is empty
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot calculate clustering coefficient for empty graph")
            
        cache_key = ('clustering', id(graph), weight)
        if cache_key in self._centrality_cache:
            return self._centrality_cache[cache_key]
            
        clustering_coefficient = nx.clustering(graph, weight=weight)
        self._centrality_cache[cache_key] = clustering_coefficient
        
        self.logger.info(f"Calculated clustering coefficient for {len(clustering_coefficient)} nodes")
        return clustering_coefficient
        
    def calculate_all_centralities(self, 
                                 graph: nx.Graph,
                                 include_weights: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Calculate all centrality metrics for the graph.
        
        Computes degree, betweenness, closeness centrality, and clustering
        coefficients for all nodes in a single call.
        
        Args:
            graph: NetworkX graph to analyze
            include_weights: Whether to use edge weights in calculations
            
        Returns:
            Dictionary containing all centrality metrics:
            {
                'degree': {node: score, ...},
                'betweenness': {node: score, ...},
                'closeness': {node: score, ...},
                'clustering': {node: score, ...}
            }
        """
        weight_attr = 'weight' if include_weights and self._has_edge_weights(graph) else None
        
        centralities = {
            'degree': self.calculate_degree_centrality(graph),
            'betweenness': self.calculate_betweenness_centrality(graph, weight=weight_attr),
            'closeness': self.calculate_closeness_centrality(graph),
            'clustering': self.calculate_clustering_coefficient(graph, weight=weight_attr)
        }
        
        self.logger.info("Calculated all centrality metrics")
        return centralities
        
    def rank_nodes_by_centrality(self, 
                                centrality_scores: Dict[str, float],
                                top_k: Optional[int] = None,
                                ascending: bool = False) -> List[Tuple[str, float]]:
        """
        Rank nodes by centrality scores.
        
        Args:
            centrality_scores: Dictionary mapping node IDs to centrality scores
            top_k: Number of top nodes to return (None for all nodes)
            ascending: Whether to sort in ascending order (default: descending)
            
        Returns:
            List of (node_id, score) tuples sorted by centrality score
        """
        if not centrality_scores:
            return []
            
        # Sort nodes by centrality score
        ranked_nodes = sorted(
            centrality_scores.items(),
            key=lambda x: x[1],
            reverse=not ascending
        )
        
        # Return top_k nodes if specified
        if top_k is not None:
            ranked_nodes = ranked_nodes[:top_k]
            
        return ranked_nodes
        
    def get_top_central_nodes(self, 
                             centralities: Dict[str, Dict[str, float]],
                             top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top nodes for each centrality metric.
        
        Args:
            centralities: Dictionary containing all centrality metrics
            top_k: Number of top nodes to return for each metric
            
        Returns:
            Dictionary mapping centrality types to lists of top nodes
        """
        top_nodes = {}
        
        for centrality_type, scores in centralities.items():
            top_nodes[centrality_type] = self.rank_nodes_by_centrality(
                scores, top_k=top_k
            )
            
        return top_nodes
        
    def compare_centrality_rankings(self, 
                                  centrality1: Dict[str, float],
                                  centrality2: Dict[str, float],
                                  method: str = 'spearman') -> float:
        """
        Compare rankings between two centrality measures.
        
        Args:
            centrality1: First centrality measure scores
            centrality2: Second centrality measure scores
            method: Correlation method ('spearman' or 'pearson')
            
        Returns:
            Correlation coefficient between the two rankings
            
        Raises:
            ValueError: If method is not supported or centralities have no common nodes
        """
        if method not in ['spearman', 'pearson']:
            raise ValueError("Method must be 'spearman' or 'pearson'")
            
        # Find common nodes
        common_nodes = set(centrality1.keys()) & set(centrality2.keys())
        
        if not common_nodes:
            raise ValueError("No common nodes between centrality measures")
            
        # Extract scores for common nodes
        scores1 = [centrality1[node] for node in common_nodes]
        scores2 = [centrality2[node] for node in common_nodes]
        
        # Calculate correlation
        if method == 'spearman':
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(scores1, scores2)
        else:  # pearson
            correlation = np.corrcoef(scores1, scores2)[0, 1]
            
        return correlation
        
    def generate_centrality_report(self, 
                                 graph: nx.Graph,
                                 include_weights: bool = True) -> Dict[str, Union[Dict, List, float]]:
        """
        Generate comprehensive centrality analysis report.
        
        Args:
            graph: NetworkX graph to analyze
            include_weights: Whether to use edge weights in calculations
            
        Returns:
            Dictionary containing comprehensive centrality analysis results
        """
        # Calculate all centralities
        centralities = self.calculate_all_centralities(graph, include_weights)
        
        # Get top nodes for each centrality
        top_nodes = self.get_top_central_nodes(centralities, top_k=10)
        
        # Calculate summary statistics
        summary_stats = {}
        for centrality_type, scores in centralities.items():
            values = list(scores.values())
            summary_stats[centrality_type] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
            
        # Calculate correlations between centrality measures
        correlations = {}
        centrality_types = list(centralities.keys())
        for i, cent1 in enumerate(centrality_types):
            for cent2 in centrality_types[i+1:]:
                try:
                    corr = self.compare_centrality_rankings(
                        centralities[cent1], centralities[cent2], method='spearman'
                    )
                    correlations[f"{cent1}_vs_{cent2}"] = corr
                except ValueError:
                    correlations[f"{cent1}_vs_{cent2}"] = None
                    
        # Compile report
        report = {
            'graph_info': {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'connected': nx.is_connected(graph)
            },
            'centrality_scores': centralities,
            'top_nodes': top_nodes,
            'summary_statistics': summary_stats,
            'centrality_correlations': correlations
        }
        
        self.logger.info("Generated comprehensive centrality report")
        return report
        
    def export_centrality_data(self, 
                              centralities: Dict[str, Dict[str, float]],
                              filepath: str,
                              format: str = 'csv') -> None:
        """
        Export centrality data to file.
        
        Args:
            centralities: Dictionary containing centrality metrics
            filepath: Output file path
            format: Export format ('csv' or 'json')
            
        Raises:
            ValueError: If format is not supported
        """
        if format not in ['csv', 'json']:
            raise ValueError("Format must be 'csv' or 'json'")
            
        if format == 'csv':
            # Convert to DataFrame for CSV export
            df = pd.DataFrame(centralities)
            df.index.name = 'node_id'
            df.to_csv(filepath)
        else:  # json
            import json
            with open(filepath, 'w') as f:
                json.dump(centralities, f, indent=2)
                
        self.logger.info(f"Exported centrality data to {filepath}")
        
    def clear_cache(self) -> None:
        """Clear the centrality calculation cache."""
        self._centrality_cache.clear()
        self.logger.info("Cleared centrality calculation cache")
        
    def _has_edge_weights(self, graph: nx.Graph) -> bool:
        """
        Check if graph has edge weights.
        
        Args:
            graph: NetworkX graph to check
            
        Returns:
            True if graph has edge weights, False otherwise
        """
        if graph.number_of_edges() == 0:
            return False
            
        # Check first edge for weight attribute
        first_edge = next(iter(graph.edges(data=True)))
        return 'weight' in first_edge[2]
        
    def get_centrality_statistics(self, 
                                centrality_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate descriptive statistics for centrality scores.
        
        Args:
            centrality_scores: Dictionary mapping node IDs to centrality scores
            
        Returns:
            Dictionary containing descriptive statistics
        """
        if not centrality_scores:
            return {}
            
        values = list(centrality_scores.values())
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'skewness': self._calculate_skewness(values),
            'kurtosis': self._calculate_kurtosis(values)
        }
        
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of values."""
        if len(values) < 3:
            return 0.0
            
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
            
        skewness = np.mean([(x - mean) ** 3 for x in values]) / (std ** 3)
        return skewness
        
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of values."""
        if len(values) < 4:
            return 0.0
            
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
            
        kurtosis = np.mean([(x - mean) ** 4 for x in values]) / (std ** 4) - 3
        return kurtosis