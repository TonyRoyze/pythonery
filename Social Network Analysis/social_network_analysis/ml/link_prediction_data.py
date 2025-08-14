"""
Link prediction data preparation pipeline for social network analysis.

This module provides functionality for preparing data for link prediction tasks,
including temporal data splitting, negative sampling, and feature extraction
for node pairs.
"""

import torch
import numpy as np
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Set, Union
import random
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from collections import defaultdict

from .pyg_utils import NetworkXToPyGConverter, GraphDataPreprocessor


class TemporalDataSplitter:
    """
    Handles temporal splitting of graph data for link prediction.
    
    Splits edges based on timestamps to create realistic train/validation/test
    scenarios where we predict future connections based on past interactions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def temporal_split(self, 
                      graph: nx.Graph,
                      relationships_df: pd.DataFrame,
                      train_ratio: float = 0.6,
                      val_ratio: float = 0.2,
                      test_ratio: float = 0.2) -> Tuple[nx.Graph, nx.Graph, nx.Graph, Dict]:
        """
        Split graph data temporally for link prediction.
        
        Args:
            graph: NetworkX graph with edge data
            relationships_df: DataFrame with relationship data including timestamps
            train_ratio: Ratio of earliest edges for training
            val_ratio: Ratio of middle edges for validation  
            test_ratio: Ratio of latest edges for testing
            
        Returns:
            Tuple of (train_graph, val_graph, test_graph, split_info)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Extract edge timestamps if available
        edge_timestamps = self._extract_edge_timestamps(graph, relationships_df)
        
        if not edge_timestamps:
            self.logger.warning("No timestamps found, falling back to random split")
            return self._random_split(graph, train_ratio, val_ratio, test_ratio)
        
        # Sort edges by timestamp
        sorted_edges = sorted(edge_timestamps.items(), key=lambda x: x[1])
        
        # Calculate split points
        total_edges = len(sorted_edges)
        train_end = int(total_edges * train_ratio)
        val_end = int(total_edges * (train_ratio + val_ratio))
        
        # Split edges
        train_edges = [edge for edge, _ in sorted_edges[:train_end]]
        val_edges = [edge for edge, _ in sorted_edges[train_end:val_end]]
        test_edges = [edge for edge, _ in sorted_edges[val_end:]]
        
        # Create subgraphs
        train_graph = self._create_subgraph(graph, train_edges)
        val_graph = self._create_subgraph(graph, train_edges + val_edges)
        test_graph = graph.copy()  # Full graph for test evaluation
        
        split_info = {
            'train_edges': len(train_edges),
            'val_edges': len(val_edges),
            'test_edges': len(test_edges),
            'train_timestamp_range': (
                min(edge_timestamps[e] for e in train_edges) if train_edges else None,
                max(edge_timestamps[e] for e in train_edges) if train_edges else None
            ),
            'val_timestamp_range': (
                min(edge_timestamps[e] for e in val_edges) if val_edges else None,
                max(edge_timestamps[e] for e in val_edges) if val_edges else None
            ),
            'test_timestamp_range': (
                min(edge_timestamps[e] for e in test_edges) if test_edges else None,
                max(edge_timestamps[e] for e in test_edges) if test_edges else None
            )
        }
        
        self.logger.info(f"Temporal split: {len(train_edges)} train, {len(val_edges)} val, {len(test_edges)} test edges")
        return train_graph, val_graph, test_graph, split_info
    
    def _extract_edge_timestamps(self, 
                                graph: nx.Graph, 
                                relationships_df: pd.DataFrame) -> Dict[Tuple, datetime]:
        """Extract timestamps for edges from relationships data."""
        edge_timestamps = {}
        
        # Check if relationships_df has timestamp column
        timestamp_cols = [col for col in relationships_df.columns 
                         if 'timestamp' in col.lower() or 'date' in col.lower() or 'time' in col.lower()]
        
        if not timestamp_cols:
            # Try to use edge attributes from graph
            for edge in graph.edges(data=True):
                edge_data = edge[2]
                if 'timestamp' in edge_data:
                    edge_key = tuple(sorted([edge[0], edge[1]]))
                    edge_timestamps[edge_key] = edge_data['timestamp']
            return edge_timestamps
        
        # Use timestamp from relationships DataFrame
        timestamp_col = timestamp_cols[0]
        
        for _, row in relationships_df.iterrows():
            source = row['Source']
            dest = row['Destination']
            edge_key = tuple(sorted([source, dest]))
            
            if timestamp_col in row and pd.notna(row[timestamp_col]):
                try:
                    timestamp = pd.to_datetime(row[timestamp_col])
                    edge_timestamps[edge_key] = timestamp
                except:
                    continue
        
        return edge_timestamps
    
    def _random_split(self, 
                     graph: nx.Graph,
                     train_ratio: float,
                     val_ratio: float, 
                     test_ratio: float) -> Tuple[nx.Graph, nx.Graph, nx.Graph, Dict]:
        """Fallback to random split when timestamps are not available."""
        edges = list(graph.edges())
        random.shuffle(edges)
        
        total_edges = len(edges)
        train_end = int(total_edges * train_ratio)
        val_end = int(total_edges * (train_ratio + val_ratio))
        
        train_edges = edges[:train_end]
        val_edges = edges[train_end:val_end]
        test_edges = edges[val_end:]
        
        train_graph = self._create_subgraph(graph, train_edges)
        val_graph = self._create_subgraph(graph, train_edges + val_edges)
        test_graph = graph.copy()
        
        split_info = {
            'train_edges': len(train_edges),
            'val_edges': len(val_edges),
            'test_edges': len(test_edges),
            'split_type': 'random'
        }
        
        return train_graph, val_graph, test_graph, split_info
    
    def _create_subgraph(self, graph: nx.Graph, edges: List[Tuple]) -> nx.Graph:
        """Create subgraph containing only specified edges."""
        subgraph = nx.Graph()
        
        # Add all nodes with their attributes
        for node, data in graph.nodes(data=True):
            subgraph.add_node(node, **data)
        
        # Add specified edges with their attributes
        for edge in edges:
            if graph.has_edge(edge[0], edge[1]):
                edge_data = graph[edge[0]][edge[1]]
                subgraph.add_edge(edge[0], edge[1], **edge_data)
        
        return subgraph


class NegativeSampler:
    """
    Generates negative samples (non-existent edges) for link prediction training.
    
    Implements various negative sampling strategies to create balanced datasets
    for link prediction tasks.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        random.seed(random_state)
        np.random.seed(random_state)
    
    def sample_negative_edges(self,
                             graph: nx.Graph,
                             num_negative: Optional[int] = None,
                             strategy: str = 'random',
                             exclude_edges: Optional[Set[Tuple]] = None) -> List[Tuple]:
        """
        Sample negative edges (non-existent connections) from the graph.
        
        Args:
            graph: NetworkX graph
            num_negative: Number of negative samples to generate (default: same as positive edges)
            strategy: Sampling strategy ('random', 'degree_based', 'community_aware')
            exclude_edges: Set of edges to exclude from negative sampling
            
        Returns:
            List of negative edge tuples
        """
        if num_negative is None:
            num_negative = graph.number_of_edges()
        
        # Get existing edges
        existing_edges = set()
        for edge in graph.edges():
            existing_edges.add(tuple(sorted([edge[0], edge[1]])))
        
        # Add excluded edges
        if exclude_edges:
            existing_edges.update(exclude_edges)
        
        # Generate negative samples based on strategy
        if strategy == 'random':
            negative_edges = self._random_negative_sampling(graph, existing_edges, num_negative)
        elif strategy == 'degree_based':
            negative_edges = self._degree_based_negative_sampling(graph, existing_edges, num_negative)
        elif strategy == 'community_aware':
            negative_edges = self._community_aware_negative_sampling(graph, existing_edges, num_negative)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        self.logger.info(f"Generated {len(negative_edges)} negative samples using {strategy} strategy")
        return negative_edges
    
    def _random_negative_sampling(self,
                                 graph: nx.Graph,
                                 existing_edges: Set[Tuple],
                                 num_negative: int) -> List[Tuple]:
        """Random negative sampling strategy."""
        nodes = list(graph.nodes())
        negative_edges = []
        max_attempts = num_negative * 10  # Prevent infinite loops
        attempts = 0
        
        while len(negative_edges) < num_negative and attempts < max_attempts:
            # Randomly select two different nodes
            node1, node2 = random.sample(nodes, 2)
            edge = tuple(sorted([node1, node2]))
            
            # Check if edge doesn't exist
            if edge not in existing_edges:
                negative_edges.append(edge)
                existing_edges.add(edge)  # Avoid duplicates
            
            attempts += 1
        
        if len(negative_edges) < num_negative:
            self.logger.warning(f"Could only generate {len(negative_edges)} negative samples out of {num_negative} requested")
        
        return negative_edges
    
    def _degree_based_negative_sampling(self,
                                       graph: nx.Graph,
                                       existing_edges: Set[Tuple],
                                       num_negative: int) -> List[Tuple]:
        """Degree-based negative sampling - prefer high-degree nodes."""
        nodes = list(graph.nodes())
        degrees = dict(graph.degree())
        
        # Create probability distribution based on node degrees
        node_probs = np.array([degrees[node] + 1 for node in nodes])  # +1 to avoid zero probability
        node_probs = node_probs / node_probs.sum()
        
        negative_edges = []
        max_attempts = num_negative * 10
        attempts = 0
        
        while len(negative_edges) < num_negative and attempts < max_attempts:
            # Sample nodes based on degree probability
            node1, node2 = np.random.choice(nodes, size=2, replace=False, p=node_probs)
            edge = tuple(sorted([node1, node2]))
            
            if edge not in existing_edges:
                negative_edges.append(edge)
                existing_edges.add(edge)
            
            attempts += 1
        
        return negative_edges
    
    def _community_aware_negative_sampling(self,
                                          graph: nx.Graph,
                                          existing_edges: Set[Tuple],
                                          num_negative: int) -> List[Tuple]:
        """Community-aware negative sampling - mix intra and inter-community edges."""
        try:
            # Detect communities using Louvain algorithm
            import community as community_louvain
            communities = community_louvain.best_partition(graph)
        except ImportError:
            self.logger.warning("python-louvain not available, falling back to random sampling")
            return self._random_negative_sampling(graph, existing_edges, num_negative)
        
        # Group nodes by community
        community_nodes = defaultdict(list)
        for node, comm in communities.items():
            community_nodes[comm].append(node)
        
        negative_edges = []
        max_attempts = num_negative * 10
        attempts = 0
        
        # Sample 50% intra-community and 50% inter-community negative edges
        intra_target = num_negative // 2
        inter_target = num_negative - intra_target
        
        intra_count = 0
        inter_count = 0
        
        while (intra_count < intra_target or inter_count < inter_target) and attempts < max_attempts:
            # Decide whether to sample intra or inter-community edge
            if intra_count < intra_target and (inter_count >= inter_target or random.random() < 0.5):
                # Sample intra-community edge
                comm = random.choice(list(community_nodes.keys()))
                if len(community_nodes[comm]) >= 2:
                    node1, node2 = random.sample(community_nodes[comm], 2)
                    edge = tuple(sorted([node1, node2]))
                    
                    if edge not in existing_edges:
                        negative_edges.append(edge)
                        existing_edges.add(edge)
                        intra_count += 1
            else:
                # Sample inter-community edge
                if len(community_nodes) >= 2:
                    comm1, comm2 = random.sample(list(community_nodes.keys()), 2)
                    node1 = random.choice(community_nodes[comm1])
                    node2 = random.choice(community_nodes[comm2])
                    edge = tuple(sorted([node1, node2]))
                    
                    if edge not in existing_edges:
                        negative_edges.append(edge)
                        existing_edges.add(edge)
                        inter_count += 1
            
            attempts += 1
        
        return negative_edges
    
    def create_balanced_dataset(self,
                               positive_edges: List[Tuple],
                               negative_edges: List[Tuple]) -> Tuple[List[Tuple], List[int]]:
        """
        Create balanced dataset with equal positive and negative samples.
        
        Args:
            positive_edges: List of positive edge samples
            negative_edges: List of negative edge samples
            
        Returns:
            Tuple of (all_edges, labels) where labels are 1 for positive, 0 for negative
        """
        min_samples = min(len(positive_edges), len(negative_edges))
        
        # Sample equal numbers
        selected_positive = random.sample(positive_edges, min_samples)
        selected_negative = random.sample(negative_edges, min_samples)
        
        # Combine and create labels
        all_edges = selected_positive + selected_negative
        labels = [1] * len(selected_positive) + [0] * len(selected_negative)
        
        # Shuffle the dataset
        combined = list(zip(all_edges, labels))
        random.shuffle(combined)
        all_edges, labels = zip(*combined)
        
        return list(all_edges), list(labels)


class NodePairFeatureExtractor:
    """
    Extracts features for node pairs for link prediction.
    
    Implements various graph-based features including common neighbors,
    preferential attachment, and other structural similarity measures.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self,
                        graph: nx.Graph,
                        node_pairs: List[Tuple],
                        feature_types: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract features for node pairs.
        
        Args:
            graph: NetworkX graph
            node_pairs: List of node pair tuples
            feature_types: List of feature types to extract (default: all available)
            
        Returns:
            Feature matrix of shape (num_pairs, num_features)
        """
        if feature_types is None:
            feature_types = [
                'common_neighbors', 'jaccard_coefficient', 'adamic_adar',
                'preferential_attachment', 'resource_allocation',
                'shortest_path', 'degree_product', 'degree_difference',
                'clustering_coefficient_product', 'centrality_features'
            ]
        
        features = []
        
        for node_pair in node_pairs:
            pair_features = []
            
            for feature_type in feature_types:
                if feature_type == 'common_neighbors':
                    pair_features.extend(self._common_neighbors_features(graph, node_pair))
                elif feature_type == 'jaccard_coefficient':
                    pair_features.append(self._jaccard_coefficient(graph, node_pair))
                elif feature_type == 'adamic_adar':
                    pair_features.append(self._adamic_adar(graph, node_pair))
                elif feature_type == 'preferential_attachment':
                    pair_features.append(self._preferential_attachment(graph, node_pair))
                elif feature_type == 'resource_allocation':
                    pair_features.append(self._resource_allocation(graph, node_pair))
                elif feature_type == 'shortest_path':
                    pair_features.append(self._shortest_path_features(graph, node_pair))
                elif feature_type == 'degree_product':
                    pair_features.append(self._degree_product(graph, node_pair))
                elif feature_type == 'degree_difference':
                    pair_features.append(self._degree_difference(graph, node_pair))
                elif feature_type == 'clustering_coefficient_product':
                    pair_features.append(self._clustering_coefficient_product(graph, node_pair))
                elif feature_type == 'centrality_features':
                    pair_features.extend(self._centrality_features(graph, node_pair))
            
            features.append(pair_features)
        
        feature_matrix = np.array(features, dtype=np.float32)
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        
        self.logger.info(f"Extracted {feature_matrix.shape[1]} features for {len(node_pairs)} node pairs")
        return feature_matrix
    
    def _common_neighbors_features(self, graph: nx.Graph, node_pair: Tuple) -> List[float]:
        """Extract common neighbors based features."""
        node1, node2 = node_pair
        
        if not graph.has_node(node1) or not graph.has_node(node2):
            return [0.0, 0.0]  # [count, normalized]
        
        neighbors1 = set(graph.neighbors(node1))
        neighbors2 = set(graph.neighbors(node2))
        
        common_neighbors = len(neighbors1.intersection(neighbors2))
        total_neighbors = len(neighbors1.union(neighbors2))
        
        normalized_common = common_neighbors / max(total_neighbors, 1)
        
        return [float(common_neighbors), normalized_common]
    
    def _jaccard_coefficient(self, graph: nx.Graph, node_pair: Tuple) -> float:
        """Calculate Jaccard coefficient for node pair."""
        node1, node2 = node_pair
        
        if not graph.has_node(node1) or not graph.has_node(node2):
            return 0.0
        
        neighbors1 = set(graph.neighbors(node1))
        neighbors2 = set(graph.neighbors(node2))
        
        intersection = len(neighbors1.intersection(neighbors2))
        union = len(neighbors1.union(neighbors2))
        
        return intersection / max(union, 1)
    
    def _adamic_adar(self, graph: nx.Graph, node_pair: Tuple) -> float:
        """Calculate Adamic-Adar index for node pair."""
        node1, node2 = node_pair
        
        if not graph.has_node(node1) or not graph.has_node(node2):
            return 0.0
        
        neighbors1 = set(graph.neighbors(node1))
        neighbors2 = set(graph.neighbors(node2))
        common_neighbors = neighbors1.intersection(neighbors2)
        
        if not common_neighbors:
            return 0.0
        
        adamic_adar = 0.0
        for neighbor in common_neighbors:
            degree = graph.degree(neighbor)
            if degree > 1:
                adamic_adar += 1.0 / np.log(degree)
        
        return adamic_adar
    
    def _preferential_attachment(self, graph: nx.Graph, node_pair: Tuple) -> float:
        """Calculate preferential attachment score for node pair."""
        node1, node2 = node_pair
        
        if not graph.has_node(node1) or not graph.has_node(node2):
            return 0.0
        
        degree1 = graph.degree(node1)
        degree2 = graph.degree(node2)
        
        return float(degree1 * degree2)
    
    def _resource_allocation(self, graph: nx.Graph, node_pair: Tuple) -> float:
        """Calculate resource allocation index for node pair."""
        node1, node2 = node_pair
        
        if not graph.has_node(node1) or not graph.has_node(node2):
            return 0.0
        
        neighbors1 = set(graph.neighbors(node1))
        neighbors2 = set(graph.neighbors(node2))
        common_neighbors = neighbors1.intersection(neighbors2)
        
        if not common_neighbors:
            return 0.0
        
        resource_allocation = 0.0
        for neighbor in common_neighbors:
            degree = graph.degree(neighbor)
            if degree > 0:
                resource_allocation += 1.0 / degree
        
        return resource_allocation
    
    def _shortest_path_features(self, graph: nx.Graph, node_pair: Tuple) -> float:
        """Calculate shortest path length between nodes."""
        node1, node2 = node_pair
        
        if not graph.has_node(node1) or not graph.has_node(node2):
            return float('inf')
        
        try:
            path_length = nx.shortest_path_length(graph, node1, node2)
            return float(path_length)
        except nx.NetworkXNoPath:
            return float('inf')
    
    def _degree_product(self, graph: nx.Graph, node_pair: Tuple) -> float:
        """Calculate product of node degrees."""
        return self._preferential_attachment(graph, node_pair)
    
    def _degree_difference(self, graph: nx.Graph, node_pair: Tuple) -> float:
        """Calculate absolute difference in node degrees."""
        node1, node2 = node_pair
        
        if not graph.has_node(node1) or not graph.has_node(node2):
            return 0.0
        
        degree1 = graph.degree(node1)
        degree2 = graph.degree(node2)
        
        return float(abs(degree1 - degree2))
    
    def _clustering_coefficient_product(self, graph: nx.Graph, node_pair: Tuple) -> float:
        """Calculate product of clustering coefficients."""
        node1, node2 = node_pair
        
        if not graph.has_node(node1) or not graph.has_node(node2):
            return 0.0
        
        clustering1 = nx.clustering(graph, node1)
        clustering2 = nx.clustering(graph, node2)
        
        return clustering1 * clustering2
    
    def _centrality_features(self, graph: nx.Graph, node_pair: Tuple) -> List[float]:
        """Extract centrality-based features for node pair."""
        node1, node2 = node_pair
        
        if not graph.has_node(node1) or not graph.has_node(node2):
            return [0.0, 0.0, 0.0, 0.0]  # [degree_cent_prod, between_cent_prod, close_cent_prod, cent_diff]
        
        # Get centrality values from node attributes if available
        degree_cent1 = graph.nodes[node1].get('degree_centrality', 0.0)
        degree_cent2 = graph.nodes[node2].get('degree_centrality', 0.0)
        
        between_cent1 = graph.nodes[node1].get('betweenness_centrality', 0.0)
        between_cent2 = graph.nodes[node2].get('betweenness_centrality', 0.0)
        
        close_cent1 = graph.nodes[node1].get('closeness_centrality', 0.0)
        close_cent2 = graph.nodes[node2].get('closeness_centrality', 0.0)
        
        # Calculate feature combinations
        degree_cent_product = degree_cent1 * degree_cent2
        between_cent_product = between_cent1 * between_cent2
        close_cent_product = close_cent1 * close_cent2
        centrality_difference = abs(degree_cent1 - degree_cent2)
        
        return [degree_cent_product, between_cent_product, close_cent_product, centrality_difference]
    
    def get_feature_names(self, feature_types: Optional[List[str]] = None) -> List[str]:
        """Get names of extracted features."""
        if feature_types is None:
            feature_types = [
                'common_neighbors', 'jaccard_coefficient', 'adamic_adar',
                'preferential_attachment', 'resource_allocation',
                'shortest_path', 'degree_product', 'degree_difference',
                'clustering_coefficient_product', 'centrality_features'
            ]
        
        feature_names = []
        
        for feature_type in feature_types:
            if feature_type == 'common_neighbors':
                feature_names.extend(['common_neighbors_count', 'common_neighbors_normalized'])
            elif feature_type == 'jaccard_coefficient':
                feature_names.append('jaccard_coefficient')
            elif feature_type == 'adamic_adar':
                feature_names.append('adamic_adar')
            elif feature_type == 'preferential_attachment':
                feature_names.append('preferential_attachment')
            elif feature_type == 'resource_allocation':
                feature_names.append('resource_allocation')
            elif feature_type == 'shortest_path':
                feature_names.append('shortest_path_length')
            elif feature_type == 'degree_product':
                feature_names.append('degree_product')
            elif feature_type == 'degree_difference':
                feature_names.append('degree_difference')
            elif feature_type == 'clustering_coefficient_product':
                feature_names.append('clustering_coefficient_product')
            elif feature_type == 'centrality_features':
                feature_names.extend([
                    'degree_centrality_product', 'betweenness_centrality_product',
                    'closeness_centrality_product', 'centrality_difference'
                ])
        
        return feature_names


class LinkPredictionDataPipeline:
    """
    Complete pipeline for preparing link prediction data.
    
    Combines temporal splitting, negative sampling, and feature extraction
    to create ready-to-use datasets for link prediction models.
    """
    
    def __init__(self, random_state: int = 42):
        self.temporal_splitter = TemporalDataSplitter()
        self.negative_sampler = NegativeSampler(random_state=random_state)
        self.feature_extractor = NodePairFeatureExtractor()
        self.pyg_converter = NetworkXToPyGConverter()
        self.preprocessor = GraphDataPreprocessor()
        self.logger = logging.getLogger(__name__)
    
    def prepare_link_prediction_data(self,
                                   graph: nx.Graph,
                                   relationships_df: pd.DataFrame,
                                   train_ratio: float = 0.6,
                                   val_ratio: float = 0.2,
                                   test_ratio: float = 0.2,
                                   negative_sampling_strategy: str = 'random',
                                   feature_types: Optional[List[str]] = None) -> Dict:
        """
        Complete pipeline to prepare link prediction data.
        
        Args:
            graph: NetworkX graph
            relationships_df: DataFrame with relationship data
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            test_ratio: Ratio for test data
            negative_sampling_strategy: Strategy for negative sampling
            feature_types: Types of features to extract
            
        Returns:
            Dictionary containing prepared datasets and metadata
        """
        # Step 1: Temporal data splitting
        self.logger.info("Performing temporal data splitting...")
        train_graph, val_graph, test_graph, split_info = self.temporal_splitter.temporal_split(
            graph, relationships_df, train_ratio, val_ratio, test_ratio
        )
        
        # Step 2: Extract positive edges for each split
        train_edges = list(train_graph.edges())
        val_edges = [edge for edge in val_graph.edges() if edge not in train_edges]
        test_edges = [edge for edge in test_graph.edges() if edge not in val_graph.edges()]
        
        # Step 3: Generate negative samples
        self.logger.info("Generating negative samples...")
        
        # For training: sample from nodes in training graph
        train_negative = self.negative_sampler.sample_negative_edges(
            train_graph, len(train_edges), negative_sampling_strategy
        )
        
        # For validation: sample from nodes in validation graph, excluding train edges
        val_exclude = set(tuple(sorted([e[0], e[1]])) for e in train_edges)
        val_negative = self.negative_sampler.sample_negative_edges(
            val_graph, len(val_edges), negative_sampling_strategy, val_exclude
        )
        
        # For test: sample from full graph, excluding train and val edges
        test_exclude = val_exclude.union(set(tuple(sorted([e[0], e[1]])) for e in val_edges))
        test_negative = self.negative_sampler.sample_negative_edges(
            test_graph, len(test_edges), negative_sampling_strategy, test_exclude
        )
        
        # Step 4: Extract features for node pairs
        self.logger.info("Extracting node pair features...")
        
        # Prepare positive and negative edge lists
        train_pos_edges = [tuple(sorted([e[0], e[1]])) for e in train_edges]
        val_pos_edges = [tuple(sorted([e[0], e[1]])) for e in val_edges]
        test_pos_edges = [tuple(sorted([e[0], e[1]])) for e in test_edges]
        
        # Extract features using the full graph for consistency
        train_all_edges = train_pos_edges + train_negative
        val_all_edges = val_pos_edges + val_negative
        test_all_edges = test_pos_edges + test_negative
        
        train_features = self.feature_extractor.extract_features(graph, train_all_edges, feature_types)
        val_features = self.feature_extractor.extract_features(graph, val_all_edges, feature_types)
        test_features = self.feature_extractor.extract_features(graph, test_all_edges, feature_types)
        
        # Step 5: Create labels
        train_labels = [1] * len(train_pos_edges) + [0] * len(train_negative)
        val_labels = [1] * len(val_pos_edges) + [0] * len(val_negative)
        test_labels = [1] * len(test_pos_edges) + [0] * len(test_negative)
        
        # Step 6: Convert to PyTorch Geometric format
        self.logger.info("Converting to PyTorch Geometric format...")
        train_pyg_data = self.pyg_converter.convert_graph(train_graph)
        val_pyg_data = self.pyg_converter.convert_graph(val_graph)
        test_pyg_data = self.pyg_converter.convert_graph(test_graph)
        
        # Step 7: Prepare final dataset
        dataset = {
            'train': {
                'graph': train_graph,
                'pyg_data': train_pyg_data,
                'edges': train_all_edges,
                'features': train_features,
                'labels': np.array(train_labels),
                'positive_edges': train_pos_edges,
                'negative_edges': train_negative
            },
            'val': {
                'graph': val_graph,
                'pyg_data': val_pyg_data,
                'edges': val_all_edges,
                'features': val_features,
                'labels': np.array(val_labels),
                'positive_edges': val_pos_edges,
                'negative_edges': val_negative
            },
            'test': {
                'graph': test_graph,
                'pyg_data': test_pyg_data,
                'edges': test_all_edges,
                'features': test_features,
                'labels': np.array(test_labels),
                'positive_edges': test_pos_edges,
                'negative_edges': test_negative
            },
            'metadata': {
                'split_info': split_info,
                'feature_names': self.feature_extractor.get_feature_names(feature_types),
                'num_features': train_features.shape[1],
                'negative_sampling_strategy': negative_sampling_strategy,
                'original_graph_stats': {
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges(),
                    'density': nx.density(graph)
                }
            }
        }
        
        self.logger.info("Link prediction data preparation completed successfully")
        return dataset
    
    def save_dataset(self, dataset: Dict, output_dir: str) -> None:
        """Save prepared dataset to disk."""
        import pickle
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the complete dataset
        with open(os.path.join(output_dir, 'link_prediction_dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        
        # Save individual components as CSV for easy inspection
        for split in ['train', 'val', 'test']:
            split_data = dataset[split]
            
            # Save features and labels
            features_df = pd.DataFrame(
                split_data['features'],
                columns=dataset['metadata']['feature_names']
            )
            features_df['label'] = split_data['labels']
            features_df.to_csv(os.path.join(output_dir, f'{split}_features.csv'), index=False)
            
            # Save edge lists
            edges_df = pd.DataFrame(split_data['edges'], columns=['node1', 'node2'])
            edges_df['label'] = split_data['labels']
            edges_df.to_csv(os.path.join(output_dir, f'{split}_edges.csv'), index=False)
        
        # Save metadata
        metadata_df = pd.DataFrame([dataset['metadata']])
        metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
        
        self.logger.info(f"Dataset saved to {output_dir}")