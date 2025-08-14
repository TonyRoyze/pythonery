"""
PyTorch Geometric utilities for converting NetworkX graphs to PyG format
and preprocessing graph data for neural network models.
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


class NetworkXToPyGConverter:
    """Converts NetworkX graphs to PyTorch Geometric format with proper feature handling."""
    
    def __init__(self):
        self.node_feature_keys = [
            'reply_count', 'degree_centrality', 'betweenness_centrality',
            'closeness_centrality', 'clustering_coefficient'
        ]
        self.edge_feature_keys = ['weight']
    
    def convert_graph(self, graph: nx.Graph, include_node_features: bool = True,
                     include_edge_features: bool = True) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            graph: NetworkX graph with node and edge attributes
            include_node_features: Whether to include node features
            include_edge_features: Whether to include edge features
            
        Returns:
            PyTorch Geometric Data object
        """
        # Create a mapping from node IDs to indices
        node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        
        # Convert edges to tensor format
        edge_index = self._create_edge_index(graph, node_mapping)
        
        # Create node features if requested
        x = None
        if include_node_features:
            x = self._create_node_features(graph, node_mapping)
        
        # Create edge features if requested
        edge_attr = None
        if include_edge_features:
            edge_attr = self._create_edge_features(graph, node_mapping)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(graph.nodes())
        )
        
        # Store node mapping for later reference
        data.node_mapping = node_mapping
        data.reverse_node_mapping = {idx: node for node, idx in node_mapping.items()}
        
        return data
    
    def _create_edge_index(self, graph: nx.Graph, node_mapping: Dict) -> torch.Tensor:
        """Create edge index tensor from NetworkX graph."""
        edges = []
        for edge in graph.edges():
            source_idx = node_mapping[edge[0]]
            target_idx = node_mapping[edge[1]]
            edges.append([source_idx, target_idx])
            # Add reverse edge for undirected graph
            edges.append([target_idx, source_idx])
        
        if not edges:
            # Handle empty graph case
            return torch.empty((2, 0), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _create_node_features(self, graph: nx.Graph, node_mapping: Dict) -> torch.Tensor:
        """Create node feature matrix from NetworkX graph attributes."""
        num_nodes = len(graph.nodes())
        features = []
        
        for node in graph.nodes():
            node_features = []
            node_data = graph.nodes[node]
            
            for feature_key in self.node_feature_keys:
                if feature_key in node_data:
                    node_features.append(float(node_data[feature_key]))
                else:
                    # Use default value if feature is missing
                    node_features.append(0.0)
            
            features.append(node_features)
        
        return torch.tensor(features, dtype=torch.float)
    
    def _create_edge_features(self, graph: nx.Graph, node_mapping: Dict) -> torch.Tensor:
        """Create edge feature matrix from NetworkX graph attributes."""
        edge_features = []
        
        for edge in graph.edges(data=True):
            edge_data = edge[2]  # Edge attributes
            features = []
            
            for feature_key in self.edge_feature_keys:
                if feature_key in edge_data:
                    features.append(float(edge_data[feature_key]))
                else:
                    features.append(1.0)  # Default weight
            
            edge_features.append(features)
            # Add same features for reverse edge
            edge_features.append(features)
        
        if not edge_features:
            # Handle empty graph case
            return torch.empty((0, len(self.edge_feature_keys)), dtype=torch.float)
        
        return torch.tensor(edge_features, dtype=torch.float)


class GraphDataPreprocessor:
    """Preprocesses graph data for neural network training."""
    
    def __init__(self, normalize_features: bool = True):
        self.normalize_features = normalize_features
        self.feature_stats = {}
    
    def preprocess_data(self, data: Data, fit_stats: bool = True) -> Data:
        """
        Preprocess PyTorch Geometric data for training.
        
        Args:
            data: PyTorch Geometric Data object
            fit_stats: Whether to fit normalization statistics
            
        Returns:
            Preprocessed Data object
        """
        processed_data = data.clone()
        
        # Normalize node features
        if processed_data.x is not None and self.normalize_features:
            processed_data.x = self._normalize_node_features(
                processed_data.x, fit_stats=fit_stats
            )
        
        # Normalize edge features
        if processed_data.edge_attr is not None and self.normalize_features:
            processed_data.edge_attr = self._normalize_edge_features(
                processed_data.edge_attr, fit_stats=fit_stats
            )
        
        return processed_data
    
    def _normalize_node_features(self, features: torch.Tensor, fit_stats: bool = True) -> torch.Tensor:
        """Normalize node features using z-score normalization."""
        if fit_stats:
            self.feature_stats['node_mean'] = features.mean(dim=0)
            self.feature_stats['node_std'] = features.std(dim=0)
            # Avoid division by zero
            self.feature_stats['node_std'] = torch.where(
                self.feature_stats['node_std'] == 0,
                torch.ones_like(self.feature_stats['node_std']),
                self.feature_stats['node_std']
            )
        
        if 'node_mean' in self.feature_stats and 'node_std' in self.feature_stats:
            normalized = (features - self.feature_stats['node_mean']) / self.feature_stats['node_std']
            return normalized
        
        return features
    
    def _normalize_edge_features(self, features: torch.Tensor, fit_stats: bool = True) -> torch.Tensor:
        """Normalize edge features using z-score normalization."""
        if fit_stats:
            self.feature_stats['edge_mean'] = features.mean(dim=0)
            self.feature_stats['edge_std'] = features.std(dim=0)
            # Avoid division by zero
            self.feature_stats['edge_std'] = torch.where(
                self.feature_stats['edge_std'] == 0,
                torch.ones_like(self.feature_stats['edge_std']),
                self.feature_stats['edge_std']
            )
        
        if 'edge_mean' in self.feature_stats and 'edge_std' in self.feature_stats:
            normalized = (features - self.feature_stats['edge_mean']) / self.feature_stats['edge_std']
            return normalized
        
        return features
    
    def create_train_test_split(self, data: Data, test_ratio: float = 0.2,
                               random_state: int = 42) -> Tuple[Data, Data]:
        """
        Create train/test split for link prediction tasks.
        
        Args:
            data: PyTorch Geometric Data object
            test_ratio: Ratio of edges to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        torch.manual_seed(random_state)
        
        num_edges = data.edge_index.size(1) // 2  # Divide by 2 for undirected graph
        num_test_edges = int(num_edges * test_ratio)
        
        # Randomly select edges for test set
        edge_indices = torch.randperm(num_edges)
        test_edge_indices = edge_indices[:num_test_edges]
        train_edge_indices = edge_indices[num_test_edges:]
        
        # Create train data (remove test edges)
        train_data = self._create_subset_data(data, train_edge_indices * 2)  # *2 for both directions
        
        # Create test data (only test edges)
        test_data = self._create_subset_data(data, test_edge_indices * 2)
        
        return train_data, test_data
    
    def _create_subset_data(self, data: Data, edge_indices: torch.Tensor) -> Data:
        """Create a subset of the data with specified edges."""
        # Select edges and their reverse counterparts
        selected_edges = []
        selected_edge_attrs = []
        
        for idx in edge_indices:
            # Add original edge
            selected_edges.append(data.edge_index[:, idx])
            if data.edge_attr is not None:
                selected_edge_attrs.append(data.edge_attr[idx])
            
            # Add reverse edge
            selected_edges.append(data.edge_index[:, idx + 1])
            if data.edge_attr is not None:
                selected_edge_attrs.append(data.edge_attr[idx + 1])
        
        new_edge_index = torch.stack(selected_edges).t() if selected_edges else torch.empty((2, 0), dtype=torch.long)
        new_edge_attr = torch.stack(selected_edge_attrs) if selected_edge_attrs else None
        
        return Data(
            x=data.x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            num_nodes=data.num_nodes,
            node_mapping=getattr(data, 'node_mapping', None),
            reverse_node_mapping=getattr(data, 'reverse_node_mapping', None)
        )


class PyGDataPipeline:
    """Complete pipeline for converting and preprocessing NetworkX graphs for PyTorch Geometric."""
    
    def __init__(self, normalize_features: bool = True):
        self.converter = NetworkXToPyGConverter()
        self.preprocessor = GraphDataPreprocessor(normalize_features=normalize_features)
    
    def process_graph(self, graph: nx.Graph, include_node_features: bool = True,
                     include_edge_features: bool = True) -> Data:
        """
        Complete pipeline to convert and preprocess NetworkX graph.
        
        Args:
            graph: NetworkX graph
            include_node_features: Whether to include node features
            include_edge_features: Whether to include edge features
            
        Returns:
            Preprocessed PyTorch Geometric Data object
        """
        # Convert to PyG format
        data = self.converter.convert_graph(
            graph, 
            include_node_features=include_node_features,
            include_edge_features=include_edge_features
        )
        
        # Preprocess the data
        processed_data = self.preprocessor.preprocess_data(data, fit_stats=True)
        
        return processed_data
    
    def prepare_for_link_prediction(self, graph: nx.Graph, test_ratio: float = 0.2,
                                   random_state: int = 42) -> Tuple[Data, Data]:
        """
        Prepare graph data for link prediction task.
        
        Args:
            graph: NetworkX graph
            test_ratio: Ratio of edges for testing
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, test_data)
        """
        # Process the full graph
        data = self.process_graph(graph)
        
        # Create train/test split
        train_data, test_data = self.preprocessor.create_train_test_split(
            data, test_ratio=test_ratio, random_state=random_state
        )
        
        return train_data, test_data
    
    def get_data_info(self, data: Data) -> Dict[str, Any]:
        """Get information about the processed data."""
        info = {
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.size(1),
            'num_node_features': data.x.size(1) if data.x is not None else 0,
            'num_edge_features': data.edge_attr.size(1) if data.edge_attr is not None else 0,
            'has_node_features': data.x is not None,
            'has_edge_features': data.edge_attr is not None,
            'device': data.edge_index.device
        }
        
        if data.x is not None:
            info['node_feature_stats'] = {
                'mean': data.x.mean(dim=0).tolist(),
                'std': data.x.std(dim=0).tolist(),
                'min': data.x.min(dim=0)[0].tolist(),
                'max': data.x.max(dim=0)[0].tolist()
            }
        
        return info