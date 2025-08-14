"""
GraphSAGE model implementation for link prediction in social networks.

This module implements a GraphSAGE (Graph Sample and Aggregate) model using
PyTorch Geometric for predicting future connections in social networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import networkx as nx
from .pyg_utils import NetworkXToPyGConverter, GraphDataPreprocessor
from .link_prediction_data import LinkPredictionDataPipeline


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for link prediction.
    
    Implements the GraphSAGE architecture with multiple layers and
    a link prediction head for binary classification of node pairs.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 aggregator: str = 'mean'):
        """
        Initialize GraphSAGE model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of final node embeddings
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            aggregator: Aggregation method ('mean', 'max', 'lstm')
        """
        super(GraphSAGEModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create GraphSAGE layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
        else:
            # Single layer case
            self.convs[0] = SAGEConv(input_dim, output_dim, aggr=aggregator)
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate node embeddings.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation to last layer
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x
    
    def predict_links(self, 
                     node_embeddings: torch.Tensor,
                     edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict link probabilities for given node pairs.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, output_dim]
            edge_index: Edge indices to predict [2, num_edges]
            
        Returns:
            Link probabilities [num_edges]
        """
        # Get embeddings for source and target nodes
        source_embeddings = node_embeddings[edge_index[0]]
        target_embeddings = node_embeddings[edge_index[1]]
        
        # Concatenate embeddings
        edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
        
        # Predict link probabilities
        link_probs = self.link_predictor(edge_embeddings).squeeze()
        
        return link_probs
    
    def forward_with_prediction(self, 
                               x: torch.Tensor,
                               edge_index: torch.Tensor,
                               pred_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Complete forward pass with link prediction.
        
        Args:
            x: Node features
            edge_index: Training edges for message passing
            pred_edge_index: Edges to predict
            
        Returns:
            Link probabilities for pred_edge_index
        """
        # Generate node embeddings
        node_embeddings = self.forward(x, edge_index)
        
        # Predict links
        link_probs = self.predict_links(node_embeddings, pred_edge_index)
        
        return link_probs


class GraphSAGEPredictor:
    """
    Complete GraphSAGE predictor with training and evaluation capabilities.
    
    Handles data preparation, model training, and performance evaluation
    for link prediction tasks using GraphSAGE.
    """
    
    def __init__(self, 
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 aggregator: str = 'mean',
                 learning_rate: float = 0.01,
                 weight_decay: float = 5e-4,
                 device: str = 'auto'):
        """
        Initialize GraphSAGE predictor.
        
        Args:
            hidden_dim: Hidden dimension size
            output_dim: Output embedding dimension
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            aggregator: Aggregation method
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.optimizer = None
        self.logger = logging.getLogger(__name__)
        
        # Data preparation utilities
        self.data_pipeline = LinkPredictionDataPipeline()
        self.pyg_converter = NetworkXToPyGConverter()
        self.preprocessor = GraphDataPreprocessor()
    
    def prepare_data(self, 
                    graph: nx.Graph,
                    relationships_df,
                    train_ratio: float = 0.6,
                    val_ratio: float = 0.2,
                    test_ratio: float = 0.2) -> Dict:
        """
        Prepare data for GraphSAGE training.
        
        Args:
            graph: NetworkX graph
            relationships_df: DataFrame with relationship data
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            
        Returns:
            Dictionary containing prepared datasets
        """
        self.logger.info("Preparing data for GraphSAGE training...")
        
        # Use the existing data pipeline
        dataset = self.data_pipeline.prepare_link_prediction_data(
            graph=graph,
            relationships_df=relationships_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            negative_sampling_strategy='random'
        )
        
        # Extract PyG data (already converted by the pipeline)
        train_data = dataset['train']['pyg_data']
        val_data = dataset['val']['pyg_data']
        test_data = dataset['test']['pyg_data']
        
        # Preprocess data
        train_data = self.preprocessor.preprocess_data(train_data, fit_stats=True)
        val_data = self.preprocessor.preprocess_data(val_data, fit_stats=False)
        test_data = self.preprocessor.preprocess_data(test_data, fit_stats=False)
        
        # Move to device
        train_data = train_data.to(self.device)
        val_data = val_data.to(self.device)
        test_data = test_data.to(self.device)
        
        # Prepare edge indices for link prediction
        prepared_data = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_pos_edges': dataset['train']['positive_edges'],
            'train_neg_edges': dataset['train']['negative_edges'],
            'val_pos_edges': dataset['val']['positive_edges'],
            'val_neg_edges': dataset['val']['negative_edges'],
            'test_pos_edges': dataset['test']['positive_edges'],
            'test_neg_edges': dataset['test']['negative_edges'],
            'split_info': dataset['metadata']['split_info']
        }
        
        self.logger.info(f"Data prepared: {len(dataset['train']['positive_edges'])} train edges, "
                        f"{len(dataset['val']['positive_edges'])} val edges, "
                        f"{len(dataset['test']['positive_edges'])} test edges")
        
        return prepared_data
    
    def _create_edge_index_tensor(self, edges: List[Tuple], node_mapping: Dict = None) -> torch.Tensor:
        """Convert edge list to PyTorch tensor format."""
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        if node_mapping is not None:
            # Map node IDs to indices
            mapped_edges = []
            for edge in edges:
                if edge[0] in node_mapping and edge[1] in node_mapping:
                    mapped_edges.append([node_mapping[edge[0]], node_mapping[edge[1]]])
            
            if not mapped_edges:
                return torch.empty((2, 0), dtype=torch.long, device=self.device)
            
            edge_array = np.array(mapped_edges)
        else:
            edge_array = np.array(edges)
        
        return torch.tensor(edge_array.T, dtype=torch.long, device=self.device)
    
    def initialize_model(self, input_dim: int):
        """Initialize the GraphSAGE model and optimizer."""
        self.model = GraphSAGEModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            aggregator=self.aggregator
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.logger.info(f"Initialized GraphSAGE model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, data: Data, pos_edges: List[Tuple], neg_edges: List[Tuple]) -> float:
        """Train the model for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get node mapping from data
        node_mapping = getattr(data, 'node_mapping', None)
        
        # Create edge indices for positive and negative samples
        pos_edge_index = self._create_edge_index_tensor(pos_edges, node_mapping)
        neg_edge_index = self._create_edge_index_tensor(neg_edges, node_mapping)
        
        # Skip if no valid edges
        if pos_edge_index.size(1) == 0 and neg_edge_index.size(1) == 0:
            return 0.0
        
        # Combine positive and negative edges
        all_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge_index.size(1), device=self.device),
            torch.zeros(neg_edge_index.size(1), device=self.device)
        ])
        
        # Forward pass
        predictions = self.model.forward_with_prediction(
            data.x, data.edge_index, all_edges
        )
        
        # Calculate loss
        loss = F.binary_cross_entropy(predictions, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data: Data, pos_edges: List[Tuple], neg_edges: List[Tuple]) -> Dict[str, float]:
        """Evaluate the model on given data."""
        self.model.eval()
        
        with torch.no_grad():
            # Get node mapping from data
            node_mapping = getattr(data, 'node_mapping', None)
            
            # Create edge indices
            pos_edge_index = self._create_edge_index_tensor(pos_edges, node_mapping)
            neg_edge_index = self._create_edge_index_tensor(neg_edges, node_mapping)
            
            # Handle case with no valid edges
            if pos_edge_index.size(1) == 0 and neg_edge_index.size(1) == 0:
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'auc_roc': 0.0
                }
            
            # Combine edges and create labels
            all_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            labels = torch.cat([
                torch.ones(pos_edge_index.size(1)),
                torch.zeros(neg_edge_index.size(1))
            ]).cpu().numpy()
            
            # Get predictions
            predictions = self.model.forward_with_prediction(
                data.x, data.edge_index, all_edges
            ).cpu().numpy()
            
            # Convert probabilities to binary predictions
            binary_predictions = (predictions > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(labels, binary_predictions),
                'precision': precision_score(labels, binary_predictions, zero_division=0),
                'recall': recall_score(labels, binary_predictions, zero_division=0),
                'f1_score': f1_score(labels, binary_predictions, zero_division=0),
                'auc_roc': roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
            }
        
        return metrics
    
    def evaluate_detailed(self, data: Data, pos_edges: List[Tuple], neg_edges: List[Tuple]) -> Dict:
        """Evaluate the model and return detailed results including predictions and labels."""
        self.model.eval()
        
        with torch.no_grad():
            # Get node mapping from data
            node_mapping = getattr(data, 'node_mapping', None)
            
            # Create edge indices
            pos_edge_index = self._create_edge_index_tensor(pos_edges, node_mapping)
            neg_edge_index = self._create_edge_index_tensor(neg_edges, node_mapping)
            
            # Handle case with no valid edges
            if pos_edge_index.size(1) == 0 and neg_edge_index.size(1) == 0:
                return {
                    'metrics': {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0,
                        'auc_roc': 0.0
                    },
                    'predictions': np.array([]),
                    'probabilities': np.array([]),
                    'labels': np.array([])
                }
            
            # Combine edges and create labels
            all_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            labels = torch.cat([
                torch.ones(pos_edge_index.size(1)),
                torch.zeros(neg_edge_index.size(1))
            ]).cpu().numpy()
            
            # Get predictions
            predictions = self.model.forward_with_prediction(
                data.x, data.edge_index, all_edges
            ).cpu().numpy()
            
            # Convert probabilities to binary predictions
            binary_predictions = (predictions > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(labels, binary_predictions),
                'precision': precision_score(labels, binary_predictions, zero_division=0),
                'recall': recall_score(labels, binary_predictions, zero_division=0),
                'f1_score': f1_score(labels, binary_predictions, zero_division=0),
                'auc_roc': roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
            }
            
            return {
                'metrics': metrics,
                'predictions': binary_predictions,
                'probabilities': predictions,
                'labels': labels
            }
    
    def train(self, 
             prepared_data: Dict,
             epochs: int = 100,
             early_stopping_patience: int = 10,
             verbose: bool = True) -> Dict:
        """
        Train the GraphSAGE model.
        
        Args:
            prepared_data: Data prepared by prepare_data method
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print training progress
            
        Returns:
            Training history and final metrics
        """
        # Initialize model if not already done
        if self.model is None:
            input_dim = prepared_data['train_data'].x.size(1)
            self.initialize_model(input_dim)
        
        train_data = prepared_data['train_data']
        val_data = prepared_data['val_data']
        
        # Training history
        history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': []
        }
        
        best_val_f1 = 0.0
        patience_counter = 0
        
        self.logger.info(f"Starting GraphSAGE training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(
                train_data,
                prepared_data['train_pos_edges'],
                prepared_data['train_neg_edges']
            )
            
            # Validation
            val_metrics = self.evaluate(
                val_data,
                prepared_data['val_pos_edges'],
                prepared_data['val_neg_edges']
            )
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1_score'])
            history['val_auc'].append(val_metrics['auc_roc'])
            
            # Early stopping check
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Loss: {train_loss:.4f}, "
                    f"Val F1: {val_metrics['f1_score']:.4f}, "
                    f"Val AUC: {val_metrics['auc_roc']:.4f}"
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation on test set
        test_evaluation = self.evaluate_detailed(
            prepared_data['test_data'],
            prepared_data['test_pos_edges'],
            prepared_data['test_neg_edges']
        )
        
        test_metrics = test_evaluation['metrics']
        test_predictions = test_evaluation['predictions']
        test_probabilities = test_evaluation['probabilities']
        test_labels = test_evaluation['labels']
        
        self.logger.info("Training completed!")
        self.logger.info(f"Final test metrics: {test_metrics}")
        
        return {
            'history': history,
            'final_test_metrics': test_metrics,
            'test_metrics': test_metrics,  # For backward compatibility
            'test_predictions': test_predictions,
            'test_probabilities': test_probabilities,
            'test_labels': test_labels,
            'best_val_f1': best_val_f1,
            'epochs_trained': epoch + 1
        }
    
    def predict_links(self, 
                     graph: nx.Graph,
                     node_pairs: List[Tuple]) -> np.ndarray:
        """
        Predict link probabilities for given node pairs.
        
        Args:
            graph: NetworkX graph
            node_pairs: List of node pairs to predict
            
        Returns:
            Array of link probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Convert graph to PyG format
        data = self.pyg_converter.convert_graph(graph)
        data = self.preprocessor.preprocess_data(data, fit_stats=False)
        data = data.to(self.device)
        
        # Convert node pairs to edge index
        node_mapping = getattr(data, 'node_mapping', None)
        edge_index = self._create_edge_index_tensor(node_pairs, node_mapping)
        
        with torch.no_grad():
            predictions = self.model.forward_with_prediction(
                data.x, data.edge_index, edge_index
            ).cpu().numpy()
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        if self.model is None:
            return {"status": "Model not initialized"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": "GraphSAGE",
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "aggregator": self.aggregator,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device)
        }