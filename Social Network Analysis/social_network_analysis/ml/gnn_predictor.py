"""
Graph Neural Network model implementation for link prediction in social networks.

This module implements a general Graph Neural Network (GNN) model using
PyTorch Geometric for predicting future connections in social networks.
Provides an alternative to GraphSAGE with different GNN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, GINConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import networkx as nx
from .pyg_utils import NetworkXToPyGConverter, GraphDataPreprocessor
from .link_prediction_data import LinkPredictionDataPipeline


class GNNModel(nn.Module):
    """
    General Graph Neural Network model for link prediction.
    
    Supports multiple GNN architectures including GCN, GAT, Transformer, and GIN
    with a link prediction head for binary classification of node pairs.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 gnn_type: str = 'GCN',
                 num_heads: int = 4,
                 edge_dim: Optional[int] = None):
        """
        Initialize GNN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of final node embeddings
            num_layers: Number of GNN layers
            dropout: Dropout probability
            gnn_type: Type of GNN ('GCN', 'GAT', 'Transformer', 'GIN')
            num_heads: Number of attention heads (for GAT and Transformer)
            edge_dim: Dimension of edge features (for Transformer)
        """
        super(GNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        
        # Create GNN layers based on type
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if gnn_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        elif gnn_type == 'Transformer':
            # For Transformer, the output dimension is heads * hidden_dim
            transformer_out_dim = hidden_dim
            self.convs.append(TransformerConv(input_dim, transformer_out_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=edge_dim))
        elif gnn_type == 'GIN':
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
            elif gnn_type == 'Transformer':
                self.convs.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=edge_dim))
            elif gnn_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            if gnn_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, output_dim))
            elif gnn_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, output_dim, heads=1, dropout=dropout))
            elif gnn_type == 'Transformer':
                self.convs.append(TransformerConv(hidden_dim, output_dim, heads=1, dropout=dropout, edge_dim=edge_dim))
            elif gnn_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, output_dim),
                    nn.ReLU(),
                    nn.Linear(output_dim, output_dim)
                )
                self.convs.append(GINConv(mlp))
        else:
            # Single layer case - modify first layer
            self.convs[0] = self._create_single_layer(input_dim, output_dim)
        
        # Link prediction head with multiple layers
        self.link_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_single_layer(self, input_dim: int, output_dim: int):
        """Create single layer for single-layer models."""
        if self.gnn_type == 'GCN':
            return GCNConv(input_dim, output_dim)
        elif self.gnn_type == 'GAT':
            return GATConv(input_dim, output_dim, heads=1, dropout=self.dropout)
        elif self.gnn_type == 'Transformer':
            return TransformerConv(input_dim, output_dim, heads=1, dropout=self.dropout, edge_dim=self.edge_dim)
        elif self.gnn_type == 'GIN':
            mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            return GINConv(mlp)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to generate node embeddings.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            if self.gnn_type in ['Transformer'] and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            
            # Apply batch normalization and activation (except for last layer)
            if i < len(self.convs) - 1:
                if x.size(0) > 1:  # Only apply batch norm if batch size > 1
                    x = self.batch_norms[i](x)
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
                               pred_edge_index: torch.Tensor,
                               edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Complete forward pass with link prediction.
        
        Args:
            x: Node features
            edge_index: Training edges for message passing
            pred_edge_index: Edges to predict
            edge_attr: Edge attributes (optional)
            
        Returns:
            Link probabilities for pred_edge_index
        """
        # Generate node embeddings
        node_embeddings = self.forward(x, edge_index, edge_attr)
        
        # Predict links
        link_probs = self.predict_links(node_embeddings, pred_edge_index)
        
        return link_probs


class GNNPredictor:
    """
    Complete GNN predictor with training, hyperparameter tuning, and evaluation capabilities.
    
    Handles data preparation, model training with hyperparameter optimization,
    and comprehensive performance evaluation for link prediction tasks.
    """
    
    def __init__(self, 
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 gnn_type: str = 'GCN',
                 num_heads: int = 4,
                 learning_rate: float = 0.01,
                 weight_decay: float = 5e-4,
                 device: str = 'auto'):
        """
        Initialize GNN predictor.
        
        Args:
            hidden_dim: Hidden dimension size
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
            gnn_type: Type of GNN architecture
            num_heads: Number of attention heads
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.logger = logging.getLogger(__name__)
        
        # Data preparation utilities
        self.data_pipeline = LinkPredictionDataPipeline()
        self.pyg_converter = NetworkXToPyGConverter()
        self.preprocessor = GraphDataPreprocessor()
        
        # Hyperparameter tuning history
        self.tuning_history = []
    
    def prepare_data(self, 
                    graph: nx.Graph,
                    relationships_df,
                    train_ratio: float = 0.6,
                    val_ratio: float = 0.2,
                    test_ratio: float = 0.2) -> Dict:
        """
        Prepare data for GNN training.
        
        Args:
            graph: NetworkX graph
            relationships_df: DataFrame with relationship data
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            
        Returns:
            Dictionary containing prepared datasets
        """
        self.logger.info("Preparing data for GNN training...")
        
        # Use the existing data pipeline
        dataset = self.data_pipeline.prepare_link_prediction_data(
            graph=graph,
            relationships_df=relationships_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            negative_sampling_strategy='random'
        )
        
        # Extract PyG data
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
    
    def initialize_model(self, input_dim: int, edge_dim: Optional[int] = None):
        """Initialize the GNN model and optimizer."""
        self.model = GNNModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            gnn_type=self.gnn_type,
            num_heads=self.num_heads,
            edge_dim=edge_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        
        self.logger.info(f"Initialized {self.gnn_type} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
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
            data.x, data.edge_index, all_edges, data.edge_attr
        )
        
        # Calculate loss (predictions are already sigmoid activated)
        loss = F.binary_cross_entropy(predictions, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
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
                data.x, data.edge_index, all_edges, data.edge_attr
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
                data.x, data.edge_index, all_edges, data.edge_attr
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
    
    def hyperparameter_tuning(self, 
                             prepared_data: Dict,
                             param_grid: Dict,
                             n_trials: int = 20,
                             epochs_per_trial: int = 50) -> Dict:
        """
        Perform hyperparameter tuning using random search.
        
        Args:
            prepared_data: Data prepared by prepare_data method
            param_grid: Dictionary of hyperparameter ranges
            n_trials: Number of hyperparameter combinations to try
            epochs_per_trial: Number of epochs per trial
            
        Returns:
            Best hyperparameters and tuning results
        """
        self.logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        
        best_score = 0.0
        best_params = None
        tuning_results = []
        
        for trial in range(n_trials):
            # Sample hyperparameters
            trial_params = {}
            for param, values in param_grid.items():
                if isinstance(values, list):
                    trial_params[param] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    if isinstance(values[0], int):
                        trial_params[param] = np.random.randint(values[0], values[1] + 1)
                    else:
                        trial_params[param] = np.random.uniform(values[0], values[1])
            
            self.logger.info(f"Trial {trial + 1}/{n_trials}: {trial_params}")
            
            # Update model parameters
            for param, value in trial_params.items():
                setattr(self, param, value)
            
            # Initialize model with new parameters
            input_dim = prepared_data['train_data'].x.size(1)
            edge_dim = prepared_data['train_data'].edge_attr.size(1) if prepared_data['train_data'].edge_attr is not None else None
            self.initialize_model(input_dim, edge_dim)
            
            # Train model
            try:
                training_results = self.train(
                    prepared_data, 
                    epochs=epochs_per_trial, 
                    early_stopping_patience=15,
                    verbose=False
                )
                
                val_score = max(training_results['history']['val_f1'])
                
                trial_result = {
                    'trial': trial + 1,
                    'params': trial_params.copy(),
                    'val_f1': val_score,
                    'test_metrics': training_results['final_test_metrics']
                }
                
                tuning_results.append(trial_result)
                
                if val_score > best_score:
                    best_score = val_score
                    best_params = trial_params.copy()
                    self.logger.info(f"New best score: {best_score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Trial {trial + 1} failed: {str(e)}")
                continue
        
        # Set best parameters
        if best_params:
            for param, value in best_params.items():
                setattr(self, param, value)
            self.logger.info(f"Best parameters: {best_params}")
        
        self.tuning_history = tuning_results
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'tuning_results': tuning_results
        }
    
    def train(self, 
             prepared_data: Dict,
             epochs: int = 100,
             early_stopping_patience: int = 15,
             verbose: bool = True) -> Dict:
        """
        Train the GNN model with validation and early stopping.
        
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
            edge_dim = prepared_data['train_data'].edge_attr.size(1) if prepared_data['train_data'].edge_attr is not None else None
            self.initialize_model(input_dim, edge_dim)
        
        train_data = prepared_data['train_data']
        val_data = prepared_data['val_data']
        
        # Training history
        history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rates': []
        }
        
        best_val_f1 = 0.0
        patience_counter = 0
        
        self.logger.info(f"Starting {self.gnn_type} training for {epochs} epochs...")
        
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
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['f1_score'])
            
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
                    f"Val AUC: {val_metrics['auc_roc']:.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
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
                data.x, data.edge_index, edge_index, data.edge_attr
            ).cpu().numpy()
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        if self.model is None:
            return {"status": "Model not initialized"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": f"GNN-{self.gnn_type}",
            "gnn_type": self.gnn_type,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "tuning_trials": len(self.tuning_history)
        }
    
    def get_tuning_summary(self) -> Dict:
        """Get summary of hyperparameter tuning results."""
        if not self.tuning_history:
            return {"status": "No tuning performed"}
        
        # Find best trial
        best_trial = max(self.tuning_history, key=lambda x: x['val_f1'])
        
        # Calculate statistics
        val_f1_scores = [trial['val_f1'] for trial in self.tuning_history]
        
        return {
            "total_trials": len(self.tuning_history),
            "best_val_f1": best_trial['val_f1'],
            "best_params": best_trial['params'],
            "best_test_metrics": best_trial['test_metrics'],
            "val_f1_mean": np.mean(val_f1_scores),
            "val_f1_std": np.std(val_f1_scores),
            "val_f1_range": (min(val_f1_scores), max(val_f1_scores))
        }