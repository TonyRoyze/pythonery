"""
Traditional machine learning models for link prediction in social networks.

This module implements traditional ML approaches (Random Forest, SVM) as baselines
for comparison with graph-based models. Includes feature engineering and
performance comparison framework.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

from .link_prediction_data import LinkPredictionDataPipeline, NodePairFeatureExtractor


class TraditionalMLPredictor:
    """
    Traditional machine learning predictor for link prediction.
    
    Implements Random Forest, SVM, and Logistic Regression models with
    comprehensive feature engineering and performance evaluation capabilities.
    """
    
    def __init__(self, 
                 models: Optional[List[str]] = None,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize traditional ML predictor.
        
        Args:
            models: List of models to use ('rf', 'svm', 'lr')
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs for model training
        """
        if models is None:
            models = ['rf', 'svm', 'lr']
        
        self.models = models
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_pipeline = LinkPredictionDataPipeline(random_state=random_state)
        self.feature_extractor = NodePairFeatureExtractor()
        self.scaler = StandardScaler()
        
        # Model storage
        self.trained_models = {}
        self.model_performances = {}
        self.feature_names = []
        self.best_model = None
        self.best_model_name = None
        
        # Initialize model configurations
        self.model_configs = self._get_model_configurations()
    
    def _get_model_configurations(self) -> Dict[str, Dict]:
        """Get default model configurations and hyperparameter grids."""
        return {
            'rf': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                ),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'name': 'Random Forest'
            },
            'svm': {
                'model': SVC(
                    random_state=self.random_state,
                    probability=True  # Enable probability estimates
                ),
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                },
                'name': 'Support Vector Machine'
            },
            'lr': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_iter=1000
                ),
                'param_grid': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]  # Only used with elasticnet
                },
                'name': 'Logistic Regression'
            }
        }
    
    def prepare_data(self,
                    graph: nx.Graph,
                    relationships_df: pd.DataFrame,
                    train_ratio: float = 0.6,
                    val_ratio: float = 0.2,
                    test_ratio: float = 0.2,
                    feature_types: Optional[List[str]] = None) -> Dict:
        """
        Prepare data for traditional ML training.
        
        Args:
            graph: NetworkX graph
            relationships_df: DataFrame with relationship data
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            feature_types: Types of features to extract
            
        Returns:
            Dictionary containing prepared datasets with features
        """
        self.logger.info("Preparing data for traditional ML training...")
        
        # Use the existing data pipeline for temporal splitting and negative sampling
        dataset = self.data_pipeline.prepare_link_prediction_data(
            graph=graph,
            relationships_df=relationships_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            negative_sampling_strategy='random',
            feature_types=feature_types
        )
        
        # Extract edge lists
        train_pos_edges = dataset['train']['positive_edges']
        train_neg_edges = dataset['train']['negative_edges']
        val_pos_edges = dataset['val']['positive_edges']
        val_neg_edges = dataset['val']['negative_edges']
        test_pos_edges = dataset['test']['positive_edges']
        test_neg_edges = dataset['test']['negative_edges']
        
        # Create balanced datasets
        train_edges = train_pos_edges + train_neg_edges
        train_labels = [1] * len(train_pos_edges) + [0] * len(train_neg_edges)
        
        val_edges = val_pos_edges + val_neg_edges
        val_labels = [1] * len(val_pos_edges) + [0] * len(val_neg_edges)
        
        test_edges = test_pos_edges + test_neg_edges
        test_labels = [1] * len(test_pos_edges) + [0] * len(test_neg_edges)
        
        # Extract features for each dataset
        self.logger.info("Extracting features for node pairs...")
        
        # For traditional ML, we can use the full graph for feature extraction
        # since we're not doing message passing like in graph neural networks
        # The temporal split is handled by the edge selection, not graph structure
        
        # Extract features using the full graph for all splits
        train_features = self.feature_extractor.extract_features(
            graph, train_edges, feature_types
        )
        val_features = self.feature_extractor.extract_features(
            graph, val_edges, feature_types
        )
        test_features = self.feature_extractor.extract_features(
            graph, test_edges, feature_types
        )
        
        # Get feature names
        self.feature_names = self.feature_extractor.get_feature_names(feature_types)
        
        # Create DataFrames for easier handling
        train_df = pd.DataFrame(train_features, columns=self.feature_names)
        train_df['label'] = train_labels
        
        val_df = pd.DataFrame(val_features, columns=self.feature_names)
        val_df['label'] = val_labels
        
        test_df = pd.DataFrame(test_features, columns=self.feature_names)
        test_df['label'] = test_labels
        
        prepared_data = {
            'train': {
                'features': train_features,
                'labels': np.array(train_labels),
                'edges': train_edges,
                'dataframe': train_df
            },
            'val': {
                'features': val_features,
                'labels': np.array(val_labels),
                'edges': val_edges,
                'dataframe': val_df
            },
            'test': {
                'features': test_features,
                'labels': np.array(test_labels),
                'edges': test_edges,
                'dataframe': test_df
            },
            'feature_names': self.feature_names,
            'split_info': dataset['metadata']['split_info']
        }
        
        self.logger.info(f"Data prepared: {len(train_edges)} train, {len(val_edges)} val, "
                        f"{len(test_edges)} test samples with {len(self.feature_names)} features")
        
        return prepared_data
    
    def train_model(self,
                   model_type: str,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None,
                   hyperparameter_tuning: bool = True,
                   cv_folds: int = 5) -> Dict:
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            model_type: Type of model ('rf', 'svm', 'lr')
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing trained model and performance metrics
        """
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = self.model_configs[model_type]
        model_name = config['name']
        
        self.logger.info(f"Training {model_name}...")
        
        # Create pipeline with scaling
        if hyperparameter_tuning:
            # Use GridSearchCV for hyperparameter tuning
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', config['model'])
            ])
            
            # Adjust parameter names for pipeline
            param_grid = {}
            for param, values in config['param_grid'].items():
                param_grid[f'model__{param}'] = values
            
            # Handle special case for logistic regression elasticnet
            if model_type == 'lr':
                # Create separate grids for different penalty types
                param_grids = []
                
                # L1 and L2 penalties
                for penalty in ['l1', 'l2']:
                    grid = {
                        'model__C': config['param_grid']['C'],
                        'model__penalty': [penalty],
                        'model__solver': ['liblinear'] if penalty == 'l1' else ['liblinear', 'saga']
                    }
                    param_grids.append(grid)
                
                # Elasticnet penalty
                elasticnet_grid = {
                    'model__C': config['param_grid']['C'],
                    'model__penalty': ['elasticnet'],
                    'model__solver': ['saga'],
                    'model__l1_ratio': config['param_grid']['l1_ratio']
                }
                param_grids.append(elasticnet_grid)
                
                grid_search = GridSearchCV(
                    pipeline,
                    param_grids,
                    cv=cv_folds,
                    scoring='f1',
                    n_jobs=self.n_jobs,
                    verbose=0
                )
            else:
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=cv_folds,
                    scoring='f1',
                    n_jobs=self.n_jobs,
                    verbose=0
                )
            
            # Fit the grid search
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
            self.logger.info(f"Best {model_name} parameters: {best_params}")
            self.logger.info(f"Best CV F1 score: {cv_score:.4f}")
            
        else:
            # Train with default parameters
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', config['model'])
            ])
            
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
            
            # Calculate CV score with default parameters
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='f1')
            cv_score = cv_scores.mean()
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_predictions = best_model.predict(X_val)
            val_probabilities = best_model.predict_proba(X_val)[:, 1]
            
            val_metrics = {
                'accuracy': accuracy_score(y_val, val_predictions),
                'precision': precision_score(y_val, val_predictions, zero_division=0),
                'recall': recall_score(y_val, val_predictions, zero_division=0),
                'f1_score': f1_score(y_val, val_predictions, zero_division=0),
                'auc_roc': roc_auc_score(y_val, val_probabilities) if len(np.unique(y_val)) > 1 else 0.0
            }
        
        # Store the trained model
        self.trained_models[model_type] = best_model
        
        training_result = {
            'model': best_model,
            'model_type': model_type,
            'model_name': model_name,
            'best_params': best_params,
            'cv_f1_score': cv_score,
            'val_metrics': val_metrics
        }
        
        self.logger.info(f"{model_name} training completed!")
        if val_metrics:
            self.logger.info(f"Validation F1: {val_metrics['f1_score']:.4f}, "
                           f"AUC: {val_metrics['auc_roc']:.4f}")
        
        return training_result
    
    def train_all_models(self,
                        prepared_data: Dict,
                        hyperparameter_tuning: bool = True,
                        cv_folds: int = 5) -> Dict:
        """
        Train all specified models.
        
        Args:
            prepared_data: Data prepared by prepare_data method
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing all training results
        """
        X_train = prepared_data['train']['features']
        y_train = prepared_data['train']['labels']
        X_val = prepared_data['val']['features']
        y_val = prepared_data['val']['labels']
        
        training_results = {}
        
        for model_type in self.models:
            try:
                result = self.train_model(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    hyperparameter_tuning=hyperparameter_tuning,
                    cv_folds=cv_folds
                )
                training_results[model_type] = result
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_type}: {str(e)}")
                continue
        
        # Find best model based on validation F1 score
        best_f1 = 0.0
        for model_type, result in training_results.items():
            val_f1 = result['val_metrics'].get('f1_score', 0.0)
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.best_model = result['model']
                self.best_model_name = result['model_name']
        
        self.logger.info(f"Best model: {self.best_model_name} (F1: {best_f1:.4f})")
        
        return training_results
    
    def evaluate_models(self,
                       prepared_data: Dict,
                       training_results: Dict) -> Dict:
        """
        Evaluate all trained models on test data.
        
        Args:
            prepared_data: Data prepared by prepare_data method
            training_results: Results from train_all_models
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        X_test = prepared_data['test']['features']
        y_test = prepared_data['test']['labels']
        
        evaluation_results = {}
        
        for model_type, training_result in training_results.items():
            model = training_result['model']
            model_name = training_result['model_name']
            
            self.logger.info(f"Evaluating {model_name} on test data...")
            
            # Make predictions
            test_predictions = model.predict(X_test)
            test_probabilities = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            test_metrics = {
                'accuracy': accuracy_score(y_test, test_predictions),
                'precision': precision_score(y_test, test_predictions, zero_division=0),
                'recall': recall_score(y_test, test_predictions, zero_division=0),
                'f1_score': f1_score(y_test, test_predictions, zero_division=0),
                'auc_roc': roc_auc_score(y_test, test_probabilities) if len(np.unique(y_test)) > 1 else 0.0
            }
            
            # Generate classification report
            class_report = classification_report(
                y_test, test_predictions, 
                target_names=['No Link', 'Link'],
                output_dict=True
            )
            
            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_test, test_predictions)
            
            evaluation_results[model_type] = {
                'model_name': model_name,
                'test_metrics': test_metrics,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'predictions': test_predictions,
                'probabilities': test_probabilities
            }
            
            # Store performance for comparison
            self.model_performances[model_type] = test_metrics
            
            self.logger.info(f"{model_name} test results: "
                           f"F1: {test_metrics['f1_score']:.4f}, "
                           f"AUC: {test_metrics['auc_roc']:.4f}")
        
        return evaluation_results
    
    def get_feature_importance(self, model_type: str) -> Dict[str, float]:
        """
        Get feature importance for a trained model.
        
        Args:
            model_type: Type of model to analyze
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if model_type not in self.trained_models:
            raise ValueError(f"Model {model_type} has not been trained")
        
        model = self.trained_models[model_type]
        
        # Extract the actual model from the pipeline
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps['model']
        else:
            actual_model = model
        
        # Get feature importance based on model type
        if hasattr(actual_model, 'feature_importances_'):
            # Random Forest
            importances = actual_model.feature_importances_
        elif hasattr(actual_model, 'coef_'):
            # Logistic Regression, SVM
            importances = np.abs(actual_model.coef_[0])
        else:
            self.logger.warning(f"Cannot extract feature importance for {model_type}")
            return {}
        
        # Create feature importance dictionary
        feature_importance = dict(zip(self.feature_names, importances))
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return feature_importance
    
    def compare_with_graph_models(self,
                                 graph_model_results: Dict,
                                 traditional_results: Dict) -> Dict:
        """
        Compare traditional ML models with graph-based models.
        
        Args:
            graph_model_results: Results from graph-based models (GraphSAGE, GNN)
            traditional_results: Results from traditional ML models
            
        Returns:
            Comprehensive comparison results
        """
        self.logger.info("Comparing traditional ML with graph-based models...")
        
        comparison_results = {
            'model_comparison': {},
            'performance_summary': {},
            'best_overall_model': None,
            'insights': []
        }
        
        # Collect all model results
        all_models = {}
        
        # Add traditional ML results
        for model_type, result in traditional_results.items():
            model_name = result['model_name']
            test_metrics = result['test_metrics']
            all_models[f"Traditional_{model_name}"] = {
                'type': 'traditional',
                'metrics': test_metrics,
                'model_type': model_type
            }
        
        # Add graph-based model results
        for model_name, result in graph_model_results.items():
            if 'final_test_metrics' in result:
                test_metrics = result['final_test_metrics']
            elif 'test_metrics' in result:
                test_metrics = result['test_metrics']
            else:
                continue
            
            all_models[f"Graph_{model_name}"] = {
                'type': 'graph',
                'metrics': test_metrics,
                'model_type': model_name
            }
        
        # Create comparison DataFrame
        metrics_df = pd.DataFrame({
            model_name: model_data['metrics'] 
            for model_name, model_data in all_models.items()
        }).T
        
        comparison_results['model_comparison'] = metrics_df.to_dict()
        
        # Find best model for each metric
        best_models = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            if metric in metrics_df.columns:
                best_model = metrics_df[metric].idxmax()
                best_score = metrics_df[metric].max()
                best_models[metric] = {
                    'model': best_model,
                    'score': best_score
                }
        
        comparison_results['best_per_metric'] = best_models
        
        # Overall best model (based on F1 score)
        if 'f1_score' in metrics_df.columns:
            overall_best = metrics_df['f1_score'].idxmax()
            comparison_results['best_overall_model'] = {
                'model': overall_best,
                'f1_score': metrics_df.loc[overall_best, 'f1_score'],
                'type': all_models[overall_best]['type']
            }
        
        # Generate insights
        insights = []
        
        # Compare traditional vs graph-based performance
        traditional_f1_scores = [
            model_data['metrics']['f1_score'] 
            for model_data in all_models.values() 
            if model_data['type'] == 'traditional'
        ]
        graph_f1_scores = [
            model_data['metrics']['f1_score'] 
            for model_data in all_models.values() 
            if model_data['type'] == 'graph'
        ]
        
        if traditional_f1_scores and graph_f1_scores:
            avg_traditional_f1 = np.mean(traditional_f1_scores)
            avg_graph_f1 = np.mean(graph_f1_scores)
            
            if avg_graph_f1 > avg_traditional_f1:
                insights.append(
                    f"Graph-based models outperform traditional ML on average "
                    f"(F1: {avg_graph_f1:.4f} vs {avg_traditional_f1:.4f})"
                )
            else:
                insights.append(
                    f"Traditional ML models perform competitively with graph-based models "
                    f"(F1: {avg_traditional_f1:.4f} vs {avg_graph_f1:.4f})"
                )
        
        # Identify best traditional model
        if traditional_f1_scores:
            best_traditional_idx = np.argmax(traditional_f1_scores)
            traditional_models = [
                name for name, data in all_models.items() 
                if data['type'] == 'traditional'
            ]
            best_traditional = traditional_models[best_traditional_idx]
            insights.append(f"Best traditional model: {best_traditional}")
        
        comparison_results['insights'] = insights
        comparison_results['performance_summary'] = {
            'traditional_avg_f1': np.mean(traditional_f1_scores) if traditional_f1_scores else 0,
            'graph_avg_f1': np.mean(graph_f1_scores) if graph_f1_scores else 0,
            'total_models_compared': len(all_models)
        }
        
        self.logger.info("Model comparison completed!")
        for insight in insights:
            self.logger.info(f"Insight: {insight}")
        
        return comparison_results
    
    def predict_links(self,
                     graph: nx.Graph,
                     node_pairs: List[Tuple],
                     model_type: Optional[str] = None) -> np.ndarray:
        """
        Predict link probabilities for given node pairs.
        
        Args:
            graph: NetworkX graph
            node_pairs: List of node pairs to predict
            model_type: Specific model to use (default: best model)
            
        Returns:
            Array of link probabilities
        """
        # Use best model if not specified
        if model_type is None:
            if self.best_model is None:
                raise ValueError("No trained models available")
            model = self.best_model
        else:
            if model_type not in self.trained_models:
                raise ValueError(f"Model {model_type} has not been trained")
            model = self.trained_models[model_type]
        
        # Extract features for node pairs
        features = self.feature_extractor.extract_features(graph, node_pairs)
        
        # Make predictions
        probabilities = model.predict_proba(features)[:, 1]
        
        return probabilities
    
    def save_models(self, filepath: str):
        """Save all trained models to disk."""
        save_data = {
            'trained_models': self.trained_models,
            'model_performances': self.model_performances,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler
        }
        
        joblib.dump(save_data, filepath)
        self.logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk."""
        save_data = joblib.load(filepath)
        
        self.trained_models = save_data['trained_models']
        self.model_performances = save_data['model_performances']
        self.feature_names = save_data['feature_names']
        self.best_model_name = save_data['best_model_name']
        self.scaler = save_data['scaler']
        
        # Set best model
        if self.best_model_name:
            for model_type, model in self.trained_models.items():
                if self.model_configs[model_type]['name'] == self.best_model_name:
                    self.best_model = model
                    break
        
        self.logger.info(f"Models loaded from {filepath}")
    
    def get_model_summary(self) -> Dict:
        """Get summary of all trained models."""
        summary = {
            'trained_models': list(self.trained_models.keys()),
            'best_model': self.best_model_name,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_performances': self.model_performances
        }
        
        return summary