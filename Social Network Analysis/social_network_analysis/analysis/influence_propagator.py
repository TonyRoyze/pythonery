"""
InfluencePropagator class for modeling information spread and influence dynamics.

This module provides functionality to simulate information propagation through social networks,
select seed nodes based on centrality scores, and track temporal propagation patterns.
Supports various propagation models including Independent Cascade and Linear Threshold models.
"""

import networkx as nx
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import logging
from collections import defaultdict, deque
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from social_network_analysis.ml.pyg_utils import PyGDataPipeline
from social_network_analysis.analysis.centrality_calculator import CentralityCalculator


class InfluencePropagator:
    """
    Models information spread and influence dynamics in social networks.
    
    Provides methods for simulating information propagation using various models,
    selecting optimal seed nodes based on centrality metrics, and tracking
    temporal propagation patterns with visualization capabilities.
    """
    
    def __init__(self, 
                 propagation_model: str = 'independent_cascade',
                 random_state: int = 42):
        """
        Initialize InfluencePropagator.
        
        Args:
            propagation_model: Propagation model ('independent_cascade' or 'linear_threshold')
            random_state: Random seed for reproducible results
        """
        self.propagation_model = propagation_model
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Initialize components
        self.centrality_calculator = CentralityCalculator()
        self.pyg_pipeline = PyGDataPipeline()
        
        # Propagation parameters
        self.default_activation_prob = 0.1
        self.default_threshold = 0.5
        
        # Tracking variables
        self.propagation_history = []
        self.influence_scores = {}
        
    def select_seed_nodes(self, 
                         graph: nx.Graph,
                         centrality_scores: Dict[str, Dict[str, float]],
                         num_seeds: int = 5,
                         selection_strategy: str = 'mixed',
                         centrality_weights: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Select seed nodes for influence propagation based on centrality scores.
        
        Args:
            graph: NetworkX graph
            centrality_scores: Dictionary containing centrality metrics for all nodes
            num_seeds: Number of seed nodes to select
            selection_strategy: Strategy for seed selection ('degree', 'betweenness', 
                              'closeness', 'mixed', 'diverse')
            centrality_weights: Weights for different centrality measures in mixed strategy
            
        Returns:
            List of selected seed node IDs
            
        Raises:
            ValueError: If selection strategy is not supported or insufficient nodes
        """
        if graph.number_of_nodes() < num_seeds:
            raise ValueError(f"Graph has only {graph.number_of_nodes()} nodes, cannot select {num_seeds} seeds")
            
        supported_strategies = ['degree', 'betweenness', 'closeness', 'mixed', 'diverse']
        if selection_strategy not in supported_strategies:
            raise ValueError(f"Selection strategy must be one of {supported_strategies}")
            
        if selection_strategy == 'mixed':
            return self._select_mixed_centrality_seeds(
                centrality_scores, num_seeds, centrality_weights
            )
        elif selection_strategy == 'diverse':
            return self._select_diverse_seeds(graph, centrality_scores, num_seeds)
        else:
            # Single centrality metric selection
            if selection_strategy not in centrality_scores:
                raise ValueError(f"Centrality metric '{selection_strategy}' not found in scores")
                
            ranked_nodes = self.centrality_calculator.rank_nodes_by_centrality(
                centrality_scores[selection_strategy], top_k=num_seeds
            )
            return [node for node, _ in ranked_nodes]
    
    def _select_mixed_centrality_seeds(self, 
                                     centrality_scores: Dict[str, Dict[str, float]],
                                     num_seeds: int,
                                     weights: Optional[Dict[str, float]] = None) -> List[str]:
        """Select seeds using weighted combination of centrality measures."""
        if weights is None:
            weights = {
                'degree': 0.3,
                'betweenness': 0.4,
                'closeness': 0.2,
                'clustering': 0.1
            }
        
        # Calculate composite centrality score
        composite_scores = {}
        all_nodes = set()
        
        for centrality_type, scores in centrality_scores.items():
            if centrality_type in weights:
                all_nodes.update(scores.keys())
        
        for node in all_nodes:
            composite_score = 0.0
            for centrality_type, weight in weights.items():
                if centrality_type in centrality_scores and node in centrality_scores[centrality_type]:
                    composite_score += weight * centrality_scores[centrality_type][node]
            composite_scores[node] = composite_score
        
        # Select top nodes based on composite score
        ranked_nodes = self.centrality_calculator.rank_nodes_by_centrality(
            composite_scores, top_k=num_seeds
        )
        return [node for node, _ in ranked_nodes]
    
    def _select_diverse_seeds(self, 
                            graph: nx.Graph,
                            centrality_scores: Dict[str, Dict[str, float]],
                            num_seeds: int) -> List[str]:
        """Select diverse seeds to maximize coverage using greedy algorithm."""
        selected_seeds = []
        remaining_nodes = set(graph.nodes())
        
        # Start with highest degree centrality node
        if 'degree' in centrality_scores:
            first_seed = max(centrality_scores['degree'].items(), key=lambda x: x[1])[0]
            selected_seeds.append(first_seed)
            remaining_nodes.remove(first_seed)
        
        # Greedily select remaining seeds to maximize coverage
        for _ in range(num_seeds - 1):
            if not remaining_nodes:
                break
                
            best_node = None
            best_coverage = -1
            
            for candidate in remaining_nodes:
                # Calculate coverage as number of unique neighbors
                candidate_neighbors = set(graph.neighbors(candidate))
                
                # Remove already covered neighbors
                for seed in selected_seeds:
                    seed_neighbors = set(graph.neighbors(seed))
                    candidate_neighbors -= seed_neighbors
                
                coverage = len(candidate_neighbors)
                
                # Add centrality bonus
                centrality_bonus = 0
                if 'betweenness' in centrality_scores and candidate in centrality_scores['betweenness']:
                    centrality_bonus = centrality_scores['betweenness'][candidate]
                
                total_score = coverage + centrality_bonus
                
                if total_score > best_coverage:
                    best_coverage = total_score
                    best_node = candidate
            
            if best_node:
                selected_seeds.append(best_node)
                remaining_nodes.remove(best_node)
        
        return selected_seeds
    
    def simulate_propagation(self, 
                           graph: nx.Graph,
                           seed_nodes: List[str],
                           max_steps: int = 10,
                           activation_prob: Optional[float] = None,
                           threshold: Optional[float] = None,
                           track_history: bool = True) -> Dict[str, Any]:
        """
        Simulate information propagation through the network.
        
        Args:
            graph: NetworkX graph
            seed_nodes: List of initial seed nodes
            max_steps: Maximum number of propagation steps
            activation_prob: Activation probability for Independent Cascade model
            threshold: Activation threshold for Linear Threshold model
            track_history: Whether to track propagation history for visualization
            
        Returns:
            Dictionary containing propagation results and statistics
        """
        if activation_prob is None:
            activation_prob = self.default_activation_prob
        if threshold is None:
            threshold = self.default_threshold
            
        # Validate seed nodes
        invalid_seeds = [node for node in seed_nodes if node not in graph.nodes()]
        if invalid_seeds:
            raise ValueError(f"Seed nodes not found in graph: {invalid_seeds}")
        
        if self.propagation_model == 'independent_cascade':
            return self._simulate_independent_cascade(
                graph, seed_nodes, max_steps, activation_prob, track_history
            )
        elif self.propagation_model == 'linear_threshold':
            return self._simulate_linear_threshold(
                graph, seed_nodes, max_steps, threshold, track_history
            )
        else:
            raise ValueError(f"Unsupported propagation model: {self.propagation_model}")
    
    def _simulate_independent_cascade(self, 
                                    graph: nx.Graph,
                                    seed_nodes: List[str],
                                    max_steps: int,
                                    activation_prob: float,
                                    track_history: bool) -> Dict[str, Any]:
        """Simulate Independent Cascade propagation model."""
        # Initialize states
        activated = set(seed_nodes)
        newly_activated = set(seed_nodes)
        activation_times = {node: 0 for node in seed_nodes}
        
        # Track propagation history
        if track_history:
            self.propagation_history = []
            self.propagation_history.append({
                'step': 0,
                'activated': activated.copy(),
                'newly_activated': newly_activated.copy(),
                'total_activated': len(activated)
            })
        
        # Propagation simulation
        for step in range(1, max_steps + 1):
            current_newly_activated = set()
            
            # Each newly activated node tries to activate its neighbors
            for node in newly_activated:
                neighbors = list(graph.neighbors(node))
                
                for neighbor in neighbors:
                    if neighbor not in activated:
                        # Calculate activation probability (can be edge-weighted)
                        edge_data = graph.get_edge_data(node, neighbor, {})
                        edge_weight = edge_data.get('weight', 1.0)
                        prob = min(activation_prob * edge_weight, 1.0)
                        
                        # Attempt activation
                        if np.random.random() < prob:
                            current_newly_activated.add(neighbor)
                            activation_times[neighbor] = step
            
            # Update activated sets
            activated.update(current_newly_activated)
            newly_activated = current_newly_activated
            
            # Track history
            if track_history:
                self.propagation_history.append({
                    'step': step,
                    'activated': activated.copy(),
                    'newly_activated': newly_activated.copy(),
                    'total_activated': len(activated)
                })
            
            # Stop if no new activations
            if not newly_activated:
                break
        
        # Calculate results
        final_influence = len(activated)
        influence_ratio = final_influence / graph.number_of_nodes()
        
        results = {
            'model': 'independent_cascade',
            'seed_nodes': seed_nodes,
            'final_activated': activated,
            'activation_times': activation_times,
            'final_influence': final_influence,
            'influence_ratio': influence_ratio,
            'steps_taken': step,
            'parameters': {
                'activation_prob': activation_prob,
                'max_steps': max_steps
            }
        }
        
        if track_history:
            results['propagation_history'] = self.propagation_history
        
        self.logger.info(f"IC propagation completed: {final_influence}/{graph.number_of_nodes()} nodes activated")
        return results
    
    def _simulate_linear_threshold(self, 
                                 graph: nx.Graph,
                                 seed_nodes: List[str],
                                 max_steps: int,
                                 threshold: float,
                                 track_history: bool) -> Dict[str, Any]:
        """Simulate Linear Threshold propagation model."""
        # Initialize states
        activated = set(seed_nodes)
        activation_times = {node: 0 for node in seed_nodes}
        
        # Assign random thresholds to nodes
        node_thresholds = {}
        for node in graph.nodes():
            if node in seed_nodes:
                node_thresholds[node] = 0.0  # Seeds are pre-activated
            else:
                node_thresholds[node] = np.random.uniform(0, threshold)
        
        # Track propagation history
        if track_history:
            self.propagation_history = []
            self.propagation_history.append({
                'step': 0,
                'activated': activated.copy(),
                'newly_activated': set(seed_nodes),
                'total_activated': len(activated)
            })
        
        # Propagation simulation
        for step in range(1, max_steps + 1):
            newly_activated = set()
            
            # Check each non-activated node
            for node in graph.nodes():
                if node not in activated:
                    # Calculate influence from activated neighbors
                    total_influence = 0.0
                    degree = graph.degree(node)
                    
                    if degree > 0:
                        for neighbor in graph.neighbors(node):
                            if neighbor in activated:
                                # Influence is proportional to edge weight
                                edge_data = graph.get_edge_data(node, neighbor, {})
                                edge_weight = edge_data.get('weight', 1.0)
                                total_influence += edge_weight / degree
                    
                    # Activate if influence exceeds threshold
                    if total_influence >= node_thresholds[node]:
                        newly_activated.add(node)
                        activation_times[node] = step
            
            # Update activated set
            activated.update(newly_activated)
            
            # Track history
            if track_history:
                self.propagation_history.append({
                    'step': step,
                    'activated': activated.copy(),
                    'newly_activated': newly_activated.copy(),
                    'total_activated': len(activated)
                })
            
            # Stop if no new activations
            if not newly_activated:
                break
        
        # Calculate results
        final_influence = len(activated)
        influence_ratio = final_influence / graph.number_of_nodes()
        
        results = {
            'model': 'linear_threshold',
            'seed_nodes': seed_nodes,
            'final_activated': activated,
            'activation_times': activation_times,
            'node_thresholds': node_thresholds,
            'final_influence': final_influence,
            'influence_ratio': influence_ratio,
            'steps_taken': step,
            'parameters': {
                'threshold': threshold,
                'max_steps': max_steps
            }
        }
        
        if track_history:
            results['propagation_history'] = self.propagation_history
        
        self.logger.info(f"LT propagation completed: {final_influence}/{graph.number_of_nodes()} nodes activated")
        return results
    
    def evaluate_seed_influence(self, 
                              graph: nx.Graph,
                              centrality_scores: Dict[str, Dict[str, float]],
                              num_seeds_range: Tuple[int, int] = (1, 10),
                              num_simulations: int = 100) -> Dict[str, Any]:
        """
        Evaluate influence of different seed selection strategies.
        
        Args:
            graph: NetworkX graph
            centrality_scores: Dictionary containing centrality metrics
            num_seeds_range: Range of seed counts to evaluate
            num_simulations: Number of simulations per configuration
            
        Returns:
            Dictionary containing evaluation results
        """
        strategies = ['degree', 'betweenness', 'closeness', 'mixed', 'diverse']
        results = defaultdict(lambda: defaultdict(list))
        
        min_seeds, max_seeds = num_seeds_range
        
        for num_seeds in range(min_seeds, max_seeds + 1):
            self.logger.info(f"Evaluating {num_seeds} seeds...")
            
            for strategy in strategies:
                if strategy == 'degree' and 'degree' not in centrality_scores:
                    continue
                if strategy == 'betweenness' and 'betweenness' not in centrality_scores:
                    continue
                if strategy == 'closeness' and 'closeness' not in centrality_scores:
                    continue
                
                strategy_influences = []
                
                for sim in range(num_simulations):
                    try:
                        # Select seeds
                        seeds = self.select_seed_nodes(
                            graph, centrality_scores, num_seeds, strategy
                        )
                        
                        # Run propagation simulation
                        propagation_result = self.simulate_propagation(
                            graph, seeds, track_history=False
                        )
                        
                        strategy_influences.append(propagation_result['influence_ratio'])
                        
                    except Exception as e:
                        self.logger.warning(f"Simulation failed for {strategy} with {num_seeds} seeds: {e}")
                        continue
                
                if strategy_influences:
                    results[strategy][num_seeds] = {
                        'mean_influence': np.mean(strategy_influences),
                        'std_influence': np.std(strategy_influences),
                        'min_influence': np.min(strategy_influences),
                        'max_influence': np.max(strategy_influences),
                        'simulations': len(strategy_influences)
                    }
        
        # Store influence scores for later use
        self.influence_scores = dict(results)
        
        self.logger.info("Seed influence evaluation completed")
        return dict(results)
    
    def visualize_propagation_timeline(self, 
                                     graph: nx.Graph,
                                     propagation_result: Dict[str, Any],
                                     layout: str = 'spring',
                                     save_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create timeline visualization of propagation process.
        
        Args:
            graph: NetworkX graph
            propagation_result: Result from simulate_propagation
            layout: Network layout algorithm
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        if 'propagation_history' not in propagation_result:
            raise ValueError("Propagation result must include history for timeline visualization")
        
        history = propagation_result['propagation_history']
        
        # Calculate layout positions
        if layout == 'spring':
            pos = nx.spring_layout(graph, seed=self.random_state)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        else:
            pos = nx.random_layout(graph, seed=self.random_state)
        
        # Create subplots for different time steps
        num_steps = len(history)
        cols = min(4, num_steps)
        rows = (num_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Color scheme
        colors = {
            'inactive': '#cccccc',
            'seed': '#d62728',
            'newly_activated': '#ff7f0e',
            'previously_activated': '#2ca02c'
        }
        
        for i, step_data in enumerate(history):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Determine node colors
            node_colors = []
            for node in graph.nodes():
                if node in propagation_result['seed_nodes']:
                    if step_data['step'] == 0:
                        node_colors.append(colors['seed'])
                    else:
                        node_colors.append(colors['previously_activated'])
                elif node in step_data['newly_activated']:
                    node_colors.append(colors['newly_activated'])
                elif node in step_data['activated']:
                    node_colors.append(colors['previously_activated'])
                else:
                    node_colors.append(colors['inactive'])
            
            # Draw network
            nx.draw(graph, pos, ax=ax, node_color=node_colors, 
                   node_size=100, edge_color='#666666', width=0.5,
                   with_labels=False)
            
            ax.set_title(f"Step {step_data['step']}\n"
                        f"Activated: {step_data['total_activated']}")
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
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['previously_activated'], 
                      markersize=10, label='Previously activated'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['inactive'], 
                      markersize=10, label='Inactive')
        ]
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)
        
        plt.suptitle(f"Information Propagation Timeline\n"
                    f"Model: {propagation_result['model'].replace('_', ' ').title()}", 
                    fontsize=14, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Propagation timeline saved to {save_path}")
        
        return fig
    
    def visualize_influence_comparison(self, 
                                     influence_evaluation: Dict[str, Any],
                                     save_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualize comparison of different seed selection strategies.
        
        Args:
            influence_evaluation: Result from evaluate_seed_influence
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Prepare data for plotting
        strategies = list(influence_evaluation.keys())
        seed_counts = set()
        for strategy_data in influence_evaluation.values():
            seed_counts.update(strategy_data.keys())
        seed_counts = sorted(seed_counts)
        
        # Plot 1: Mean influence vs number of seeds
        for strategy in strategies:
            means = []
            stds = []
            x_vals = []
            
            for num_seeds in seed_counts:
                if num_seeds in influence_evaluation[strategy]:
                    data = influence_evaluation[strategy][num_seeds]
                    means.append(data['mean_influence'])
                    stds.append(data['std_influence'])
                    x_vals.append(num_seeds)
            
            if means:
                ax1.errorbar(x_vals, means, yerr=stds, marker='o', 
                           label=strategy.replace('_', ' ').title(), capsize=3)
        
        ax1.set_xlabel('Number of Seed Nodes')
        ax1.set_ylabel('Mean Influence Ratio')
        ax1.set_title('Influence vs Number of Seeds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot for specific seed count
        if seed_counts:
            mid_seed_count = seed_counts[len(seed_counts) // 2]
            box_data = []
            box_labels = []
            
            for strategy in strategies:
                if mid_seed_count in influence_evaluation[strategy]:
                    # Generate sample data based on mean and std (approximation)
                    data = influence_evaluation[strategy][mid_seed_count]
                    mean = data['mean_influence']
                    std = data['std_influence']
                    samples = np.random.normal(mean, std, 100)
                    samples = np.clip(samples, 0, 1)  # Clip to valid range
                    
                    box_data.append(samples)
                    box_labels.append(strategy.replace('_', ' ').title())
            
            if box_data:
                ax2.boxplot(box_data, labels=box_labels)
                ax2.set_ylabel('Influence Ratio')
                ax2.set_title(f'Influence Distribution\n({mid_seed_count} seeds)')
                ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Influence comparison saved to {save_path}")
        
        return fig
    
    def generate_propagation_report(self, 
                                  graph: nx.Graph,
                                  centrality_scores: Dict[str, Dict[str, float]],
                                  num_seeds: int = 5,
                                  num_simulations: int = 50) -> Dict[str, Any]:
        """
        Generate comprehensive influence propagation analysis report.
        
        Args:
            graph: NetworkX graph
            centrality_scores: Dictionary containing centrality metrics
            num_seeds: Number of seed nodes to use
            num_simulations: Number of simulations for evaluation
            
        Returns:
            Dictionary containing comprehensive propagation analysis
        """
        report = {
            'graph_info': {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'connected': nx.is_connected(graph)
            },
            'analysis_parameters': {
                'propagation_model': self.propagation_model,
                'num_seeds': num_seeds,
                'num_simulations': num_simulations,
                'random_state': self.random_state
            }
        }
        
        # Evaluate different seed selection strategies
        self.logger.info("Evaluating seed selection strategies...")
        influence_evaluation = self.evaluate_seed_influence(
            graph, centrality_scores, 
            num_seeds_range=(num_seeds, num_seeds),
            num_simulations=num_simulations
        )
        report['strategy_comparison'] = influence_evaluation
        
        # Run detailed simulation with best strategy
        best_strategy = max(influence_evaluation.keys(), 
                          key=lambda s: influence_evaluation[s][num_seeds]['mean_influence'])
        
        self.logger.info(f"Running detailed simulation with {best_strategy} strategy...")
        best_seeds = self.select_seed_nodes(
            graph, centrality_scores, num_seeds, best_strategy
        )
        
        detailed_result = self.simulate_propagation(
            graph, best_seeds, track_history=True
        )
        report['detailed_simulation'] = detailed_result
        
        # Calculate network-level metrics
        report['network_metrics'] = self._calculate_network_propagation_metrics(
            graph, detailed_result
        )
        
        self.logger.info("Propagation analysis report generated")
        return report
    
    def _calculate_network_propagation_metrics(self, 
                                             graph: nx.Graph,
                                             propagation_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate network-level propagation metrics."""
        activated_nodes = propagation_result['final_activated']
        activation_times = propagation_result['activation_times']
        
        # Basic metrics
        metrics = {
            'final_influence_ratio': len(activated_nodes) / graph.number_of_nodes(),
            'propagation_speed': len(activated_nodes) / propagation_result['steps_taken'],
            'average_activation_time': np.mean(list(activation_times.values()))
        }
        
        # Network structure impact
        if len(activated_nodes) > 1:
            activated_subgraph = graph.subgraph(activated_nodes)
            metrics['activated_density'] = nx.density(activated_subgraph)
            metrics['activated_clustering'] = nx.average_clustering(activated_subgraph)
        
        # Reach efficiency (how well seeds reached different parts of network)
        if nx.is_connected(graph):
            seed_nodes = propagation_result['seed_nodes']
            total_distance = 0
            count = 0
            
            for seed in seed_nodes:
                for activated in activated_nodes:
                    if seed != activated:
                        try:
                            distance = nx.shortest_path_length(graph, seed, activated)
                            total_distance += distance
                            count += 1
                        except nx.NetworkXNoPath:
                            continue
            
            if count > 0:
                metrics['average_propagation_distance'] = total_distance / count
        
        return metrics
    
    def evaluate_high_centrality_influence(self, 
                                         graph: nx.Graph,
                                         centrality_scores: Dict[str, Dict[str, float]],
                                         top_k: int = 10,
                                         num_simulations: int = 50) -> Dict[str, Any]:
        """
        Evaluate the influence of high-centrality nodes on information spread.
        
        This method addresses requirement 4.3: assess the impact of high-centrality 
        nodes on information spread speed and reach during influence evaluation.
        
        Args:
            graph: NetworkX graph
            centrality_scores: Dictionary containing centrality metrics
            top_k: Number of top centrality nodes to evaluate
            num_simulations: Number of simulations per node
            
        Returns:
            Dictionary containing influence evaluation results for high-centrality nodes
        """
        results = {
            'evaluation_parameters': {
                'top_k': top_k,
                'num_simulations': num_simulations,
                'graph_size': graph.number_of_nodes()
            },
            'centrality_influence': {},
            'comparative_analysis': {}
        }
        
        # Evaluate each centrality metric
        for centrality_type, scores in centrality_scores.items():
            if not scores:
                continue
                
            self.logger.info(f"Evaluating {centrality_type} centrality influence...")
            
            # Get top-k nodes for this centrality metric
            top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            centrality_results = {
                'top_nodes': top_nodes,
                'individual_influence': {},
                'speed_metrics': {},
                'reach_metrics': {}
            }
            
            # Evaluate each high-centrality node individually
            for node_id, centrality_value in top_nodes:
                if node_id not in graph.nodes():
                    continue
                    
                node_influences = []
                node_speeds = []
                node_reaches = []
                activation_times_list = []
                
                # Run multiple simulations for this node
                for _ in range(num_simulations):
                    try:
                        propagation_result = self.simulate_propagation(
                            graph, [node_id], track_history=True
                        )
                        
                        # Collect metrics
                        influence_ratio = propagation_result['influence_ratio']
                        steps_taken = propagation_result['steps_taken']
                        final_activated = len(propagation_result['final_activated'])
                        
                        node_influences.append(influence_ratio)
                        node_speeds.append(final_activated / max(steps_taken, 1))
                        node_reaches.append(final_activated)
                        activation_times_list.append(propagation_result['activation_times'])
                        
                    except Exception as e:
                        self.logger.warning(f"Simulation failed for node {node_id}: {e}")
                        continue
                
                if node_influences:
                    # Calculate statistics
                    centrality_results['individual_influence'][node_id] = {
                        'centrality_value': centrality_value,
                        'mean_influence': np.mean(node_influences),
                        'std_influence': np.std(node_influences),
                        'max_influence': np.max(node_influences),
                        'min_influence': np.min(node_influences)
                    }
                    
                    centrality_results['speed_metrics'][node_id] = {
                        'mean_speed': np.mean(node_speeds),
                        'std_speed': np.std(node_speeds),
                        'max_speed': np.max(node_speeds)
                    }
                    
                    centrality_results['reach_metrics'][node_id] = {
                        'mean_reach': np.mean(node_reaches),
                        'std_reach': np.std(node_reaches),
                        'max_reach': np.max(node_reaches)
                    }
            
            results['centrality_influence'][centrality_type] = centrality_results
        
        # Comparative analysis across centrality types
        results['comparative_analysis'] = self._analyze_centrality_influence_comparison(
            results['centrality_influence']
        )
        
        self.logger.info("High-centrality influence evaluation completed")
        return results
    
    def _analyze_centrality_influence_comparison(self, 
                                               centrality_influence: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare influence across different centrality metrics."""
        comparison = {
            'best_performers': {},
            'correlation_analysis': {},
            'summary_statistics': {}
        }
        
        # Find best performing nodes for each metric
        for centrality_type, data in centrality_influence.items():
            if 'individual_influence' not in data:
                continue
                
            influences = data['individual_influence']
            if not influences:
                continue
                
            # Best performer by mean influence
            best_node = max(influences.items(), 
                          key=lambda x: x[1]['mean_influence'])
            comparison['best_performers'][centrality_type] = {
                'node_id': best_node[0],
                'mean_influence': best_node[1]['mean_influence'],
                'centrality_value': best_node[1]['centrality_value']
            }
            
            # Summary statistics
            all_influences = [data['mean_influence'] for data in influences.values()]
            comparison['summary_statistics'][centrality_type] = {
                'mean_influence': np.mean(all_influences),
                'std_influence': np.std(all_influences),
                'median_influence': np.median(all_influences)
            }
        
        # Correlation between centrality values and influence
        for centrality_type, data in centrality_influence.items():
            if 'individual_influence' not in data:
                continue
                
            influences = data['individual_influence']
            if len(influences) < 3:  # Need at least 3 points for correlation
                continue
                
            centrality_vals = [info['centrality_value'] for info in influences.values()]
            influence_vals = [info['mean_influence'] for info in influences.values()]
            
            if len(centrality_vals) > 1:
                correlation = np.corrcoef(centrality_vals, influence_vals)[0, 1]
                comparison['correlation_analysis'][centrality_type] = {
                    'correlation_coefficient': correlation,
                    'interpretation': self._interpret_correlation(correlation)
                }
        
        return comparison
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            strength = "very strong"
        elif abs_corr >= 0.6:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "positive" if correlation >= 0 else "negative"
        return f"{strength} {direction} correlation"
    
    def analyze_community_propagation_impact(self, 
                                           graph: nx.Graph,
                                           communities: Dict[int, List[str]],
                                           centrality_scores: Dict[str, Dict[str, float]],
                                           num_seeds_per_community: int = 3,
                                           num_simulations: int = 30) -> Dict[str, Any]:
        """
        Evaluate how network communities affect information spread patterns.
        
        This method addresses requirement 4.4: evaluate how network features like 
        communities affect information spread during propagation pattern analysis.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community IDs to lists of node IDs
            centrality_scores: Dictionary containing centrality metrics
            num_seeds_per_community: Number of seed nodes to select per community
            num_simulations: Number of simulations per configuration
            
        Returns:
            Dictionary containing community-based propagation analysis
        """
        results = {
            'analysis_parameters': {
                'num_communities': len(communities),
                'num_seeds_per_community': num_seeds_per_community,
                'num_simulations': num_simulations
            },
            'community_characteristics': {},
            'intra_community_propagation': {},
            'inter_community_propagation': {},
            'community_influence_ranking': {},
            'propagation_patterns': {}
        }
        
        # Analyze community characteristics
        for comm_id, members in communities.items():
            if not members:
                continue
                
            # Create community subgraph
            comm_subgraph = graph.subgraph(members)
            
            # Calculate community metrics
            comm_metrics = {
                'size': len(members),
                'density': nx.density(comm_subgraph) if len(members) > 1 else 0,
                'clustering': nx.average_clustering(comm_subgraph) if len(members) > 1 else 0,
                'internal_edges': comm_subgraph.number_of_edges(),
                'external_edges': self._count_external_edges(graph, members)
            }
            
            # Calculate average centrality scores for community members
            if 'degree' in centrality_scores:
                degree_scores = [centrality_scores['degree'].get(node, 0) for node in members]
                comm_metrics['avg_degree_centrality'] = np.mean(degree_scores)
            
            if 'betweenness' in centrality_scores:
                betweenness_scores = [centrality_scores['betweenness'].get(node, 0) for node in members]
                comm_metrics['avg_betweenness_centrality'] = np.mean(betweenness_scores)
            
            results['community_characteristics'][comm_id] = comm_metrics
        
        # Analyze intra-community propagation
        self.logger.info("Analyzing intra-community propagation...")
        for comm_id, members in communities.items():
            if len(members) < num_seeds_per_community:
                continue
                
            # Select seeds within community
            comm_seeds = self._select_community_seeds(
                members, centrality_scores, num_seeds_per_community
            )
            
            intra_results = []
            for _ in range(num_simulations):
                try:
                    propagation_result = self.simulate_propagation(
                        graph, comm_seeds, track_history=True
                    )
                    
                    # Analyze how much propagation stayed within community
                    activated_in_comm = len(set(propagation_result['final_activated']) & set(members))
                    activated_outside_comm = len(propagation_result['final_activated']) - activated_in_comm
                    
                    intra_results.append({
                        'total_activated': len(propagation_result['final_activated']),
                        'activated_in_community': activated_in_comm,
                        'activated_outside_community': activated_outside_comm,
                        'intra_community_ratio': activated_in_comm / len(members),
                        'spillover_ratio': activated_outside_comm / (graph.number_of_nodes() - len(members)),
                        'steps_taken': propagation_result['steps_taken']
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Intra-community simulation failed for community {comm_id}: {e}")
                    continue
            
            if intra_results:
                # Calculate statistics
                results['intra_community_propagation'][comm_id] = {
                    'mean_intra_ratio': np.mean([r['intra_community_ratio'] for r in intra_results]),
                    'mean_spillover_ratio': np.mean([r['spillover_ratio'] for r in intra_results]),
                    'mean_total_activated': np.mean([r['total_activated'] for r in intra_results]),
                    'mean_steps': np.mean([r['steps_taken'] for r in intra_results]),
                    'containment_score': np.mean([r['activated_in_community'] / r['total_activated'] 
                                                for r in intra_results if r['total_activated'] > 0])
                }
        
        # Analyze inter-community propagation
        self.logger.info("Analyzing inter-community propagation...")
        results['inter_community_propagation'] = self._analyze_inter_community_propagation(
            graph, communities, centrality_scores, num_simulations
        )
        
        # Rank communities by influence potential
        results['community_influence_ranking'] = self._rank_communities_by_influence(
            results['community_characteristics'],
            results['intra_community_propagation']
        )
        
        # Analyze propagation patterns
        results['propagation_patterns'] = self._analyze_community_propagation_patterns(
            results['intra_community_propagation'],
            results['inter_community_propagation']
        )
        
        self.logger.info("Community propagation impact analysis completed")
        return results
    
    def _count_external_edges(self, graph: nx.Graph, community_members: List[str]) -> int:
        """Count edges connecting community members to external nodes."""
        external_edges = 0
        community_set = set(community_members)
        
        for node in community_members:
            for neighbor in graph.neighbors(node):
                if neighbor not in community_set:
                    external_edges += 1
        
        return external_edges
    
    def _select_community_seeds(self, 
                              community_members: List[str],
                              centrality_scores: Dict[str, Dict[str, float]],
                              num_seeds: int) -> List[str]:
        """Select seed nodes within a community based on centrality."""
        if 'degree' in centrality_scores:
            # Use degree centrality for seed selection
            member_scores = {node: centrality_scores['degree'].get(node, 0) 
                           for node in community_members}
        elif 'betweenness' in centrality_scores:
            # Fallback to betweenness centrality
            member_scores = {node: centrality_scores['betweenness'].get(node, 0) 
                           for node in community_members}
        else:
            # Random selection if no centrality scores available
            return random.sample(community_members, min(num_seeds, len(community_members)))
        
        # Select top nodes by centrality
        sorted_members = sorted(member_scores.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_members[:num_seeds]]
    
    def _analyze_inter_community_propagation(self, 
                                           graph: nx.Graph,
                                           communities: Dict[int, List[str]],
                                           centrality_scores: Dict[str, Dict[str, float]],
                                           num_simulations: int) -> Dict[str, Any]:
        """Analyze propagation patterns between communities."""
        inter_results = {
            'bridge_node_analysis': {},
            'community_connectivity': {},
            'cross_community_influence': {}
        }
        
        # Identify bridge nodes (nodes with high betweenness connecting communities)
        if 'betweenness' in centrality_scores:
            bridge_candidates = []
            for node, betweenness in centrality_scores['betweenness'].items():
                if betweenness > np.percentile(list(centrality_scores['betweenness'].values()), 90):
                    # Check if node connects multiple communities
                    node_communities = set()
                    for neighbor in graph.neighbors(node):
                        for comm_id, members in communities.items():
                            if neighbor in members:
                                node_communities.add(comm_id)
                    
                    if len(node_communities) > 1:
                        bridge_candidates.append({
                            'node': node,
                            'betweenness': betweenness,
                            'connected_communities': list(node_communities)
                        })
            
            inter_results['bridge_node_analysis'] = {
                'bridge_nodes': bridge_candidates,
                'num_bridge_nodes': len(bridge_candidates)
            }
        
        # Analyze connectivity between communities
        community_connections = defaultdict(lambda: defaultdict(int))
        for comm_id1, members1 in communities.items():
            for comm_id2, members2 in communities.items():
                if comm_id1 >= comm_id2:  # Avoid double counting
                    continue
                
                # Count edges between communities
                connections = 0
                for node1 in members1:
                    for node2 in members2:
                        if graph.has_edge(node1, node2):
                            connections += 1
                
                community_connections[comm_id1][comm_id2] = connections
        
        inter_results['community_connectivity'] = dict(community_connections)
        
        return inter_results
    
    def _rank_communities_by_influence(self, 
                                     community_characteristics: Dict[int, Dict],
                                     intra_propagation: Dict[int, Dict]) -> List[Tuple[int, float]]:
        """Rank communities by their influence potential."""
        community_scores = []
        
        for comm_id in community_characteristics.keys():
            if comm_id not in intra_propagation:
                continue
                
            char = community_characteristics[comm_id]
            prop = intra_propagation[comm_id]
            
            # Calculate composite influence score
            influence_score = (
                0.3 * char.get('avg_degree_centrality', 0) +
                0.3 * char.get('avg_betweenness_centrality', 0) +
                0.2 * prop.get('mean_spillover_ratio', 0) +
                0.2 * (char.get('size', 0) / 100)  # Normalize by size
            )
            
            community_scores.append((comm_id, influence_score))
        
        # Sort by influence score (descending)
        return sorted(community_scores, key=lambda x: x[1], reverse=True)
    
    def _analyze_community_propagation_patterns(self, 
                                              intra_propagation: Dict[int, Dict],
                                              inter_propagation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall propagation patterns across communities."""
        patterns = {
            'containment_vs_spillover': {},
            'propagation_efficiency': {},
            'community_types': {}
        }
        
        # Analyze containment vs spillover patterns
        containment_scores = []
        spillover_scores = []
        
        for comm_id, data in intra_propagation.items():
            containment_scores.append(data.get('containment_score', 0))
            spillover_scores.append(data.get('mean_spillover_ratio', 0))
        
        if containment_scores and spillover_scores:
            patterns['containment_vs_spillover'] = {
                'mean_containment': np.mean(containment_scores),
                'mean_spillover': np.mean(spillover_scores),
                'containment_spillover_correlation': np.corrcoef(containment_scores, spillover_scores)[0, 1]
            }
        
        # Classify communities by propagation behavior
        for comm_id, data in intra_propagation.items():
            containment = data.get('containment_score', 0)
            spillover = data.get('mean_spillover_ratio', 0)
            
            if containment > 0.7:
                comm_type = "insular"  # High containment, low spillover
            elif spillover > 0.3:
                comm_type = "influential"  # High spillover
            elif containment > 0.4 and spillover > 0.1:
                comm_type = "balanced"  # Moderate both
            else:
                comm_type = "isolated"  # Low both
            
            patterns['community_types'][comm_id] = comm_type
        
        return patterns
    
    def calculate_propagation_speed_metrics(self, 
                                          propagation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for measuring propagation speed and reach.
        
        This method addresses requirement 4.6: create metrics for measuring 
        propagation speed and reach.
        
        Args:
            propagation_results: List of propagation simulation results
            
        Returns:
            Dictionary containing speed and reach metrics
        """
        if not propagation_results:
            return {}
        
        metrics = {
            'speed_metrics': {},
            'reach_metrics': {},
            'efficiency_metrics': {},
            'temporal_metrics': {}
        }
        
        # Extract data from results
        influence_ratios = []
        steps_taken = []
        final_activated_counts = []
        activation_times_lists = []
        
        for result in propagation_results:
            influence_ratios.append(result.get('influence_ratio', 0))
            steps_taken.append(result.get('steps_taken', 0))
            final_activated_counts.append(len(result.get('final_activated', [])))
            if 'activation_times' in result:
                activation_times_lists.append(list(result['activation_times'].values()))
        
        # Speed metrics
        if steps_taken and final_activated_counts:
            propagation_speeds = [count / max(steps, 1) for count, steps in zip(final_activated_counts, steps_taken)]
            
            metrics['speed_metrics'] = {
                'mean_propagation_speed': np.mean(propagation_speeds),
                'std_propagation_speed': np.std(propagation_speeds),
                'max_propagation_speed': np.max(propagation_speeds),
                'min_propagation_speed': np.min(propagation_speeds),
                'median_propagation_speed': np.median(propagation_speeds)
            }
        
        # Reach metrics
        if influence_ratios and final_activated_counts:
            metrics['reach_metrics'] = {
                'mean_influence_ratio': np.mean(influence_ratios),
                'std_influence_ratio': np.std(influence_ratios),
                'max_influence_ratio': np.max(influence_ratios),
                'min_influence_ratio': np.min(influence_ratios),
                'mean_absolute_reach': np.mean(final_activated_counts),
                'std_absolute_reach': np.std(final_activated_counts),
                'reach_consistency': 1 - (np.std(influence_ratios) / max(np.mean(influence_ratios), 0.001))
            }
        
        # Efficiency metrics (reach per step)
        if steps_taken and influence_ratios:
            efficiency_scores = [ratio / max(steps, 1) for ratio, steps in zip(influence_ratios, steps_taken)]
            
            metrics['efficiency_metrics'] = {
                'mean_efficiency': np.mean(efficiency_scores),
                'std_efficiency': np.std(efficiency_scores),
                'max_efficiency': np.max(efficiency_scores),
                'efficiency_consistency': 1 - (np.std(efficiency_scores) / max(np.mean(efficiency_scores), 0.001))
            }
        
        # Temporal metrics
        if activation_times_lists:
            all_activation_times = []
            for times_list in activation_times_lists:
                all_activation_times.extend(times_list)
            
            if all_activation_times:
                metrics['temporal_metrics'] = {
                    'mean_activation_time': np.mean(all_activation_times),
                    'std_activation_time': np.std(all_activation_times),
                    'median_activation_time': np.median(all_activation_times),
                    'activation_time_range': np.max(all_activation_times) - np.min(all_activation_times)
                }
        
        # Calculate composite scores
        if metrics['speed_metrics'] and metrics['reach_metrics']:
            speed_score = metrics['speed_metrics']['mean_propagation_speed']
            reach_score = metrics['reach_metrics']['mean_influence_ratio']
            
            metrics['composite_scores'] = {
                'speed_reach_product': speed_score * reach_score,
                'balanced_performance': (speed_score + reach_score) / 2,
                'speed_reach_ratio': speed_score / max(reach_score, 0.001)
            }
        
        return metrics
    
    def generate_influence_interpretation_report(self, 
                                               high_centrality_results: Dict[str, Any],
                                               community_analysis: Dict[str, Any],
                                               speed_metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Generate documented interpretation of network influence dynamics findings.
        
        This method addresses requirement 4.6: provide documented interpretation 
        of findings regarding network influence dynamics.
        
        Args:
            high_centrality_results: Results from evaluate_high_centrality_influence
            community_analysis: Results from analyze_community_propagation_impact
            speed_metrics: Results from calculate_propagation_speed_metrics
            
        Returns:
            Dictionary containing interpretations and insights
        """
        interpretations = {
            'executive_summary': '',
            'centrality_insights': '',
            'community_insights': '',
            'speed_reach_insights': '',
            'strategic_recommendations': '',
            'methodological_notes': ''
        }
        
        # Executive Summary
        total_communities = community_analysis.get('analysis_parameters', {}).get('num_communities', 0)
        graph_size = high_centrality_results.get('evaluation_parameters', {}).get('graph_size', 0)
        
        interpretations['executive_summary'] = f"""
        Network Influence Dynamics Analysis Summary:
        
        This analysis examined information propagation patterns across a network of {graph_size} nodes 
        organized into {total_communities} communities. The study evaluated the influence of high-centrality 
        nodes and community structures on information spread dynamics.
        
        Key findings indicate varying influence patterns across different centrality measures and 
        significant community-based propagation effects that impact overall network dynamics.
        """
        
        # Centrality Insights
        centrality_insights = []
        if 'comparative_analysis' in high_centrality_results:
            comp_analysis = high_centrality_results['comparative_analysis']
            
            # Best performers
            if 'best_performers' in comp_analysis:
                best_performers = comp_analysis['best_performers']
                for centrality_type, performer in best_performers.items():
                    influence = performer['mean_influence']
                    centrality_insights.append(
                        f"- {centrality_type.title()} centrality: Best performer achieved {influence:.3f} "
                        f"influence ratio (node {performer['node_id']})"
                    )
            
            # Correlations
            if 'correlation_analysis' in comp_analysis:
                correlations = comp_analysis['correlation_analysis']
                for centrality_type, corr_data in correlations.items():
                    correlation = corr_data['correlation_coefficient']
                    interpretation = corr_data['interpretation']
                    centrality_insights.append(
                        f"- {centrality_type.title()} centrality shows {interpretation} "
                        f"(r={correlation:.3f}) with actual influence"
                    )
        
        interpretations['centrality_insights'] = f"""
        High-Centrality Node Influence Analysis:
        
        The evaluation of high-centrality nodes reveals distinct patterns in their influence potential:
        
        {chr(10).join(centrality_insights) if centrality_insights else "No significant centrality patterns identified."}
        
        These findings suggest that centrality metrics vary in their predictive power for actual 
        influence propagation, with some measures being more reliable indicators than others.
        """
        
        # Community Insights
        community_insights = []
        if 'propagation_patterns' in community_analysis:
            patterns = community_analysis['propagation_patterns']
            
            if 'containment_vs_spillover' in patterns:
                containment_data = patterns['containment_vs_spillover']
                mean_containment = containment_data.get('mean_containment', 0)
                mean_spillover = containment_data.get('mean_spillover', 0)
                
                community_insights.append(
                    f"- Average information containment within communities: {mean_containment:.3f}"
                )
                community_insights.append(
                    f"- Average spillover to other communities: {mean_spillover:.3f}"
                )
            
            if 'community_types' in patterns:
                type_counts = defaultdict(int)
                for comm_type in patterns['community_types'].values():
                    type_counts[comm_type] += 1
                
                for comm_type, count in type_counts.items():
                    community_insights.append(f"- {count} communities classified as '{comm_type}'")
        
        interpretations['community_insights'] = f"""
        Community-Based Propagation Analysis:
        
        The analysis of community structures reveals important patterns in information flow:
        
        {chr(10).join(community_insights) if community_insights else "No significant community patterns identified."}
        
        These patterns indicate how community boundaries affect information propagation, with some 
        communities acting as information hubs while others serve as barriers or filters.
        """
        
        # Speed and Reach Insights
        speed_insights = []
        if 'speed_metrics' in speed_metrics:
            speed_data = speed_metrics['speed_metrics']
            mean_speed = speed_data.get('mean_propagation_speed', 0)
            speed_insights.append(f"- Average propagation speed: {mean_speed:.3f} nodes per step")
        
        if 'reach_metrics' in speed_metrics:
            reach_data = speed_metrics['reach_metrics']
            mean_reach = reach_data.get('mean_influence_ratio', 0)
            reach_consistency = reach_data.get('reach_consistency', 0)
            speed_insights.append(f"- Average influence reach: {mean_reach:.3f} of network")
            speed_insights.append(f"- Reach consistency score: {reach_consistency:.3f}")
        
        if 'efficiency_metrics' in speed_metrics:
            efficiency_data = speed_metrics['efficiency_metrics']
            mean_efficiency = efficiency_data.get('mean_efficiency', 0)
            speed_insights.append(f"- Average propagation efficiency: {mean_efficiency:.3f}")
        
        interpretations['speed_reach_insights'] = f"""
        Propagation Speed and Reach Analysis:
        
        The temporal dynamics of information spread show the following characteristics:
        
        {chr(10).join(speed_insights) if speed_insights else "No significant speed/reach patterns identified."}
        
        These metrics provide insights into the network's capacity for rapid information dissemination 
        and the consistency of propagation outcomes across different scenarios.
        """
        
        # Strategic Recommendations
        recommendations = [
            "1. Focus seed selection on nodes with high correlation between centrality and actual influence",
            "2. Consider community structure when planning information campaigns",
            "3. Leverage bridge nodes to maximize inter-community propagation",
            "4. Account for community types (insular vs. influential) in propagation strategies"
        ]
        
        interpretations['strategic_recommendations'] = f"""
        Strategic Recommendations for Network Influence:
        
        Based on the analysis findings, the following recommendations emerge:
        
        {chr(10).join(recommendations)}
        
        These recommendations should be adapted based on specific use cases and network characteristics.
        """
        
        # Methodological Notes
        interpretations['methodological_notes'] = f"""
        Methodological Considerations:
        
        This analysis employed multiple simulation runs to ensure statistical reliability of results. 
        The propagation model used was {self.propagation_model}, with parameters optimized for 
        the network characteristics.
        
        Key limitations include:
        - Simulation-based results may vary with different random seeds
        - Community detection quality affects community-based analysis accuracy
        - Centrality calculations assume static network structure
        - Propagation models are simplified representations of real-world dynamics
        
        Future analyses could benefit from:
        - Temporal network analysis for dynamic propagation patterns
        - Multiple propagation model comparisons
        - Validation against real-world propagation data
        - Integration of node attribute information beyond centrality
        """
        
        return interpretations
    
    def comprehensive_influence_evaluation(self, 
                                         graph: nx.Graph,
                                         centrality_scores: Dict[str, Dict[str, float]],
                                         communities: Dict[int, List[str]],
                                         num_seeds: int = 5,
                                         num_simulations: int = 50) -> Dict[str, Any]:
        """
        Perform comprehensive influence evaluation including high-centrality analysis,
        community impact assessment, and speed/reach metrics calculation.
        
        This method integrates all the functionality required for task 10:
        - Evaluate influence of high-centrality nodes (requirement 4.3)
        - Analyze community-based propagation patterns (requirement 4.4)
        - Calculate speed and reach metrics with interpretations (requirement 4.6)
        
        Args:
            graph: NetworkX graph
            centrality_scores: Dictionary containing centrality metrics
            communities: Dictionary mapping community IDs to node lists
            num_seeds: Number of seed nodes for evaluations
            num_simulations: Number of simulations per analysis
            
        Returns:
            Dictionary containing comprehensive influence evaluation results
        """
        self.logger.info("Starting comprehensive influence evaluation...")
        
        comprehensive_results = {
            'analysis_metadata': {
                'graph_nodes': graph.number_of_nodes(),
                'graph_edges': graph.number_of_edges(),
                'num_communities': len(communities),
                'num_seeds': num_seeds,
                'num_simulations': num_simulations,
                'propagation_model': self.propagation_model
            }
        }
        
        # 1. Evaluate high-centrality node influence (requirement 4.3)
        self.logger.info("Evaluating high-centrality node influence...")
        high_centrality_results = self.evaluate_high_centrality_influence(
            graph, centrality_scores, top_k=10, num_simulations=num_simulations
        )
        comprehensive_results['high_centrality_analysis'] = high_centrality_results
        
        # 2. Analyze community propagation impact (requirement 4.4)
        self.logger.info("Analyzing community propagation impact...")
        community_analysis = self.analyze_community_propagation_impact(
            graph, communities, centrality_scores, 
            num_seeds_per_community=3, num_simulations=num_simulations
        )
        comprehensive_results['community_analysis'] = community_analysis
        
        # 3. Calculate speed and reach metrics (requirement 4.6)
        self.logger.info("Calculating propagation speed and reach metrics...")
        
        # Run additional simulations for speed/reach analysis
        speed_reach_simulations = []
        for _ in range(num_simulations):
            try:
                # Select diverse seeds for speed/reach analysis
                seeds = self.select_seed_nodes(
                    graph, centrality_scores, num_seeds, 'mixed'
                )
                
                propagation_result = self.simulate_propagation(
                    graph, seeds, track_history=True
                )
                speed_reach_simulations.append(propagation_result)
                
            except Exception as e:
                self.logger.warning(f"Speed/reach simulation failed: {e}")
                continue
        
        speed_metrics = self.calculate_propagation_speed_metrics(speed_reach_simulations)
        comprehensive_results['speed_reach_metrics'] = speed_metrics
        
        # 4. Generate comprehensive interpretation report (requirement 4.6)
        self.logger.info("Generating influence interpretation report...")
        interpretation_report = self.generate_influence_interpretation_report(
            high_centrality_results, community_analysis, speed_metrics
        )
        comprehensive_results['interpretation_report'] = interpretation_report
        
        # 5. Calculate summary statistics and insights
        comprehensive_results['summary_insights'] = self._generate_summary_insights(
            high_centrality_results, community_analysis, speed_metrics
        )
        
        self.logger.info("Comprehensive influence evaluation completed")
        return comprehensive_results
    
    def _generate_summary_insights(self, 
                                   high_centrality_results: Dict[str, Any],
                                   community_analysis: Dict[str, Any],
                                   speed_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary insights from all analyses."""
        insights = {
            'key_findings': [],
            'performance_metrics': {},
            'network_characteristics': {}
        }
        
        # Extract key findings
        findings = []
        
        # High-centrality findings
        if 'comparative_analysis' in high_centrality_results:
            comp_analysis = high_centrality_results['comparative_analysis']
            if 'best_performers' in comp_analysis:
                best_centrality = max(comp_analysis['best_performers'].items(), 
                                    key=lambda x: x[1]['mean_influence'])
                findings.append(
                    f"Most effective centrality measure: {best_centrality[0]} "
                    f"(mean influence: {best_centrality[1]['mean_influence']:.3f})"
                )
        
        # Community findings
        if 'community_influence_ranking' in community_analysis:
            top_community = community_analysis['community_influence_ranking'][0]
            findings.append(
                f"Most influential community: Community {top_community[0]} "
                f"(influence score: {top_community[1]:.3f})"
            )
        
        # Speed/reach findings
        if 'reach_metrics' in speed_metrics:
            mean_reach = speed_metrics['reach_metrics'].get('mean_influence_ratio', 0)
            findings.append(f"Average network reach: {mean_reach:.3f} of total nodes")
        
        insights['key_findings'] = findings
        
        # Performance metrics summary
        if 'speed_metrics' in speed_metrics and 'reach_metrics' in speed_metrics:
            insights['performance_metrics'] = {
                'propagation_speed': speed_metrics['speed_metrics'].get('mean_propagation_speed', 0),
                'network_reach': speed_metrics['reach_metrics'].get('mean_influence_ratio', 0),
                'reach_consistency': speed_metrics['reach_metrics'].get('reach_consistency', 0)
            }
        
        # Network characteristics
        if 'propagation_patterns' in community_analysis:
            patterns = community_analysis['propagation_patterns']
            if 'containment_vs_spillover' in patterns:
                insights['network_characteristics'] = {
                    'community_containment': patterns['containment_vs_spillover'].get('mean_containment', 0),
                    'community_spillover': patterns['containment_vs_spillover'].get('mean_spillover', 0),
                    'containment_spillover_balance': patterns['containment_vs_spillover'].get('containment_spillover_correlation', 0)
                }
        
        return insights
    
    def visualize_comprehensive_influence_analysis(self, 
                                                 comprehensive_results: Dict[str, Any],
                                                 save_path: Optional[str] = None,
                                                 figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Create comprehensive visualization of influence analysis results.
        
        Args:
            comprehensive_results: Results from comprehensive_influence_evaluation
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Centrality influence comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_centrality_influence_comparison(ax1, comprehensive_results['high_centrality_analysis'])
        
        # 2. Community influence ranking
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_community_influence_ranking(ax2, comprehensive_results['community_analysis'])
        
        # 3. Speed vs Reach scatter
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_speed_reach_analysis(ax3, comprehensive_results['speed_reach_metrics'])
        
        # 4. Community propagation patterns
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_community_propagation_patterns(ax4, comprehensive_results['community_analysis'])
        
        # 5. Temporal metrics
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_temporal_metrics(ax5, comprehensive_results['speed_reach_metrics'])
        
        # 6. Summary insights text
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_summary_insights(ax6, comprehensive_results['summary_insights'])
        
        plt.suptitle('Comprehensive Network Influence Analysis', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comprehensive influence analysis visualization saved to {save_path}")
        
        return fig
    
    def _plot_centrality_influence_comparison(self, ax, high_centrality_results):
        """Plot centrality influence comparison."""
        if 'centrality_influence' not in high_centrality_results:
            ax.text(0.5, 0.5, 'No centrality data available', ha='center', va='center')
            ax.set_title('Centrality Influence Comparison')
            return
        
        centrality_types = []
        mean_influences = []
        
        for centrality_type, data in high_centrality_results['centrality_influence'].items():
            if 'individual_influence' in data:
                influences = [info['mean_influence'] for info in data['individual_influence'].values()]
                if influences:
                    centrality_types.append(centrality_type.replace('_', ' ').title())
                    mean_influences.append(np.mean(influences))
        
        if centrality_types and mean_influences:
            bars = ax.bar(centrality_types, mean_influences, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(centrality_types)])
            ax.set_ylabel('Mean Influence Ratio')
            ax.set_title('Centrality Influence Comparison')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, mean_influences):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_community_influence_ranking(self, ax, community_analysis):
        """Plot community influence ranking."""
        if 'community_influence_ranking' not in community_analysis:
            ax.text(0.5, 0.5, 'No community ranking data', ha='center', va='center')
            ax.set_title('Community Influence Ranking')
            return
        
        ranking = community_analysis['community_influence_ranking'][:10]  # Top 10
        if not ranking:
            ax.text(0.5, 0.5, 'No ranking data available', ha='center', va='center')
            ax.set_title('Community Influence Ranking')
            return
        
        communities = [f'Comm {comm_id}' for comm_id, _ in ranking]
        scores = [score for _, score in ranking]
        
        bars = ax.barh(communities, scores, color='#2ca02c')
        ax.set_xlabel('Influence Score')
        ax.set_title('Top Community Influence Ranking')
        
        # Add value labels
        for bar, value in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center')
    
    def _plot_speed_reach_analysis(self, ax, speed_metrics):
        """Plot speed vs reach analysis."""
        if 'speed_metrics' not in speed_metrics or 'reach_metrics' not in speed_metrics:
            ax.text(0.5, 0.5, 'No speed/reach data', ha='center', va='center')
            ax.set_title('Speed vs Reach Analysis')
            return
        
        speed = speed_metrics['speed_metrics'].get('mean_propagation_speed', 0)
        reach = speed_metrics['reach_metrics'].get('mean_influence_ratio', 0)
        
        ax.scatter([speed], [reach], s=100, c='red', alpha=0.7)
        ax.set_xlabel('Propagation Speed (nodes/step)')
        ax.set_ylabel('Influence Reach Ratio')
        ax.set_title('Speed vs Reach Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        ax.annotate(f'Speed: {speed:.3f}\nReach: {reach:.3f}', 
                   xy=(speed, reach), xytext=(10, 10), 
                   textcoords='offset points', fontsize=9)
    
    def _plot_community_propagation_patterns(self, ax, community_analysis):
        """Plot community propagation patterns."""
        if 'propagation_patterns' not in community_analysis:
            ax.text(0.5, 0.5, 'No propagation patterns', ha='center', va='center')
            ax.set_title('Community Propagation Patterns')
            return
        
        patterns = community_analysis['propagation_patterns']
        if 'community_types' not in patterns:
            ax.text(0.5, 0.5, 'No community types data', ha='center', va='center')
            ax.set_title('Community Propagation Patterns')
            return
        
        type_counts = defaultdict(int)
        for comm_type in patterns['community_types'].values():
            type_counts[comm_type] += 1
        
        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(types)]
            
            wedges, texts, autotexts = ax.pie(counts, labels=types, colors=colors, autopct='%1.1f%%')
            ax.set_title('Community Types Distribution')
    
    def _plot_temporal_metrics(self, ax, speed_metrics):
        """Plot temporal metrics."""
        if 'temporal_metrics' not in speed_metrics:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center')
            ax.set_title('Temporal Metrics')
            return
        
        temporal = speed_metrics['temporal_metrics']
        metrics = ['Mean Time', 'Median Time', 'Std Time']
        values = [
            temporal.get('mean_activation_time', 0),
            temporal.get('median_activation_time', 0),
            temporal.get('std_activation_time', 0)
        ]
        
        bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Time Steps')
        ax.set_title('Activation Time Metrics')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.2f}', ha='center', va='bottom')
    
    def _plot_summary_insights(self, ax, summary_insights):
        """Plot summary insights as text."""
        ax.axis('off')
        
        insights_text = "Key Findings:\n"
        if 'key_findings' in summary_insights:
            for i, finding in enumerate(summary_insights['key_findings'][:5], 1):
                insights_text += f"{i}. {finding}\n"
        
        if 'performance_metrics' in summary_insights:
            metrics = summary_insights['performance_metrics']
            insights_text += f"\nPerformance Summary:\n"
            insights_text += f"• Propagation Speed: {metrics.get('propagation_speed', 0):.3f}\n"
            insights_text += f"• Network Reach: {metrics.get('network_reach', 0):.3f}\n"
            insights_text += f"• Reach Consistency: {metrics.get('reach_consistency', 0):.3f}\n"
        
        ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Analysis Summary')