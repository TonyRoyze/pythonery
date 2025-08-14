"""
Community Detection Module

This module implements community detection algorithms for social network analysis,
including the Louvain algorithm and modularity calculations.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

try:
    import community as community_louvain
except ImportError:
    community_louvain = None


class CommunityDetector:
    """
    A class for detecting communities in social networks using various algorithms.
    
    This class implements the Louvain algorithm for community detection and provides
    modularity calculation capabilities for assessing community quality.
    """
    
    def __init__(self):
        """Initialize the CommunityDetector."""
        if community_louvain is None:
            raise ImportError(
                "python-louvain package is required for community detection. "
                "Install it with: pip install python-louvain"
            )
    
    def detect_louvain_communities(self, graph: nx.Graph, resolution: float = 1.0, 
                                 random_state: Optional[int] = None) -> Dict:
        """
        Detect communities using the Louvain algorithm.
        
        Args:
            graph: NetworkX graph to analyze
            resolution: Resolution parameter for community detection (default: 1.0)
            random_state: Random seed for reproducible results
            
        Returns:
            Dictionary containing:
            - 'communities': Dict mapping community_id to list of nodes
            - 'node_communities': Dict mapping node to community_id
            - 'modularity': Modularity score of the partition
            - 'num_communities': Number of detected communities
        """
        if len(graph.nodes()) == 0:
            return {
                'communities': {},
                'node_communities': {},
                'modularity': 0.0,
                'num_communities': 0
            }
        
        # Apply Louvain algorithm
        partition = community_louvain.best_partition(
            graph, 
            resolution=resolution,
            random_state=random_state
        )
        
        # Convert partition to communities format
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        
        # Convert to regular dict
        communities = dict(communities)
        
        # Calculate modularity
        modularity = community_louvain.modularity(partition, graph)
        
        return {
            'communities': communities,
            'node_communities': partition,
            'modularity': modularity,
            'num_communities': len(communities)
        }
    
    def calculate_modularity(self, graph: nx.Graph, communities: Dict[int, List]) -> float:
        """
        Calculate the modularity of a given community partition.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community_id to list of nodes
            
        Returns:
            Modularity score (float between -1 and 1)
        """
        if len(communities) == 0:
            return 0.0
        
        # Convert communities format to partition format
        partition = {}
        for community_id, nodes in communities.items():
            for node in nodes:
                partition[node] = community_id
        
        return community_louvain.modularity(partition, graph)
    
    def detect_spectral_communities(self, graph: nx.Graph, n_clusters: Optional[int] = None,
                                   random_state: Optional[int] = None) -> Dict:
        """
        Detect communities using Spectral Clustering algorithm.
        
        Args:
            graph: NetworkX graph to analyze
            n_clusters: Number of clusters to detect. If None, will be auto-determined
            random_state: Random seed for reproducible results
            
        Returns:
            Dictionary containing:
            - 'communities': Dict mapping community_id to list of nodes
            - 'node_communities': Dict mapping node to community_id
            - 'modularity': Modularity score of the partition
            - 'num_communities': Number of detected communities
            - 'silhouette_score': Silhouette score for cluster quality
        """
        if len(graph.nodes()) == 0:
            return {
                'communities': {},
                'node_communities': {},
                'modularity': 0.0,
                'num_communities': 0,
                'silhouette_score': 0.0
            }
        
        # Convert graph to adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(graph).toarray()
        
        # Auto-determine number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters_spectral(adjacency_matrix, random_state)
        
        # Ensure n_clusters doesn't exceed number of nodes
        n_clusters = min(n_clusters, len(graph.nodes()))
        
        # Apply Spectral Clustering
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=random_state,
            assign_labels='discretize'
        )
        
        cluster_labels = spectral.fit_predict(adjacency_matrix)
        
        # Convert to communities format
        communities = defaultdict(list)
        node_communities = {}
        nodes = list(graph.nodes())
        
        for i, node in enumerate(nodes):
            community_id = int(cluster_labels[i])
            communities[community_id].append(node)
            node_communities[node] = community_id
        
        # Convert to regular dict
        communities = dict(communities)
        
        # Calculate modularity
        modularity = self.calculate_modularity(graph, communities)
        
        # Calculate silhouette score for cluster quality assessment
        silhouette = silhouette_score(adjacency_matrix, cluster_labels) if len(set(cluster_labels)) > 1 else 0.0
        
        return {
            'communities': communities,
            'node_communities': node_communities,
            'modularity': modularity,
            'num_communities': len(communities),
            'silhouette_score': silhouette
        }
    
    def _find_optimal_clusters_spectral(self, adjacency_matrix: np.ndarray, 
                                      random_state: Optional[int] = None,
                                      max_clusters: Optional[int] = None) -> int:
        """
        Find optimal number of clusters for spectral clustering using silhouette analysis.
        
        Args:
            adjacency_matrix: Adjacency matrix of the graph
            random_state: Random seed for reproducible results
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        n_nodes = adjacency_matrix.shape[0]
        
        # Set reasonable bounds for cluster search
        min_clusters = 2
        if max_clusters is None:
            max_clusters = min(10, n_nodes // 2)  # Don't test more than half the nodes
        
        max_clusters = max(min_clusters, min(max_clusters, n_nodes - 1))
        
        if min_clusters >= max_clusters:
            return min_clusters
        
        best_score = -1
        best_n_clusters = min_clusters
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=random_state,
                    assign_labels='discretize'
                )
                
                cluster_labels = spectral.fit_predict(adjacency_matrix)
                
                # Calculate silhouette score
                if len(set(cluster_labels)) > 1:
                    score = silhouette_score(adjacency_matrix, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        
            except Exception:
                # Skip this number of clusters if it fails
                continue
        
        return best_n_clusters
    
    def tune_spectral_parameters(self, graph: nx.Graph, 
                               cluster_range: Tuple[int, int] = (2, 10),
                               random_state: Optional[int] = None) -> Dict:
        """
        Tune spectral clustering parameters to find optimal number of clusters.
        
        Args:
            graph: NetworkX graph to analyze
            cluster_range: Range of cluster numbers to test (min, max)
            random_state: Random seed for reproducible results
            
        Returns:
            Dictionary with tuning results including optimal parameters and scores
        """
        if len(graph.nodes()) == 0:
            return {
                'optimal_clusters': 2,
                'best_modularity': 0.0,
                'best_silhouette': 0.0,
                'tuning_results': []
            }
        
        adjacency_matrix = nx.adjacency_matrix(graph).toarray()
        min_clusters, max_clusters = cluster_range
        
        # Ensure valid range
        max_clusters = min(max_clusters, len(graph.nodes()) - 1)
        min_clusters = max(2, min_clusters)
        
        if min_clusters >= max_clusters:
            min_clusters = 2
            max_clusters = min(10, len(graph.nodes()) - 1)
        
        tuning_results = []
        best_modularity = -1
        best_silhouette = -1
        optimal_clusters = min_clusters
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                result = self.detect_spectral_communities(graph, n_clusters, random_state)
                
                tuning_results.append({
                    'n_clusters': n_clusters,
                    'modularity': result['modularity'],
                    'silhouette_score': result['silhouette_score'],
                    'num_communities': result['num_communities']
                })
                
                # Use modularity as primary criterion, silhouette as secondary
                if (result['modularity'] > best_modularity or 
                    (result['modularity'] == best_modularity and result['silhouette_score'] > best_silhouette)):
                    best_modularity = result['modularity']
                    best_silhouette = result['silhouette_score']
                    optimal_clusters = n_clusters
                    
            except Exception as e:
                # Log failed attempts but continue
                tuning_results.append({
                    'n_clusters': n_clusters,
                    'modularity': 0.0,
                    'silhouette_score': 0.0,
                    'num_communities': 0,
                    'error': str(e)
                })
        
        return {
            'optimal_clusters': optimal_clusters,
            'best_modularity': best_modularity,
            'best_silhouette': best_silhouette,
            'tuning_results': tuning_results
        }
    
    def compare_community_methods(self, graph: nx.Graph, 
                                spectral_clusters: Optional[int] = None,
                                random_state: Optional[int] = None) -> Dict:
        """
        Compare Louvain and Spectral clustering methods on the same graph.
        
        Args:
            graph: NetworkX graph to analyze
            spectral_clusters: Number of clusters for spectral method (auto-determined if None)
            random_state: Random seed for reproducible results
            
        Returns:
            Dictionary with comparison results between methods
        """
        if len(graph.nodes()) == 0:
            return {
                'louvain_results': {},
                'spectral_results': {},
                'comparison': {
                    'modularity_difference': 0.0,
                    'better_method': 'none',
                    'louvain_communities': 0,
                    'spectral_communities': 0
                }
            }
        
        # Run Louvain algorithm
        louvain_results = self.detect_louvain_communities(graph, random_state=random_state)
        
        # Run Spectral clustering
        spectral_results = self.detect_spectral_communities(
            graph, n_clusters=spectral_clusters, random_state=random_state
        )
        
        # Compare results
        modularity_diff = louvain_results['modularity'] - spectral_results['modularity']
        better_method = 'louvain' if modularity_diff > 0 else 'spectral' if modularity_diff < 0 else 'tie'
        
        comparison = {
            'modularity_difference': modularity_diff,
            'better_method': better_method,
            'louvain_communities': louvain_results['num_communities'],
            'spectral_communities': spectral_results['num_communities'],
            'louvain_modularity': louvain_results['modularity'],
            'spectral_modularity': spectral_results['modularity'],
            'spectral_silhouette': spectral_results['silhouette_score']
        }
        
        return {
            'louvain_results': louvain_results,
            'spectral_results': spectral_results,
            'comparison': comparison
        }
    
    def get_community_statistics(self, graph: nx.Graph, communities: Dict[int, List]) -> Dict:
        """
        Calculate statistics for each detected community.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community_id to list of nodes
            
        Returns:
            Dictionary with community statistics
        """
        stats = {}
        
        for community_id, nodes in communities.items():
            subgraph = graph.subgraph(nodes)
            
            # Basic statistics
            num_nodes = len(nodes)
            num_edges = subgraph.number_of_edges()
            
            # Density calculation
            max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 0
            density = num_edges / max_edges if max_edges > 0 else 0
            
            # Internal vs external edges
            internal_edges = num_edges
            external_edges = 0
            for node in nodes:
                for neighbor in graph.neighbors(node):
                    if neighbor not in nodes:
                        external_edges += 1
            
            stats[community_id] = {
                'size': num_nodes,
                'internal_edges': internal_edges,
                'external_edges': external_edges,
                'density': density,
                'nodes': nodes
            }
        
        return stats
    
    def visualize_communities(self, graph: nx.Graph, communities: Dict[int, List],
                            method: str = 'louvain', layout: str = 'spring', 
                            figsize: Tuple[int, int] = (12, 8),
                            node_size: int = 300, save_path: Optional[str] = None) -> None:
        """
        Visualize the network with communities highlighted using distinct colors.
        
        Args:
            graph: NetworkX graph to visualize
            communities: Dictionary mapping community_id to list of nodes
            method: Community detection method used ('louvain', 'spectral', or 'comparison')
            layout: Layout algorithm ('spring', 'circular', 'random', etc.)
            figsize: Figure size as (width, height)
            node_size: Size of nodes in the visualization
            save_path: Optional path to save the visualization
        """
        plt.figure(figsize=figsize)
        
        # Generate layout
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'random':
            pos = nx.random_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # Generate distinct colors for communities
        num_communities = len(communities)
        if num_communities > 0:
            # Use a colormap to generate distinct colors
            cmap = plt.cm.Set3 if num_communities <= 12 else plt.cm.tab20
            colors = [cmap(i / max(num_communities - 1, 1)) for i in range(num_communities)]
        else:
            colors = ['blue']
        
        # Create node color mapping
        node_colors = {}
        for i, (community_id, nodes) in enumerate(communities.items()):
            color = colors[i % len(colors)]
            for node in nodes:
                node_colors[node] = color
        
        # Handle nodes not in any community
        for node in graph.nodes():
            if node not in node_colors:
                node_colors[node] = 'gray'
        
        # Draw the network
        node_color_list = [node_colors[node] for node in graph.nodes()]
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_color_list, 
                              node_size=node_size, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, width=0.5)
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
        
        # Set title based on method
        method_name = method.title() if method != 'comparison' else 'Community Comparison'
        plt.title(f'Network Communities ({method_name} Algorithm)\n'
                 f'{num_communities} communities detected', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Add legend
        if num_communities > 0:
            legend_elements = []
            for i, (community_id, nodes) in enumerate(communities.items()):
                color = colors[i % len(colors)]
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10,
                                                label=f'Community {community_id} ({len(nodes)} nodes)'))
            
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Community visualization saved to: {save_path}")
        
        plt.show()
    
    def visualize_method_comparison(self, graph: nx.Graph, comparison_results: Dict,
                                  layout: str = 'spring', figsize: Tuple[int, int] = (15, 6),
                                  node_size: int = 300, save_path: Optional[str] = None) -> None:
        """
        Create side-by-side visualization comparing Louvain and Spectral clustering results.
        
        Args:
            graph: NetworkX graph to visualize
            comparison_results: Results from compare_community_methods()
            layout: Layout algorithm to use for both plots
            figsize: Figure size as (width, height)
            node_size: Size of nodes in the visualization
            save_path: Optional path to save the visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Generate consistent layout for both plots
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'random':
            pos = nx.random_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # Plot Louvain results
        plt.sca(ax1)
        louvain_communities = comparison_results['louvain_results']['communities']
        self._plot_communities_on_axis(graph, louvain_communities, pos, node_size, ax1)
        ax1.set_title(f'Louvain Algorithm\n'
                     f'{len(louvain_communities)} communities, '
                     f'Modularity: {comparison_results["louvain_results"]["modularity"]:.3f}',
                     fontsize=12, fontweight='bold')
        
        # Plot Spectral results
        plt.sca(ax2)
        spectral_communities = comparison_results['spectral_results']['communities']
        self._plot_communities_on_axis(graph, spectral_communities, pos, node_size, ax2)
        ax2.set_title(f'Spectral Clustering\n'
                     f'{len(spectral_communities)} communities, '
                     f'Modularity: {comparison_results["spectral_results"]["modularity"]:.3f}',
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Method comparison visualization saved to: {save_path}")
        
        plt.show()
    
    def _plot_communities_on_axis(self, graph: nx.Graph, communities: Dict[int, List],
                                pos: Dict, node_size: int, ax) -> None:
        """
        Helper method to plot communities on a specific axis.
        
        Args:
            graph: NetworkX graph to plot
            communities: Dictionary mapping community_id to list of nodes
            pos: Node positions dictionary
            node_size: Size of nodes
            ax: Matplotlib axis to plot on
        """
        # Generate colors
        num_communities = len(communities)
        if num_communities > 0:
            cmap = plt.cm.Set3 if num_communities <= 12 else plt.cm.tab20
            colors = [cmap(i / max(num_communities - 1, 1)) for i in range(num_communities)]
        else:
            colors = ['blue']
        
        # Create node color mapping
        node_colors = {}
        for i, (community_id, nodes) in enumerate(communities.items()):
            color = colors[i % len(colors)]
            for node in nodes:
                node_colors[node] = color
        
        # Handle nodes not in any community
        for node in graph.nodes():
            if node not in node_colors:
                node_colors[node] = 'gray'
        
        # Draw the network
        node_color_list = [node_colors[node] for node in graph.nodes()]
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_color_list, 
                              node_size=node_size, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, width=0.5, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.axis('off')
    
    def compare_with_ground_truth(self, detected_communities: Dict[int, List],
                                ground_truth_communities: Dict[int, List]) -> Dict:
        """
        Compare detected communities with ground truth communities.
        
        Args:
            detected_communities: Communities detected by algorithm
            ground_truth_communities: Known ground truth communities
            
        Returns:
            Dictionary with comparison metrics
        """
        # This is a placeholder for future implementation
        # Could include metrics like Normalized Mutual Information (NMI),
        # Adjusted Rand Index (ARI), etc.
        return {
            'detected_count': len(detected_communities),
            'ground_truth_count': len(ground_truth_communities),
            'comparison_available': False,
            'note': 'Ground truth comparison not yet implemented'
        }