"""
Network Topology Analysis Module

This module provides comprehensive network topology analysis capabilities,
comparing different network models and their impact on information diffusion.
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx


class TopologyAnalyzer:
    """
    Analyzes network topologies and their impact on information diffusion.
    
    This class provides methods to compare small-world, scale-free, and random network models
    against the actual network structure and analyze their diffusion properties.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the TopologyAnalyzer.
        
        Args:
            logger: Optional logger instance for logging operations
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_network_properties(self, graph: nx.Graph) -> Dict:
        """
        Analyze basic properties of the current network.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            Dictionary containing network properties
        """
        try:
            properties = {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'clustering_coefficient': nx.average_clustering(graph),
                'is_connected': nx.is_connected(graph)
            }
            
            # Calculate path-based metrics only for connected graphs
            if properties['is_connected']:
                properties['average_shortest_path'] = nx.average_shortest_path_length(graph)
                properties['diameter'] = nx.diameter(graph)
            else:
                # For disconnected graphs, analyze the largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                largest_subgraph = graph.subgraph(largest_cc)
                properties['average_shortest_path'] = nx.average_shortest_path_length(largest_subgraph)
                properties['diameter'] = nx.diameter(largest_subgraph)
                properties['largest_component_size'] = len(largest_cc)
                properties['num_components'] = nx.number_connected_components(graph)
            
            # Calculate degree assortativity
            properties['assortativity'] = nx.degree_assortativity_coefficient(graph)
            
            return properties
            
        except Exception as e:
            self.logger.error(f"Error analyzing network properties: {e}")
            return {}
    
    def generate_synthetic_networks(self, graph: nx.Graph) -> Dict[str, nx.Graph]:
        """
        Generate synthetic networks for comparison.
        
        Args:
            graph: Original graph to base synthetic networks on
            
        Returns:
            Dictionary of synthetic networks
        """
        try:
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            
            if n_nodes == 0:
                return {}
            
            # Calculate average degree for better network generation
            avg_degree = 2 * n_edges / n_nodes
            
            synthetic_networks = {}
            
            # Random network (Erdős-Rényi)
            try:
                random_graph = nx.gnm_random_graph(n_nodes, n_edges, seed=42)
                # Ensure connectivity by adding edges between components if needed
                if not nx.is_connected(random_graph) and n_nodes > 1:
                    components = list(nx.connected_components(random_graph))
                    while len(components) > 1:
                        comp1, comp2 = components[0], components[1]
                        node1, node2 = list(comp1)[0], list(comp2)[0]
                        random_graph.add_edge(node1, node2)
                        components = list(nx.connected_components(random_graph))
                
                synthetic_networks['random'] = random_graph
            except Exception as e:
                self.logger.warning(f"Failed to generate random network: {e}")
            
            # Small-world network (Watts-Strogatz)
            try:
                k = max(2, int(avg_degree))
                if k % 2 == 1:  # Ensure k is even
                    k += 1
                k = min(k, n_nodes - 1)  # Ensure k < n
                
                if k >= 2 and n_nodes > k:
                    small_world_graph = nx.watts_strogatz_graph(n_nodes, k, 0.1, seed=42)
                    synthetic_networks['small_world'] = small_world_graph
            except Exception as e:
                self.logger.warning(f"Failed to generate small-world network: {e}")
            
            # Scale-free network (Barabási-Albert)
            try:
                m = max(1, int(avg_degree / 2))
                m = min(m, n_nodes - 1)  # Ensure m < n
                
                if m >= 1 and n_nodes > m:
                    scale_free_graph = nx.barabasi_albert_graph(n_nodes, m, seed=42)
                    synthetic_networks['scale_free'] = scale_free_graph
            except Exception as e:
                self.logger.warning(f"Failed to generate scale-free network: {e}")
            
            return synthetic_networks
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic networks: {e}")
            return {}
    
    def simulate_information_diffusion(self, graph: nx.Graph, seed_nodes: List, 
                                     infection_probability: float = 0.3, 
                                     max_steps: int = 10) -> Tuple[int, int]:
        """
        Simulate information diffusion on a graph.
        
        Args:
            graph: NetworkX graph
            seed_nodes: Initial nodes with information
            infection_probability: Probability of information spread per edge
            max_steps: Maximum simulation steps
            
        Returns:
            Tuple of (final_infected_count, steps_taken)
        """
        try:
            infected = set(seed_nodes)
            newly_infected = set(seed_nodes)
            
            for step in range(max_steps):
                next_infected = set()
                
                for node in newly_infected:
                    if node not in graph.nodes():
                        continue
                        
                    neighbors = set(graph.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in infected and np.random.random() < infection_probability:
                            next_infected.add(neighbor)
                
                if not next_infected:
                    break
                
                newly_infected = next_infected
                infected.update(newly_infected)
            
            return len(infected), step + 1
            
        except Exception as e:
            self.logger.warning(f"Diffusion simulation failed: {e}")
            return len(seed_nodes), 0
    
    def analyze_diffusion_patterns(self, networks: Dict[str, nx.Graph], 
                                 num_simulations: int = 10) -> Dict:
        """
        Analyze information diffusion patterns across different network topologies.
        
        Args:
            networks: Dictionary of networks to analyze
            num_simulations: Number of simulation runs per network
            
        Returns:
            Dictionary with diffusion analysis results
        """
        diffusion_results = {}
        
        for name, graph in networks.items():
            try:
                if graph.number_of_nodes() == 0:
                    continue
                
                # Select seed nodes (top 5% by degree or minimum 5 nodes)
                degrees = dict(graph.degree())
                sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
                num_seeds = max(5, int(0.05 * graph.number_of_nodes()))
                seed_nodes = sorted_nodes[:num_seeds]
                
                # Run multiple simulations
                simulation_results = []
                for _ in range(num_simulations):
                    final_infected, steps = self.simulate_information_diffusion(graph, seed_nodes)
                    simulation_results.append({
                        'final_infected': final_infected,
                        'steps_taken': steps,
                        'infection_rate': final_infected / graph.number_of_nodes()
                    })
                
                # Calculate statistics
                infection_rates = [r['infection_rate'] for r in simulation_results]
                steps_taken = [r['steps_taken'] for r in simulation_results]
                
                diffusion_results[name] = {
                    'avg_infection_rate': np.mean(infection_rates),
                    'std_infection_rate': np.std(infection_rates),
                    'avg_steps': np.mean(steps_taken),
                    'std_steps': np.std(steps_taken),
                    'max_infection_rate': np.max(infection_rates),
                    'min_infection_rate': np.min(infection_rates),
                    'simulation_count': num_simulations
                }
                
            except Exception as e:
                self.logger.warning(f"Diffusion analysis failed for {name}: {e}")
                diffusion_results[name] = {'error': str(e)}
        
        return diffusion_results
    
    def analyze_network_topologies(self, graph: nx.Graph, output_dir: str = 'outputs/topology_analysis') -> Optional[Dict]:
        """
        Comprehensive network topology analysis and comparison.
        
        Args:
            graph: NetworkX graph to analyze
            output_dir: Directory to save results
            
        Returns:
            Dictionary with topology analysis results or None if failed
        """
        self.logger.info("Analyzing network topologies and their impact...")
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Analyze current network properties
            current_properties = self.analyze_network_properties(graph)
            
            # Generate synthetic networks for comparison
            synthetic_networks = self.generate_synthetic_networks(graph)
            
            # Analyze properties of synthetic networks
            topology_analysis = {
                'current_network': current_properties
            }
            
            for name, synthetic_graph in synthetic_networks.items():
                topology_analysis[f'{name}_network'] = self.analyze_network_properties(synthetic_graph)
            
            # Analyze information diffusion patterns
            all_networks = {'current': graph}
            all_networks.update(synthetic_networks)
            
            diffusion_results = self.analyze_diffusion_patterns(all_networks)
            topology_analysis['diffusion_results'] = diffusion_results
            
            # Generate comparative analysis
            comparative_analysis = self._generate_comparative_analysis(topology_analysis)
            topology_analysis['comparative_analysis'] = comparative_analysis
            
            # Save topology analysis results
            with open(f'{output_dir}/topology_analysis.json', 'w') as f:
                json.dump(topology_analysis, f, indent=2, default=str)
            
            # Generate visualization if matplotlib is available
            self._generate_topology_visualization(topology_analysis, output_dir)
            
            self.logger.info("Network topology analysis completed")
            return topology_analysis
            
        except Exception as e:
            self.logger.error(f"Topology analysis failed: {e}")
            return None
    
    def _generate_comparative_analysis(self, topology_analysis: Dict) -> Dict:
        """Generate comparative analysis between different network topologies."""
        try:
            comparative = {
                'clustering_comparison': {},
                'path_length_comparison': {},
                'diffusion_efficiency': {},
                'network_characteristics': {}
            }
            
            # Extract metrics for comparison
            networks = ['current', 'random', 'small_world', 'scale_free']
            
            for network in networks:
                network_key = f'{network}_network' if network != 'current' else 'current_network'
                
                if network_key in topology_analysis:
                    props = topology_analysis[network_key]
                    
                    comparative['clustering_comparison'][network] = props.get('clustering_coefficient', 0)
                    comparative['path_length_comparison'][network] = props.get('average_shortest_path', float('inf'))
                    
                    # Diffusion efficiency from diffusion results
                    if 'diffusion_results' in topology_analysis and network in topology_analysis['diffusion_results']:
                        diffusion_data = topology_analysis['diffusion_results'][network]
                        if 'avg_infection_rate' in diffusion_data:
                            comparative['diffusion_efficiency'][network] = diffusion_data['avg_infection_rate']
            
            # Identify network characteristics
            current_clustering = comparative['clustering_comparison'].get('current', 0)
            current_path_length = comparative['path_length_comparison'].get('current', float('inf'))
            
            # Classify current network
            if current_clustering > 0.3 and current_path_length < 10:
                network_type = 'small_world_like'
            elif current_clustering < 0.1:
                network_type = 'random_like'
            else:
                network_type = 'intermediate'
            
            comparative['network_characteristics'] = {
                'primary_type': network_type,
                'clustering_level': 'high' if current_clustering > 0.3 else 'medium' if current_clustering > 0.1 else 'low',
                'path_efficiency': 'high' if current_path_length < 5 else 'medium' if current_path_length < 10 else 'low'
            }
            
            return comparative
            
        except Exception as e:
            self.logger.warning(f"Error generating comparative analysis: {e}")
            return {}
    
    def _generate_topology_visualization(self, topology_analysis: Dict, output_dir: str):
        """Generate comprehensive topology comparison visualization."""
        try:
            import matplotlib.pyplot as plt
            
            # Create comprehensive comparison figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Network Topology Analysis and Comparison', fontsize=16, fontweight='bold')
            
            networks = ['current', 'random', 'small_world', 'scale_free']
            colors = ['blue', 'red', 'green', 'orange']
            
            # Extract data for plotting
            clustering_values = []
            path_values = []
            diameter_values = []
            assortativity_values = []
            diffusion_rates = []
            
            valid_networks = []
            valid_colors = []
            
            for i, network in enumerate(networks):
                network_key = f'{network}_network' if network != 'current' else 'current_network'
                
                if network_key in topology_analysis:
                    props = topology_analysis[network_key]
                    
                    clustering_values.append(props.get('clustering_coefficient', 0))
                    path_values.append(props.get('average_shortest_path', 0) if props.get('average_shortest_path', float('inf')) != float('inf') else 0)
                    diameter_values.append(props.get('diameter', 0) if props.get('diameter', float('inf')) != float('inf') else 0)
                    assortativity_values.append(props.get('assortativity', 0))
                    
                    # Get diffusion rate
                    if 'diffusion_results' in topology_analysis and network in topology_analysis['diffusion_results']:
                        diffusion_data = topology_analysis['diffusion_results'][network]
                        diffusion_rates.append(diffusion_data.get('avg_infection_rate', 0))
                    else:
                        diffusion_rates.append(0)
                    
                    valid_networks.append(network.replace('_', ' ').title())
                    valid_colors.append(colors[i])
            
            if not valid_networks:
                self.logger.warning("No valid network data for visualization")
                return
            
            # Plot 1: Clustering Coefficients
            if clustering_values:
                bars1 = axes[0, 0].bar(valid_networks, clustering_values, color=valid_colors, alpha=0.7)
                axes[0, 0].set_title('Clustering Coefficients')
                axes[0, 0].set_ylabel('Clustering Coefficient')
                axes[0, 0].set_ylim(0, max(clustering_values) * 1.1 if max(clustering_values) > 0 else 1)
                axes[0, 0].grid(True, alpha=0.3, axis='y')
                
                for bar, v in zip(bars1, clustering_values):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                                   f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Average Shortest Path Lengths
            if path_values:
                bars2 = axes[0, 1].bar(valid_networks, path_values, color=valid_colors, alpha=0.7)
                axes[0, 1].set_title('Average Shortest Path Length')
                axes[0, 1].set_ylabel('Path Length')
                axes[0, 1].grid(True, alpha=0.3, axis='y')
                
                for bar, v in zip(bars2, path_values):
                    if v > 0:
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                       f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Network Diameter
            if diameter_values:
                bars3 = axes[0, 2].bar(valid_networks, diameter_values, color=valid_colors, alpha=0.7)
                axes[0, 2].set_title('Network Diameter')
                axes[0, 2].set_ylabel('Diameter')
                axes[0, 2].grid(True, alpha=0.3, axis='y')
                
                for bar, v in zip(bars3, diameter_values):
                    if v > 0:
                        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                       f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 4: Degree Assortativity
            if assortativity_values:
                bars4 = axes[1, 0].bar(valid_networks, assortativity_values, color=valid_colors, alpha=0.7)
                axes[1, 0].set_title('Degree Assortativity')
                axes[1, 0].set_ylabel('Assortativity Coefficient')
                axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                for bar, v in zip(bars4, assortativity_values):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                   f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 5: Information Diffusion Rates
            if diffusion_rates:
                bars5 = axes[1, 1].bar(valid_networks, diffusion_rates, color=valid_colors, alpha=0.7)
                axes[1, 1].set_title('Information Diffusion Rate')
                axes[1, 1].set_ylabel('Average Infection Rate')
                axes[1, 1].set_ylim(0, max(diffusion_rates) * 1.1 if max(diffusion_rates) > 0 else 1)
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                
                for bar, v in zip(bars5, diffusion_rates):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                   f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 6: Network Properties Summary
            axes[1, 2].axis('off')
            if 'current_network' in topology_analysis:
                current_props = topology_analysis['current_network']
                summary_text = f"""
Network Topology Summary:
• Nodes: {current_props.get('nodes', 'N/A')}
• Edges: {current_props.get('edges', 'N/A')}
• Density: {current_props.get('density', 0):.4f}
• Clustering: {current_props.get('clustering_coefficient', 0):.3f}
• Avg Path: {current_props.get('average_shortest_path', 0):.2f}
• Diameter: {current_props.get('diameter', 0):.1f}
• Assortativity: {current_props.get('assortativity', 0):.3f}

Topology Characteristics:
• Random: Low clustering, short paths
• Small-World: High clustering, short paths  
• Scale-Free: Low clustering, variable paths
• Current: Real-world network properties
                """
                axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                               fontsize=10, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/comprehensive_topology_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Comprehensive topology analysis visualization generated")
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping topology visualization")
        except Exception as e:
            self.logger.warning(f"Error generating topology visualization: {e}")