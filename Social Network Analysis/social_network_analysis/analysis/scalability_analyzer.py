"""
Scalability Analysis Module

This module provides scalability analysis capabilities for distributed graph processing,
evaluating performance across different graph sizes and processing approaches.
"""

import logging
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class ScalabilityAnalyzer:
    """
    Analyzes scalability of graph processing algorithms and distributed processing tools.
    
    This class provides methods to evaluate performance across different graph sizes,
    compare distributed vs. single-machine processing, and identify bottlenecks.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the ScalabilityAnalyzer.
        
        Args:
            logger: Optional logger instance for logging operations
        """
        self.logger = logger or logging.getLogger(__name__)
        self._dask_available = self._check_dask_availability()
    
    def _check_dask_availability(self) -> bool:
        """Check if Dask is available for distributed processing."""
        try:
            import dask.dataframe as dd
            import dask.array as da
            return True
        except ImportError:
            return False
    
    def measure_algorithm_performance(self, graph: nx.Graph, algorithm_name: str) -> Dict:
        """
        Measure performance of a specific graph algorithm.
        
        Args:
            graph: NetworkX graph
            algorithm_name: Name of the algorithm to test
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            start_time = time.time()
            memory_before = self._get_memory_usage()
            
            if algorithm_name == 'centrality':
                # Test centrality calculations
                degree_centrality = nx.degree_centrality(graph)
                # Use sampling for large graphs to avoid excessive computation time
                k = min(100, graph.number_of_nodes())
                betweenness_centrality = nx.betweenness_centrality(graph, k=k)
                result_size = len(degree_centrality) + len(betweenness_centrality)
                
            elif algorithm_name == 'community':
                # Test community detection
                communities = nx.community.louvain_communities(graph)
                result_size = len(communities)
                
            elif algorithm_name == 'shortest_path':
                # Test shortest path calculations
                if nx.is_connected(graph):
                    avg_path_length = nx.average_shortest_path_length(graph)
                    result_size = 1
                else:
                    # For disconnected graphs, analyze largest component
                    largest_cc = max(nx.connected_components(graph), key=len)
                    largest_subgraph = graph.subgraph(largest_cc)
                    avg_path_length = nx.average_shortest_path_length(largest_subgraph)
                    result_size = 1
                    
            elif algorithm_name == 'properties':
                # Test basic graph properties
                density = nx.density(graph)
                clustering = nx.average_clustering(graph)
                result_size = 2
                
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            end_time = time.time()
            memory_after = self._get_memory_usage()
            
            return {
                'execution_time': end_time - start_time,
                'memory_used': memory_after - memory_before,
                'result_size': result_size,
                'nodes_processed': graph.number_of_nodes(),
                'edges_processed': graph.number_of_edges()
            }
            
        except Exception as e:
            self.logger.warning(f"Performance measurement failed for {algorithm_name}: {e}")
            return {
                'execution_time': float('inf'),
                'memory_used': 0,
                'result_size': 0,
                'nodes_processed': graph.number_of_nodes(),
                'edges_processed': graph.number_of_edges(),
                'error': str(e)
            }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def test_graph_sizes(self, graph: nx.Graph, target_sizes: List[int] = None) -> Dict:
        """
        Test performance across different graph sizes.
        
        Args:
            graph: Original graph to sample from
            target_sizes: List of target graph sizes to test
            
        Returns:
            Dictionary with scalability results
        """
        if target_sizes is None:
            max_size = graph.number_of_nodes()
            target_sizes = [100, 500, 1000, 2000, max_size]
            target_sizes = [size for size in target_sizes if size <= max_size]
        
        scalability_results = {}
        algorithms = ['centrality', 'community', 'shortest_path', 'properties']
        
        for target_size in target_sizes:
            self.logger.info(f"Testing scalability for graph size: {target_size}")
            
            try:
                # Sample the graph
                if target_size < graph.number_of_nodes():
                    sampled_nodes = np.random.choice(
                        list(graph.nodes()), 
                        size=target_size, 
                        replace=False
                    )
                    sampled_graph = graph.subgraph(sampled_nodes).copy()
                else:
                    sampled_graph = graph
                
                # Test each algorithm
                size_results = {
                    'nodes': sampled_graph.number_of_nodes(),
                    'edges': sampled_graph.number_of_edges(),
                    'density': nx.density(sampled_graph),
                    'performance': {}
                }
                
                for algorithm in algorithms:
                    performance = self.measure_algorithm_performance(sampled_graph, algorithm)
                    size_results['performance'][algorithm] = performance
                
                scalability_results[target_size] = size_results
                
            except Exception as e:
                self.logger.warning(f"Scalability test failed for size {target_size}: {e}")
                scalability_results[target_size] = {'error': str(e)}
        
        return scalability_results
    
    def test_distributed_processing(self, graph: nx.Graph) -> Dict:
        """
        Test distributed processing capabilities using Dask.
        
        Args:
            graph: NetworkX graph to process
            
        Returns:
            Dictionary with distributed processing results
        """
        distributed_results = {
            'dask_available': self._dask_available,
            'tests_performed': []
        }
        
        if not self._dask_available:
            distributed_results['note'] = 'Dask not available for distributed processing'
            return distributed_results
        
        try:
            import dask.dataframe as dd
            import dask.array as da
            from dask import delayed
            
            # Convert graph to DataFrame for distributed processing
            edges_df = pd.DataFrame([
                {'source': u, 'target': v, 'weight': graph[u][v].get('weight', 1)}
                for u, v in graph.edges()
            ])
            
            nodes_df = pd.DataFrame([
                {'node': node, 'degree': graph.degree(node)}
                for node in graph.nodes()
            ])
            
            # Test 1: Distributed edge processing
            start_time = time.time()
            dask_edges = dd.from_pandas(edges_df, npartitions=4)
            edge_count = dask_edges.count().compute()
            edge_processing_time = time.time() - start_time
            
            distributed_results['tests_performed'].append({
                'test_name': 'edge_processing',
                'processing_time': edge_processing_time,
                'result': {'edge_count': len(edge_count)}
            })
            
            # Test 2: Distributed degree distribution calculation
            start_time = time.time()
            degree_distribution = dask_edges.groupby('source').size().compute()
            degree_processing_time = time.time() - start_time
            
            distributed_results['tests_performed'].append({
                'test_name': 'degree_distribution',
                'processing_time': degree_processing_time,
                'result': {
                    'mean_degree': float(degree_distribution.mean()),
                    'std_degree': float(degree_distribution.std()),
                    'max_degree': int(degree_distribution.max())
                }
            })
            
            # Test 3: Distributed node processing
            start_time = time.time()
            dask_nodes = dd.from_pandas(nodes_df, npartitions=4)
            node_stats = dask_nodes['degree'].describe().compute()
            node_processing_time = time.time() - start_time
            
            distributed_results['tests_performed'].append({
                'test_name': 'node_statistics',
                'processing_time': node_processing_time,
                'result': node_stats.to_dict()
            })
            
            # Calculate overall distributed processing efficiency
            total_distributed_time = sum(test['processing_time'] for test in distributed_results['tests_performed'])
            
            # Compare with sequential processing
            start_time = time.time()
            sequential_edge_count = len(edges_df)
            sequential_degree_dist = edges_df.groupby('source').size()
            sequential_node_stats = nodes_df['degree'].describe()
            sequential_time = time.time() - start_time
            
            distributed_results['performance_comparison'] = {
                'distributed_time': total_distributed_time,
                'sequential_time': sequential_time,
                'speedup_ratio': sequential_time / total_distributed_time if total_distributed_time > 0 else 0,
                'efficiency': 'better' if total_distributed_time < sequential_time else 'worse'
            }
            
            self.logger.info("Distributed processing tests completed successfully")
            
        except Exception as e:
            self.logger.warning(f"Distributed processing test failed: {e}")
            distributed_results['error'] = str(e)
        
        return distributed_results
    
    def analyze_scaling_patterns(self, scalability_results: Dict) -> Dict:
        """
        Analyze scaling patterns from scalability test results.
        
        Args:
            scalability_results: Results from test_graph_sizes
            
        Returns:
            Dictionary with scaling pattern analysis
        """
        try:
            scaling_analysis = {
                'algorithms': {},
                'overall_patterns': {},
                'bottlenecks': []
            }
            
            # Extract data for each algorithm
            algorithms = ['centrality', 'community', 'shortest_path', 'properties']
            
            for algorithm in algorithms:
                sizes = []
                times = []
                memory_usage = []
                
                for size, results in scalability_results.items():
                    if 'performance' in results and algorithm in results['performance']:
                        perf = results['performance'][algorithm]
                        if 'error' not in perf:
                            sizes.append(size)
                            times.append(perf['execution_time'])
                            memory_usage.append(perf['memory_used'])
                
                if len(sizes) >= 2:
                    # Calculate scaling coefficients
                    scaling_analysis['algorithms'][algorithm] = {
                        'sizes': sizes,
                        'execution_times': times,
                        'memory_usage': memory_usage,
                        'time_complexity': self._estimate_complexity(sizes, times),
                        'memory_complexity': self._estimate_complexity(sizes, memory_usage),
                        'scalability_rating': self._rate_scalability(sizes, times)
                    }
                    
                    # Identify bottlenecks
                    if max(times) > 60:  # More than 1 minute
                        scaling_analysis['bottlenecks'].append({
                            'algorithm': algorithm,
                            'issue': 'high_execution_time',
                            'max_time': max(times),
                            'problematic_size': sizes[times.index(max(times))]
                        })
            
            # Overall scaling patterns
            all_times = []
            all_sizes = []
            
            for size, results in scalability_results.items():
                if 'performance' in results:
                    total_time = sum(
                        perf['execution_time'] for perf in results['performance'].values()
                        if 'error' not in perf and perf['execution_time'] != float('inf')
                    )
                    if total_time > 0:
                        all_times.append(total_time)
                        all_sizes.append(size)
            
            if len(all_sizes) >= 2:
                scaling_analysis['overall_patterns'] = {
                    'overall_complexity': self._estimate_complexity(all_sizes, all_times),
                    'scalability_trend': 'linear' if self._is_linear_scaling(all_sizes, all_times) else 'non_linear',
                    'recommended_max_size': self._recommend_max_size(all_sizes, all_times)
                }
            
            return scaling_analysis
            
        except Exception as e:
            self.logger.warning(f"Error analyzing scaling patterns: {e}")
            return {}
    
    def _estimate_complexity(self, sizes: List[int], times: List[float]) -> str:
        """Estimate computational complexity from size vs time data."""
        try:
            if len(sizes) < 2:
                return 'insufficient_data'
            
            # Calculate ratios
            size_ratios = [sizes[i] / sizes[i-1] for i in range(1, len(sizes))]
            time_ratios = [times[i] / times[i-1] for i in range(1, len(times)) if times[i-1] > 0]
            
            if not time_ratios:
                return 'constant'
            
            avg_size_ratio = np.mean(size_ratios)
            avg_time_ratio = np.mean(time_ratios)
            
            if avg_time_ratio < avg_size_ratio * 0.5:
                return 'sub_linear'
            elif avg_time_ratio < avg_size_ratio * 1.5:
                return 'linear'
            elif avg_time_ratio < avg_size_ratio ** 2 * 1.5:
                return 'quadratic'
            else:
                return 'super_quadratic'
                
        except Exception:
            return 'unknown'
    
    def _is_linear_scaling(self, sizes: List[int], times: List[float]) -> bool:
        """Check if scaling is approximately linear."""
        try:
            if len(sizes) < 3:
                return True
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(sizes, times)[0, 1]
            return correlation > 0.8
            
        except Exception:
            return False
    
    def _rate_scalability(self, sizes: List[int], times: List[float]) -> str:
        """Rate the scalability of an algorithm."""
        try:
            if not times or max(times) == 0:
                return 'excellent'
            
            max_time = max(times)
            max_size = max(sizes)
            
            # Rate based on maximum execution time and size
            if max_time < 1 and max_size > 1000:
                return 'excellent'
            elif max_time < 10 and max_size > 500:
                return 'good'
            elif max_time < 60:
                return 'fair'
            else:
                return 'poor'
                
        except Exception:
            return 'unknown'
    
    def _recommend_max_size(self, sizes: List[int], times: List[float]) -> int:
        """Recommend maximum practical graph size based on performance."""
        try:
            # Find size where execution time becomes impractical (>60 seconds)
            for i, time in enumerate(times):
                if time > 60:
                    return sizes[max(0, i-1)]
            
            # If no bottleneck found, extrapolate
            if len(sizes) >= 2:
                return int(max(sizes) * 2)
            else:
                return max(sizes) if sizes else 1000
                
        except Exception:
            return 1000
    
    def perform_scalability_analysis(self, graph: nx.Graph, output_dir: str = 'outputs/scalability_analysis') -> Optional[Dict]:
        """
        Perform comprehensive scalability analysis.
        
        Args:
            graph: NetworkX graph to analyze
            output_dir: Directory to save results
            
        Returns:
            Dictionary with scalability analysis results or None if failed
        """
        self.logger.info("Performing scalability analysis with distributed processing...")
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Test scalability across different graph sizes
            scalability_results = self.test_graph_sizes(graph)
            
            # Test distributed processing capabilities
            distributed_results = self.test_distributed_processing(graph)
            
            # Analyze scaling patterns
            scaling_patterns = self.analyze_scaling_patterns(scalability_results)
            
            # Compile comprehensive results
            comprehensive_results = {
                'graph_info': {
                    'original_nodes': graph.number_of_nodes(),
                    'original_edges': graph.number_of_edges(),
                    'original_density': nx.density(graph)
                },
                'scalability_tests': scalability_results,
                'distributed_processing': distributed_results,
                'scaling_analysis': scaling_patterns,
                'recommendations': self._generate_recommendations(scaling_patterns, distributed_results)
            }
            
            # Save results
            with open(f'{output_dir}/scalability_results.json', 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            
            # Generate visualization
            self._generate_scalability_visualization(comprehensive_results, output_dir)
            
            self.logger.info("Scalability analysis completed")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Scalability analysis failed: {e}")
            return None
    
    def _generate_recommendations(self, scaling_patterns: Dict, distributed_results: Dict) -> Dict:
        """Generate recommendations based on scalability analysis."""
        recommendations = {
            'performance_optimization': [],
            'distributed_processing': [],
            'resource_management': []
        }
        
        try:
            # Performance optimization recommendations
            if 'algorithms' in scaling_patterns:
                for algorithm, data in scaling_patterns['algorithms'].items():
                    if data.get('scalability_rating') == 'poor':
                        recommendations['performance_optimization'].append(
                            f"Consider optimizing {algorithm} algorithm or using approximation methods for large graphs"
                        )
                    
                    if data.get('time_complexity') in ['quadratic', 'super_quadratic']:
                        recommendations['performance_optimization'].append(
                            f"{algorithm} shows {data['time_complexity']} scaling - consider parallel processing"
                        )
            
            # Distributed processing recommendations
            if distributed_results.get('dask_available'):
                if distributed_results.get('performance_comparison', {}).get('efficiency') == 'better':
                    recommendations['distributed_processing'].append(
                        "Distributed processing shows performance benefits - consider using for large graphs"
                    )
                else:
                    recommendations['distributed_processing'].append(
                        "Distributed processing overhead may not be beneficial for current graph size"
                    )
            else:
                recommendations['distributed_processing'].append(
                    "Consider installing Dask for distributed processing capabilities"
                )
            
            # Resource management recommendations
            if 'overall_patterns' in scaling_patterns:
                max_size = scaling_patterns['overall_patterns'].get('recommended_max_size', 1000)
                recommendations['resource_management'].append(
                    f"Recommended maximum graph size for current setup: {max_size} nodes"
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {e}")
            return recommendations
    
    def _generate_scalability_visualization(self, results: Dict, output_dir: str):
        """Generate scalability analysis visualization."""
        try:
            import matplotlib.pyplot as plt
            
            scalability_data = results.get('scalability_tests', {})
            if not scalability_data:
                return
            
            # Extract data for plotting
            sizes = list(scalability_data.keys())
            algorithms = ['centrality', 'community', 'shortest_path', 'properties']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Scalability Analysis Results', fontsize=16, fontweight='bold')
            
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, algorithm in enumerate(algorithms):
                ax = axes[i // 2, i % 2]
                
                algorithm_sizes = []
                algorithm_times = []
                
                for size in sizes:
                    if (size in scalability_data and 
                        'performance' in scalability_data[size] and 
                        algorithm in scalability_data[size]['performance']):
                        
                        perf = scalability_data[size]['performance'][algorithm]
                        if 'error' not in perf and perf['execution_time'] != float('inf'):
                            algorithm_sizes.append(size)
                            algorithm_times.append(perf['execution_time'])
                
                if algorithm_sizes:
                    ax.plot(algorithm_sizes, algorithm_times, 'o-', 
                           color=colors[i], label=algorithm.replace('_', ' ').title())
                    ax.set_xlabel('Graph Size (nodes)')
                    ax.set_ylabel('Execution Time (seconds)')
                    ax.set_title(f'{algorithm.replace("_", " ").title()} Scaling')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/scalability_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Scalability analysis visualization generated")
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping scalability visualization")
        except Exception as e:
            self.logger.warning(f"Error generating scalability visualization: {e}")