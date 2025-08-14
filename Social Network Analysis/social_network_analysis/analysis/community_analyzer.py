"""
Community Analysis and Interpretation Module

This module provides comprehensive analysis and interpretation of detected communities
in social networks, including community characteristics, key member identification,
and documentation generation.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


class CommunityAnalyzer:
    """
    A class for analyzing and interpreting detected communities in social networks.
    
    This class provides methods to analyze community characteristics such as size,
    density, key members, and generates comprehensive documentation of findings.
    """
    
    def __init__(self):
        """Initialize the CommunityAnalyzer."""
        self.analysis_results = {}
        self.documentation_cache = {}
    
    def analyze_community_characteristics(self, graph: nx.Graph, 
                                        communities: Dict[int, List],
                                        centrality_scores: Optional[Dict[str, Dict]] = None) -> Dict:
        """
        Analyze comprehensive characteristics of detected communities.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community_id to list of nodes
            centrality_scores: Optional dictionary with centrality metrics
            
        Returns:
            Dictionary with detailed community characteristics analysis
        """
        if not communities:
            return {
                'summary': {
                    'total_communities': 0,
                    'total_nodes': len(graph.nodes()),
                    'coverage': 0.0
                },
                'communities': {}
            }
        
        community_analysis = {}
        total_nodes_in_communities = 0
        
        for community_id, nodes in communities.items():
            analysis = self._analyze_single_community(
                graph, community_id, nodes, centrality_scores
            )
            community_analysis[community_id] = analysis
            total_nodes_in_communities += len(nodes)
        
        # Calculate overall statistics
        sizes = [len(nodes) for nodes in communities.values()]
        densities = [community_analysis[cid]['density'] for cid in communities.keys()]
        
        summary = {
            'total_communities': len(communities),
            'total_nodes': len(graph.nodes()),
            'nodes_in_communities': total_nodes_in_communities,
            'coverage': total_nodes_in_communities / len(graph.nodes()) if len(graph.nodes()) > 0 else 0.0,
            'size_statistics': {
                'mean_size': np.mean(sizes),
                'median_size': np.median(sizes),
                'std_size': np.std(sizes),
                'min_size': min(sizes),
                'max_size': max(sizes)
            },
            'density_statistics': {
                'mean_density': np.mean(densities),
                'median_density': np.median(densities),
                'std_density': np.std(densities),
                'min_density': min(densities),
                'max_density': max(densities)
            }
        }
        
        return {
            'summary': summary,
            'communities': community_analysis
        }
    
    def _analyze_single_community(self, graph: nx.Graph, community_id: int, 
                                 nodes: List, centrality_scores: Optional[Dict] = None) -> Dict:
        """
        Analyze characteristics of a single community.
        
        Args:
            graph: NetworkX graph
            community_id: ID of the community
            nodes: List of nodes in the community
            centrality_scores: Optional centrality metrics
            
        Returns:
            Dictionary with single community analysis
        """
        subgraph = graph.subgraph(nodes)
        num_nodes = len(nodes)
        num_edges = subgraph.number_of_edges()
        
        # Basic metrics
        max_possible_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 0
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0
        
        # Connectivity analysis
        internal_edges = num_edges
        external_edges = 0
        external_connections = defaultdict(int)
        
        for node in nodes:
            for neighbor in graph.neighbors(node):
                if neighbor not in nodes:
                    external_edges += 1
                    # Find which community the external neighbor belongs to
                    external_connections[neighbor] += 1
        
        # Calculate conductance (ratio of external to total edges)
        total_edges = internal_edges + external_edges
        conductance = external_edges / total_edges if total_edges > 0 else 0.0
        
        # Identify key members
        key_members = self._identify_key_members(graph, nodes, centrality_scores)
        
        # Calculate community-specific centrality
        community_centrality = self._calculate_community_centrality(subgraph)
        
        # Analyze node attributes if available
        node_attributes = self._analyze_node_attributes(graph, nodes)
        
        return {
            'id': community_id,
            'size': num_nodes,
            'internal_edges': internal_edges,
            'external_edges': external_edges,
            'density': density,
            'conductance': conductance,
            'key_members': key_members,
            'community_centrality': community_centrality,
            'node_attributes': node_attributes,
            'nodes': nodes,
            'external_connections_count': len(external_connections),
            'avg_external_connections': np.mean(list(external_connections.values())) if external_connections else 0.0
        }
    
    def _identify_key_members(self, graph: nx.Graph, nodes: List, 
                            centrality_scores: Optional[Dict] = None) -> Dict:
        """
        Identify key members within a community based on various criteria.
        
        Args:
            graph: NetworkX graph
            nodes: List of nodes in the community
            centrality_scores: Optional global centrality scores
            
        Returns:
            Dictionary with key members identified by different criteria
        """
        subgraph = graph.subgraph(nodes)
        
        # Calculate local centrality within the community
        local_degree = dict(subgraph.degree())
        local_betweenness = nx.betweenness_centrality(subgraph) if len(nodes) > 2 else {}
        local_closeness = nx.closeness_centrality(subgraph) if len(nodes) > 1 else {}
        
        # Sort nodes by different criteria
        top_k = min(5, len(nodes))  # Top 5 or all nodes if fewer
        
        key_members = {
            'by_local_degree': sorted(local_degree.items(), key=lambda x: x[1], reverse=True)[:top_k],
            'by_local_betweenness': sorted(local_betweenness.items(), key=lambda x: x[1], reverse=True)[:top_k],
            'by_local_closeness': sorted(local_closeness.items(), key=lambda x: x[1], reverse=True)[:top_k]
        }
        
        # Add global centrality rankings if available
        if centrality_scores:
            for centrality_type, scores in centrality_scores.items():
                if scores:
                    community_scores = {node: scores.get(node, 0) for node in nodes}
                    key_members[f'by_global_{centrality_type}'] = sorted(
                        community_scores.items(), key=lambda x: x[1], reverse=True
                    )[:top_k]
        
        # Identify bridge nodes (high external connectivity)
        bridge_scores = {}
        for node in nodes:
            external_connections = sum(1 for neighbor in graph.neighbors(node) if neighbor not in nodes)
            bridge_scores[node] = external_connections
        
        key_members['bridge_nodes'] = sorted(
            bridge_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        
        return key_members
    
    def _calculate_community_centrality(self, subgraph: nx.Graph) -> Dict:
        """
        Calculate centrality metrics within a community subgraph.
        
        Args:
            subgraph: NetworkX subgraph of the community
            
        Returns:
            Dictionary with community-specific centrality metrics
        """
        if len(subgraph.nodes()) == 0:
            return {}
        
        centrality_metrics = {}
        
        # Degree centrality
        centrality_metrics['degree'] = dict(subgraph.degree())
        
        # Betweenness centrality (only if more than 2 nodes)
        if len(subgraph.nodes()) > 2:
            centrality_metrics['betweenness'] = nx.betweenness_centrality(subgraph)
        else:
            centrality_metrics['betweenness'] = {node: 0.0 for node in subgraph.nodes()}
        
        # Closeness centrality (only if more than 1 node)
        if len(subgraph.nodes()) > 1:
            centrality_metrics['closeness'] = nx.closeness_centrality(subgraph)
        else:
            centrality_metrics['closeness'] = {node: 0.0 for node in subgraph.nodes()}
        
        # Clustering coefficient
        centrality_metrics['clustering'] = nx.clustering(subgraph)
        
        return centrality_metrics
    
    def _analyze_node_attributes(self, graph: nx.Graph, nodes: List) -> Dict:
        """
        Analyze node attributes within a community.
        
        Args:
            graph: NetworkX graph
            nodes: List of nodes in the community
            
        Returns:
            Dictionary with node attribute analysis
        """
        attributes_analysis = {
            'available_attributes': [],
            'attribute_distributions': {}
        }
        
        # Check what attributes are available
        if nodes:
            sample_node = nodes[0]
            if sample_node in graph.nodes():
                node_data = graph.nodes[sample_node]
                attributes_analysis['available_attributes'] = list(node_data.keys())
                
                # Analyze distribution of each attribute
                for attr in node_data.keys():
                    attr_values = []
                    for node in nodes:
                        if node in graph.nodes() and attr in graph.nodes[node]:
                            attr_values.append(graph.nodes[node][attr])
                    
                    if attr_values:
                        if isinstance(attr_values[0], (int, float)):
                            # Numerical attribute
                            attributes_analysis['attribute_distributions'][attr] = {
                                'type': 'numerical',
                                'mean': np.mean(attr_values),
                                'std': np.std(attr_values),
                                'min': min(attr_values),
                                'max': max(attr_values),
                                'median': np.median(attr_values)
                            }
                        else:
                            # Categorical attribute
                            value_counts = Counter(attr_values)
                            attributes_analysis['attribute_distributions'][attr] = {
                                'type': 'categorical',
                                'unique_values': len(value_counts),
                                'most_common': value_counts.most_common(5),
                                'distribution': dict(value_counts)
                            }
        
        return attributes_analysis
    
    def compare_communities(self, graph: nx.Graph, communities: Dict[int, List],
                          comparison_metrics: List[str] = None) -> Dict:
        """
        Compare communities across different metrics.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community_id to list of nodes
            comparison_metrics: List of metrics to compare (default: all available)
            
        Returns:
            Dictionary with community comparison results
        """
        if comparison_metrics is None:
            comparison_metrics = ['size', 'density', 'conductance', 'internal_edges', 'external_edges']
        
        comparison_data = {}
        
        for community_id, nodes in communities.items():
            analysis = self._analyze_single_community(graph, community_id, nodes)
            comparison_data[community_id] = {
                metric: analysis.get(metric, 0) for metric in comparison_metrics
            }
        
        # Calculate rankings for each metric
        rankings = {}
        for metric in comparison_metrics:
            metric_values = [(cid, data[metric]) for cid, data in comparison_data.items()]
            # Sort in descending order (higher is better for most metrics)
            reverse_sort = metric not in ['conductance']  # conductance: lower is better
            sorted_values = sorted(metric_values, key=lambda x: x[1], reverse=reverse_sort)
            rankings[metric] = {cid: rank + 1 for rank, (cid, _) in enumerate(sorted_values)}
        
        return {
            'comparison_data': comparison_data,
            'rankings': rankings,
            'metrics_used': comparison_metrics
        }
    
    def identify_community_roles(self, graph: nx.Graph, communities: Dict[int, List]) -> Dict:
        """
        Identify the role of each community in the overall network structure.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community_id to list of nodes
            
        Returns:
            Dictionary with community role analysis
        """
        community_roles = {}
        
        # Calculate inter-community connections
        inter_community_edges = defaultdict(lambda: defaultdict(int))
        
        for community_id, nodes in communities.items():
            for node in nodes:
                for neighbor in graph.neighbors(node):
                    # Find neighbor's community
                    neighbor_community = None
                    for other_id, other_nodes in communities.items():
                        if neighbor in other_nodes:
                            neighbor_community = other_id
                            break
                    
                    if neighbor_community is not None and neighbor_community != community_id:
                        inter_community_edges[community_id][neighbor_community] += 1
        
        # Analyze each community's role
        for community_id, nodes in communities.items():
            analysis = self._analyze_single_community(graph, community_id, nodes)
            
            # Determine role based on characteristics
            role_indicators = {
                'size_percentile': self._calculate_percentile(
                    len(nodes), [len(n) for n in communities.values()]
                ),
                'density_percentile': self._calculate_percentile(
                    analysis['density'], [self._analyze_single_community(graph, cid, n)['density'] 
                                        for cid, n in communities.items()]
                ),
                'external_connectivity': len(inter_community_edges[community_id]),
                'bridge_strength': sum(inter_community_edges[community_id].values())
            }
            
            # Classify role
            role = self._classify_community_role(role_indicators)
            
            community_roles[community_id] = {
                'role': role,
                'indicators': role_indicators,
                'connected_communities': list(inter_community_edges[community_id].keys()),
                'inter_community_connections': dict(inter_community_edges[community_id])
            }
        
        return community_roles
    
    def _calculate_percentile(self, value: float, all_values: List[float]) -> float:
        """Calculate the percentile rank of a value in a list."""
        if not all_values:
            return 0.0
        return (sum(1 for v in all_values if v <= value) / len(all_values)) * 100
    
    def _classify_community_role(self, indicators: Dict) -> str:
        """
        Classify community role based on indicators.
        
        Args:
            indicators: Dictionary with role indicators
            
        Returns:
            String describing the community role
        """
        size_pct = indicators['size_percentile']
        density_pct = indicators['density_percentile']
        external_conn = indicators['external_connectivity']
        bridge_strength = indicators['bridge_strength']
        
        # Classification logic
        if size_pct >= 80 and density_pct >= 70:
            return 'core_hub'  # Large, dense, central community
        elif size_pct >= 80:
            return 'major_community'  # Large but not necessarily dense
        elif density_pct >= 80:
            return 'tight_knit'  # Small but very dense
        elif external_conn >= 3 and bridge_strength >= 5:
            return 'bridge_community'  # Connects many other communities
        elif external_conn <= 1:
            return 'isolated'  # Few external connections
        elif size_pct <= 20:
            return 'peripheral'  # Small and on the periphery
        else:
            return 'intermediate'  # Middle ground
    
    def generate_community_summary(self, graph: nx.Graph, communities: Dict[int, List],
                                 method: str = 'unknown', modularity: float = None) -> str:
        """
        Generate a human-readable summary of community analysis results.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community_id to list of nodes
            method: Community detection method used
            modularity: Modularity score of the partition
            
        Returns:
            String with comprehensive community analysis summary
        """
        if not communities:
            return "No communities detected in the network."
        
        # Perform comprehensive analysis
        characteristics = self.analyze_community_characteristics(graph, communities)
        roles = self.identify_community_roles(graph, communities)
        
        summary_parts = []
        
        # Header
        summary_parts.append(f"Community Analysis Summary ({method.title()} Method)")
        summary_parts.append("=" * 60)
        
        # Overall statistics
        summary = characteristics['summary']
        summary_parts.append(f"\nOverall Network Statistics:")
        summary_parts.append(f"  • Total nodes: {summary['total_nodes']}")
        summary_parts.append(f"  • Communities detected: {summary['total_communities']}")
        summary_parts.append(f"  • Coverage: {summary['coverage']:.1%} of nodes in communities")
        if modularity is not None:
            summary_parts.append(f"  • Modularity score: {modularity:.3f}")
        
        # Community size distribution
        size_stats = summary['size_statistics']
        summary_parts.append(f"\nCommunity Size Distribution:")
        summary_parts.append(f"  • Average size: {size_stats['mean_size']:.1f} nodes")
        summary_parts.append(f"  • Size range: {size_stats['min_size']} - {size_stats['max_size']} nodes")
        summary_parts.append(f"  • Standard deviation: {size_stats['std_size']:.1f}")
        
        # Community density distribution
        density_stats = summary['density_statistics']
        summary_parts.append(f"\nCommunity Density Distribution:")
        summary_parts.append(f"  • Average density: {density_stats['mean_density']:.3f}")
        summary_parts.append(f"  • Density range: {density_stats['min_density']:.3f} - {density_stats['max_density']:.3f}")
        
        # Top communities by different criteria
        communities_by_size = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
        summary_parts.append(f"\nLargest Communities:")
        for i, (cid, nodes) in enumerate(communities_by_size[:3]):
            role = roles[cid]['role'].replace('_', ' ').title()
            summary_parts.append(f"  {i+1}. Community {cid}: {len(nodes)} nodes ({role})")
        
        # Community roles summary
        role_counts = Counter(role_data['role'] for role_data in roles.values())
        summary_parts.append(f"\nCommunity Roles Distribution:")
        for role, count in role_counts.most_common():
            role_name = role.replace('_', ' ').title()
            summary_parts.append(f"  • {role_name}: {count} communities")
        
        # Key insights
        summary_parts.append(f"\nKey Insights:")
        
        # Identify most connected communities
        bridge_communities = [cid for cid, data in roles.items() if data['role'] == 'bridge_community']
        if bridge_communities:
            summary_parts.append(f"  • {len(bridge_communities)} bridge communities facilitate inter-community connections")
        
        # Identify isolated communities
        isolated_communities = [cid for cid, data in roles.items() if data['role'] == 'isolated']
        if isolated_communities:
            summary_parts.append(f"  • {len(isolated_communities)} isolated communities with minimal external connections")
        
        # Network cohesion assessment
        if density_stats['mean_density'] > 0.5:
            summary_parts.append("  • High intra-community density suggests strong community cohesion")
        elif density_stats['mean_density'] < 0.2:
            summary_parts.append("  • Low intra-community density suggests loose community structure")
        
        return "\n".join(summary_parts)    

    def generate_detailed_documentation(self, graph: nx.Graph, communities: Dict[int, List],
                                      method: str = 'unknown', modularity: float = None,
                                      centrality_scores: Optional[Dict] = None,
                                      output_format: str = 'markdown') -> str:
        """
        Generate comprehensive documentation of community analysis results.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community_id to list of nodes
            method: Community detection method used
            modularity: Modularity score of the partition
            centrality_scores: Optional centrality metrics
            output_format: Output format ('markdown', 'html', 'text')
            
        Returns:
            String with detailed documentation in specified format
        """
        # Perform comprehensive analysis
        characteristics = self.analyze_community_characteristics(graph, communities, centrality_scores)
        roles = self.identify_community_roles(graph, communities)
        comparison = self.compare_communities(graph, communities)
        
        if output_format.lower() == 'markdown':
            return self._generate_markdown_documentation(
                graph, communities, characteristics, roles, comparison, method, modularity
            )
        elif output_format.lower() == 'html':
            return self._generate_html_documentation(
                graph, communities, characteristics, roles, comparison, method, modularity
            )
        else:
            return self._generate_text_documentation(
                graph, communities, characteristics, roles, comparison, method, modularity
            )
    
    def _generate_markdown_documentation(self, graph: nx.Graph, communities: Dict[int, List],
                                       characteristics: Dict, roles: Dict, comparison: Dict,
                                       method: str, modularity: float) -> str:
        """Generate documentation in Markdown format."""
        doc_parts = []
        
        # Header
        doc_parts.append(f"# Community Analysis Report")
        doc_parts.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc_parts.append(f"**Detection Method:** {method.title()}")
        if modularity is not None:
            doc_parts.append(f"**Modularity Score:** {modularity:.4f}")
        doc_parts.append("")
        
        # Executive Summary
        doc_parts.append("## Executive Summary")
        summary = characteristics['summary']
        doc_parts.append(f"The network analysis identified **{summary['total_communities']} communities** "
                        f"within a network of {summary['total_nodes']} nodes, achieving "
                        f"{summary['coverage']:.1%} coverage. ")
        
        if modularity is not None:
            if modularity > 0.3:
                doc_parts.append("The high modularity score indicates strong community structure.")
            elif modularity > 0.1:
                doc_parts.append("The moderate modularity score suggests reasonable community structure.")
            else:
                doc_parts.append("The low modularity score indicates weak community structure.")
        
        doc_parts.append("")
        
        # Network Overview
        doc_parts.append("## Network Overview")
        doc_parts.append("| Metric | Value |")
        doc_parts.append("|--------|-------|")
        doc_parts.append(f"| Total Nodes | {summary['total_nodes']} |")
        doc_parts.append(f"| Total Communities | {summary['total_communities']} |")
        doc_parts.append(f"| Coverage | {summary['coverage']:.1%} |")
        doc_parts.append(f"| Average Community Size | {summary['size_statistics']['mean_size']:.1f} |")
        doc_parts.append(f"| Average Community Density | {summary['density_statistics']['mean_density']:.3f} |")
        doc_parts.append("")
        
        # Community Details
        doc_parts.append("## Community Details")
        
        # Sort communities by size for presentation
        sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
        
        for community_id, nodes in sorted_communities:
            community_data = characteristics['communities'][community_id]
            role_data = roles[community_id]
            
            doc_parts.append(f"### Community {community_id}")
            doc_parts.append(f"**Role:** {role_data['role'].replace('_', ' ').title()}")
            doc_parts.append(f"**Size:** {community_data['size']} nodes")
            doc_parts.append(f"**Density:** {community_data['density']:.3f}")
            doc_parts.append(f"**Internal Edges:** {community_data['internal_edges']}")
            doc_parts.append(f"**External Edges:** {community_data['external_edges']}")
            doc_parts.append(f"**Conductance:** {community_data['conductance']:.3f}")
            
            # Key members
            if community_data['key_members']['by_local_degree']:
                doc_parts.append(f"**Top Members by Local Degree:**")
                for node, degree in community_data['key_members']['by_local_degree'][:3]:
                    doc_parts.append(f"- {node} (degree: {degree})")
            
            # Connected communities
            if role_data['connected_communities']:
                doc_parts.append(f"**Connected to Communities:** {', '.join(map(str, role_data['connected_communities']))}")
            
            doc_parts.append("")
        
        # Community Roles Analysis
        doc_parts.append("## Community Roles Analysis")
        role_counts = Counter(role_data['role'] for role_data in roles.values())
        
        for role, count in role_counts.most_common():
            role_name = role.replace('_', ' ').title()
            doc_parts.append(f"**{role_name}:** {count} communities")
            
            # List communities with this role
            role_communities = [cid for cid, data in roles.items() if data['role'] == role]
            if len(role_communities) <= 5:
                doc_parts.append(f"- Communities: {', '.join(map(str, role_communities))}")
            else:
                doc_parts.append(f"- Communities: {', '.join(map(str, role_communities[:5]))} and {len(role_communities)-5} others")
            doc_parts.append("")
        
        # Rankings
        doc_parts.append("## Community Rankings")
        rankings = comparison['rankings']
        
        for metric in ['size', 'density', 'internal_edges']:
            if metric in rankings:
                doc_parts.append(f"### By {metric.replace('_', ' ').title()}")
                sorted_by_metric = sorted(rankings[metric].items(), key=lambda x: x[1])
                for rank, (community_id, rank_value) in enumerate(sorted_by_metric[:5], 1):
                    actual_value = comparison['comparison_data'][community_id][metric]
                    doc_parts.append(f"{rank}. Community {community_id}: {actual_value}")
                doc_parts.append("")
        
        # Methodology
        doc_parts.append("## Methodology")
        doc_parts.append(f"This analysis used the **{method.title()}** algorithm for community detection. ")
        doc_parts.append("Key metrics calculated include:")
        doc_parts.append("- **Density:** Ratio of actual edges to possible edges within the community")
        doc_parts.append("- **Conductance:** Ratio of external edges to total edges (lower is better)")
        doc_parts.append("- **Modularity:** Quality measure of the community partition")
        doc_parts.append("- **Centrality:** Various measures of node importance within communities")
        doc_parts.append("")
        
        return "\n".join(doc_parts)
    
    def _generate_html_documentation(self, graph: nx.Graph, communities: Dict[int, List],
                                   characteristics: Dict, roles: Dict, comparison: Dict,
                                   method: str, modularity: float) -> str:
        """Generate documentation in HTML format."""
        # Convert markdown to basic HTML structure
        markdown_doc = self._generate_markdown_documentation(
            graph, communities, characteristics, roles, comparison, method, modularity
        )
        
        # Basic markdown to HTML conversion
        html_doc = markdown_doc.replace("# ", "<h1>").replace("\n## ", "</h1>\n<h2>")
        html_doc = html_doc.replace("\n### ", "</h2>\n<h3>").replace("**", "<strong>")
        html_doc = html_doc.replace("</strong>", "</strong>")
        html_doc = html_doc.replace("\n- ", "\n<li>").replace("\n\n", "</h3>\n\n")
        
        # Wrap in basic HTML structure
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Community Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        h1, h2, h3 {{ color: #333; }}
    </style>
</head>
<body>
{html_doc}
</body>
</html>
"""
        return html_template
    
    def _generate_text_documentation(self, graph: nx.Graph, communities: Dict[int, List],
                                   characteristics: Dict, roles: Dict, comparison: Dict,
                                   method: str, modularity: float) -> str:
        """Generate documentation in plain text format."""
        # Convert markdown to plain text by removing formatting
        markdown_doc = self._generate_markdown_documentation(
            graph, communities, characteristics, roles, comparison, method, modularity
        )
        
        # Remove markdown formatting
        text_doc = markdown_doc.replace("# ", "").replace("## ", "").replace("### ", "")
        text_doc = text_doc.replace("**", "").replace("| ", "").replace(" |", "")
        text_doc = text_doc.replace("|--------|-------|", "")
        
        return text_doc
    
    def export_analysis_results(self, graph: nx.Graph, communities: Dict[int, List],
                              method: str = 'unknown', modularity: float = None,
                              centrality_scores: Optional[Dict] = None,
                              output_dir: str = '.', formats: List[str] = None) -> Dict[str, str]:
        """
        Export community analysis results in multiple formats.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community_id to list of nodes
            method: Community detection method used
            modularity: Modularity score
            centrality_scores: Optional centrality metrics
            output_dir: Directory to save files
            formats: List of formats to export ('json', 'csv', 'markdown', 'html')
            
        Returns:
            Dictionary mapping format to file path
        """
        if formats is None:
            formats = ['json', 'csv', 'markdown']
        
        # Perform comprehensive analysis
        characteristics = self.analyze_community_characteristics(graph, communities, centrality_scores)
        roles = self.identify_community_roles(graph, communities)
        comparison = self.compare_communities(graph, communities)
        
        exported_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"community_analysis_{method}_{timestamp}"
        
        # Export JSON
        if 'json' in formats:
            json_data = {
                'metadata': {
                    'method': method,
                    'modularity': modularity,
                    'analysis_date': datetime.now().isoformat(),
                    'total_nodes': len(graph.nodes()),
                    'total_communities': len(communities)
                },
                'communities': communities,
                'characteristics': characteristics,
                'roles': roles,
                'comparison': comparison
            }
            
            json_path = f"{output_dir}/{base_filename}.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            exported_files['json'] = json_path
        
        # Export CSV
        if 'csv' in formats:
            # Create community summary CSV
            csv_data = []
            for community_id, nodes in communities.items():
                community_data = characteristics['communities'][community_id]
                role_data = roles[community_id]
                
                csv_data.append({
                    'community_id': community_id,
                    'size': community_data['size'],
                    'density': community_data['density'],
                    'internal_edges': community_data['internal_edges'],
                    'external_edges': community_data['external_edges'],
                    'conductance': community_data['conductance'],
                    'role': role_data['role'],
                    'connected_communities': len(role_data['connected_communities']),
                    'nodes': ','.join(map(str, nodes))
                })
            
            df = pd.DataFrame(csv_data)
            csv_path = f"{output_dir}/{base_filename}.csv"
            df.to_csv(csv_path, index=False)
            exported_files['csv'] = csv_path
        
        # Export Markdown
        if 'markdown' in formats:
            markdown_doc = self.generate_detailed_documentation(
                graph, communities, method, modularity, centrality_scores, 'markdown'
            )
            markdown_path = f"{output_dir}/{base_filename}.md"
            with open(markdown_path, 'w') as f:
                f.write(markdown_doc)
            exported_files['markdown'] = markdown_path
        
        # Export HTML
        if 'html' in formats:
            html_doc = self.generate_detailed_documentation(
                graph, communities, method, modularity, centrality_scores, 'html'
            )
            html_path = f"{output_dir}/{base_filename}.html"
            with open(html_path, 'w') as f:
                f.write(html_doc)
            exported_files['html'] = html_path
        
        return exported_files
    
    def create_community_visualization_report(self, graph: nx.Graph, communities: Dict[int, List],
                                            method: str = 'unknown', save_dir: str = '.') -> str:
        """
        Create a comprehensive visualization report for communities.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community_id to list of nodes
            method: Community detection method used
            save_dir: Directory to save visualizations
            
        Returns:
            Path to the generated visualization report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create visualizations
        plt.style.use('default')
        
        # 1. Community size distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sizes = [len(nodes) for nodes in communities.values()]
        plt.hist(sizes, bins=min(10, len(communities)), alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Community Size')
        plt.ylabel('Frequency')
        plt.title('Community Size Distribution')
        plt.grid(True, alpha=0.3)
        
        # 2. Community density distribution
        plt.subplot(2, 2, 2)
        characteristics = self.analyze_community_characteristics(graph, communities)
        densities = [characteristics['communities'][cid]['density'] for cid in communities.keys()]
        plt.hist(densities, bins=min(10, len(communities)), alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Community Density')
        plt.ylabel('Frequency')
        plt.title('Community Density Distribution')
        plt.grid(True, alpha=0.3)
        
        # 3. Size vs Density scatter plot
        plt.subplot(2, 2, 3)
        plt.scatter(sizes, densities, alpha=0.7, color='coral', s=60)
        plt.xlabel('Community Size')
        plt.ylabel('Community Density')
        plt.title('Size vs Density Relationship')
        plt.grid(True, alpha=0.3)
        
        # 4. Community roles pie chart
        plt.subplot(2, 2, 4)
        roles = self.identify_community_roles(graph, communities)
        role_counts = Counter(role_data['role'] for role_data in roles.values())
        
        labels = [role.replace('_', ' ').title() for role in role_counts.keys()]
        sizes_pie = list(role_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        plt.pie(sizes_pie, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Community Roles Distribution')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = f"{save_dir}/community_analysis_viz_{method}_{timestamp}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def get_community_insights(self, graph: nx.Graph, communities: Dict[int, List],
                             centrality_scores: Optional[Dict] = None) -> List[str]:
        """
        Generate actionable insights from community analysis.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping community_id to list of nodes
            centrality_scores: Optional centrality metrics
            
        Returns:
            List of insight strings
        """
        insights = []
        
        if not communities:
            insights.append("No communities detected - the network may be too sparse or uniform.")
            return insights
        
        # Analyze characteristics
        characteristics = self.analyze_community_characteristics(graph, communities, centrality_scores)
        roles = self.identify_community_roles(graph, communities)
        
        summary = characteristics['summary']
        
        # Coverage insights
        if summary['coverage'] < 0.5:
            insights.append(f"Low community coverage ({summary['coverage']:.1%}) suggests many isolated nodes or weak community structure.")
        elif summary['coverage'] > 0.9:
            insights.append(f"High community coverage ({summary['coverage']:.1%}) indicates strong community organization.")
        
        # Size distribution insights
        size_stats = summary['size_statistics']
        if size_stats['std_size'] > size_stats['mean_size']:
            insights.append("High variation in community sizes suggests hierarchical or scale-free community structure.")
        
        # Density insights
        density_stats = summary['density_statistics']
        if density_stats['mean_density'] > 0.5:
            insights.append("High average community density indicates strong internal cohesion.")
        elif density_stats['mean_density'] < 0.2:
            insights.append("Low average community density suggests loose community boundaries.")
        
        # Role-based insights
        role_counts = Counter(role_data['role'] for role_data in roles.values())
        
        if role_counts.get('bridge_community', 0) > 0:
            insights.append(f"{role_counts['bridge_community']} bridge communities facilitate information flow between groups.")
        
        if role_counts.get('isolated', 0) > len(communities) * 0.3:
            insights.append("Many isolated communities suggest fragmented network structure.")
        
        if role_counts.get('core_hub', 0) > 0:
            insights.append(f"{role_counts['core_hub']} core hub communities likely drive network dynamics.")
        
        # Largest community insights
        largest_community = max(communities.items(), key=lambda x: len(x[1]))
        largest_size = len(largest_community[1])
        if largest_size > len(graph.nodes()) * 0.3:
            insights.append(f"The largest community contains {largest_size} nodes ({largest_size/len(graph.nodes()):.1%} of network), indicating potential dominance.")
        
        return insights