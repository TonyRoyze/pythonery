"""
ResultInterpreter class for interpreting analysis results.

This module provides interpretation methods for all analysis components,
generating human-readable explanations and insights from numerical results.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import Counter


class ResultInterpreter:
    """
    Interprets social network analysis results and generates insights.
    
    Provides methods to convert numerical analysis results into meaningful
    interpretations and actionable insights for different stakeholders.
    """
    
    def __init__(self):
        """Initialize ResultInterpreter."""
        self.logger = logging.getLogger(__name__)
        
        # Interpretation thresholds
        self.centrality_thresholds = {
            'high': 0.7,
            'medium': 0.3,
            'low': 0.1
        }
        
        self.modularity_thresholds = {
            'excellent': 0.5,
            'good': 0.3,
            'moderate': 0.1,
            'poor': 0.0
        }
        
        self.influence_thresholds = {
            'high': 0.5,
            'medium': 0.2,
            'low': 0.05
        }
    
    def extract_key_insights(self, graph: nx.Graph, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key insights from comprehensive analysis results.
        
        Args:
            graph: NetworkX graph that was analyzed
            analysis_results: Dictionary containing all analysis results
            
        Returns:
            Dictionary containing key insights and summary
        """
        insights = []
        
        # Network structure insights
        network_insights = self._analyze_network_structure(graph)
        insights.extend(network_insights)
        
        # Component-specific insights
        if 'centrality' in analysis_results:
            centrality_insights = self._extract_centrality_insights(analysis_results['centrality'])
            insights.extend(centrality_insights)
        
        if 'communities' in analysis_results:
            community_insights = self._extract_community_insights(analysis_results['communities'])
            insights.extend(community_insights)
        
        if 'influence_propagation' in analysis_results:
            influence_insights = self._extract_influence_insights(analysis_results['influence_propagation'])
            insights.extend(influence_insights)
        
        if 'link_prediction' in analysis_results:
            prediction_insights = self._extract_prediction_insights(analysis_results['link_prediction'])
            insights.extend(prediction_insights)
        
        # Generate executive summary
        summary = self._generate_executive_summary(graph, analysis_results, insights)
        
        return {
            'insights': insights,
            'summary': summary,
            'recommendations': self._generate_recommendations(insights)
        }
    
    def interpret_centrality_results(self, centrality_results: Dict[str, Any]) -> str:
        """
        Interpret centrality analysis results.
        
        Args:
            centrality_results: Results from centrality analysis
            
        Returns:
            Human-readable interpretation of centrality results
        """
        interpretation_parts = []
        
        # Analyze centrality distributions
        if 'summary_statistics' in centrality_results:
            stats = centrality_results['summary_statistics']
            
            # Degree centrality interpretation
            if 'degree' in stats:
                degree_stats = stats['degree']
                if degree_stats['std'] > degree_stats['mean']:
                    interpretation_parts.append(
                        "The network shows high variability in node connectivity, "
                        "indicating the presence of both highly connected hubs and peripheral nodes."
                    )
                else:
                    interpretation_parts.append(
                        "Node connectivity is relatively uniform across the network, "
                        "suggesting a more egalitarian structure."
                    )
            
            # Betweenness centrality interpretation
            if 'betweenness' in stats:
                betweenness_stats = stats['betweenness']
                if betweenness_stats['max'] > 0.1:
                    interpretation_parts.append(
                        "Some nodes serve as critical bridges in the network, "
                        "controlling information flow between different parts."
                    )
                else:
                    interpretation_parts.append(
                        "The network lacks strong bridge nodes, indicating "
                        "more distributed information flow patterns."
                    )
        
        # Analyze correlations
        if 'centrality_correlations' in centrality_results:
            correlations = centrality_results['centrality_correlations']
            
            # Degree vs Betweenness correlation
            degree_betweenness_corr = correlations.get('degree_vs_betweenness')
            if degree_betweenness_corr is not None:
                if degree_betweenness_corr > 0.7:
                    interpretation_parts.append(
                        "High-degree nodes also tend to be important bridges, "
                        "suggesting a centralized network structure."
                    )
                elif degree_betweenness_corr < 0.3:
                    interpretation_parts.append(
                        "Bridge nodes are not necessarily the most connected, "
                        "indicating a more complex network topology."
                    )
        
        # Top nodes analysis
        if 'top_nodes' in centrality_results:
            top_nodes = centrality_results['top_nodes']
            
            # Check for overlap in top nodes across metrics
            if 'degree' in top_nodes and 'betweenness' in top_nodes:
                degree_top = set(node for node, _ in top_nodes['degree'][:5])
                betweenness_top = set(node for node, _ in top_nodes['betweenness'][:5])
                overlap = len(degree_top & betweenness_top)
                
                if overlap >= 3:
                    interpretation_parts.append(
                        f"There is significant overlap ({overlap}/5) between the most connected "
                        "and most influential bridge nodes, indicating clear network leaders."
                    )
                else:
                    interpretation_parts.append(
                        "Different nodes excel in different centrality measures, "
                        "suggesting diverse roles within the network."
                    )
        
        return " ".join(interpretation_parts) if interpretation_parts else "Centrality analysis completed successfully."
    
    def interpret_community_results(self, community_results: Dict[str, Any]) -> str:
        """
        Interpret community detection results.
        
        Args:
            community_results: Results from community detection
            
        Returns:
            Human-readable interpretation of community results
        """
        interpretation_parts = []
        
        # Overall community structure
        if 'summary' in community_results:
            summary = community_results['summary']
            num_communities = summary.get('total_communities', 0)
            coverage = summary.get('coverage', 0)
            
            if num_communities == 0:
                return "No clear community structure was detected in the network."
            
            if coverage > 0.9:
                interpretation_parts.append(
                    f"The network exhibits clear community structure with {num_communities} "
                    f"distinct communities covering {coverage:.1%} of all nodes."
                )
            else:
                interpretation_parts.append(
                    f"The network shows moderate community structure with {num_communities} "
                    f"communities, though {1-coverage:.1%} of nodes remain unclustered."
                )
            
            # Community size analysis
            size_stats = summary.get('size_statistics', {})
            mean_size = size_stats.get('mean_size', 0)
            std_size = size_stats.get('std_size', 0)
            
            if std_size > mean_size:
                interpretation_parts.append(
                    "Community sizes vary significantly, with some large communities "
                    "and many smaller groups."
                )
            else:
                interpretation_parts.append(
                    "Communities are relatively similar in size, indicating "
                    "balanced group formation."
                )
        
        # Modularity interpretation
        modularity = community_results.get('modularity')
        if modularity is not None:
            if modularity > self.modularity_thresholds['excellent']:
                interpretation_parts.append(
                    f"The modularity score of {modularity:.3f} indicates excellent "
                    "community structure with strong internal connections and weak inter-community ties."
                )
            elif modularity > self.modularity_thresholds['good']:
                interpretation_parts.append(
                    f"The modularity score of {modularity:.3f} indicates good "
                    "community structure with clear group boundaries."
                )
            elif modularity > self.modularity_thresholds['moderate']:
                interpretation_parts.append(
                    f"The modularity score of {modularity:.3f} indicates moderate "
                    "community structure with some group cohesion."
                )
            else:
                interpretation_parts.append(
                    f"The low modularity score of {modularity:.3f} suggests weak "
                    "community structure or a more random network organization."
                )
        
        # Community roles analysis
        if 'communities' in community_results:
            communities = community_results['communities']
            
            # Analyze community roles if available
            roles = []
            for community_data in communities.values():
                role = community_data.get('role', 'unknown')
                roles.append(role)
            
            role_counts = Counter(roles)
            
            if 'core_hub' in role_counts:
                interpretation_parts.append(
                    f"{role_counts['core_hub']} communities serve as core hubs, "
                    "likely representing central groups in the network."
                )
            
            if 'bridge_community' in role_counts:
                interpretation_parts.append(
                    f"{role_counts['bridge_community']} communities act as bridges, "
                    "facilitating connections between different network regions."
                )
            
            if 'isolated' in role_counts:
                interpretation_parts.append(
                    f"{role_counts['isolated']} communities are relatively isolated, "
                    "suggesting specialized or peripheral groups."
                )
        
        return " ".join(interpretation_parts) if interpretation_parts else "Community analysis completed successfully."
    
    def interpret_influence_results(self, influence_results: Dict[str, Any]) -> str:
        """
        Interpret influence propagation results.
        
        Args:
            influence_results: Results from influence propagation analysis
            
        Returns:
            Human-readable interpretation of influence results
        """
        interpretation_parts = []
        
        # Strategy comparison analysis
        if 'strategy_comparison' in influence_results:
            strategy_comparison = influence_results['strategy_comparison']
            
            # Find best strategy
            best_strategy = None
            best_influence = 0
            
            for strategy, results in strategy_comparison.items():
                for num_seeds, metrics in results.items():
                    mean_influence = metrics.get('mean_influence', 0)
                    if mean_influence > best_influence:
                        best_influence = mean_influence
                        best_strategy = strategy
                    break  # Only check first entry
            
            if best_strategy:
                interpretation_parts.append(
                    f"The {best_strategy.replace('_', ' ')} seed selection strategy "
                    f"achieved the highest influence spread of {best_influence:.1%}."
                )
        
        # Best strategy detailed results
        if 'best_strategy_results' in influence_results:
            best_results = influence_results['best_strategy_results']
            influence_ratio = best_results.get('influence_ratio', 0)
            steps_taken = best_results.get('steps_taken', 0)
            
            if influence_ratio > self.influence_thresholds['high']:
                interpretation_parts.append(
                    f"Information spread achieved high penetration ({influence_ratio:.1%}) "
                    f"across the network in {steps_taken} propagation steps."
                )
            elif influence_ratio > self.influence_thresholds['medium']:
                interpretation_parts.append(
                    f"Information spread achieved moderate penetration ({influence_ratio:.1%}) "
                    f"in {steps_taken} steps, indicating some network resistance."
                )
            else:
                interpretation_parts.append(
                    f"Information spread was limited ({influence_ratio:.1%}), "
                    "suggesting strong network clustering or low activation probabilities."
                )
        
        # Model-specific insights
        model = influence_results.get('analysis_parameters', {}).get('propagation_model', 'unknown')
        if model == 'independent_cascade':
            interpretation_parts.append(
                "The Independent Cascade model simulates viral spread where "
                "each activated node has one chance to influence its neighbors."
            )
        elif model == 'linear_threshold':
            interpretation_parts.append(
                "The Linear Threshold model simulates cumulative influence where "
                "nodes activate when enough neighbors are already active."
            )
        
        return " ".join(interpretation_parts) if interpretation_parts else "Influence propagation analysis completed successfully."
    
    def interpret_link_prediction_results(self, link_prediction_results: Dict[str, Any]) -> str:
        """
        Interpret link prediction results.
        
        Args:
            link_prediction_results: Results from link prediction analysis
            
        Returns:
            Human-readable interpretation of link prediction results
        """
        interpretation_parts = []
        
        # Model performance comparison
        if 'model_comparison' in link_prediction_results:
            comparison = link_prediction_results['model_comparison']
            
            # Best model analysis
            if 'model_ranking' in comparison and 'overall_best' in comparison['model_ranking']:
                best_model = comparison['model_ranking']['overall_best']
                model_name = best_model.get('model', 'Unknown')
                f1_score = best_model.get('metrics', {}).get('f1_score', 0)
                
                if f1_score > 0.8:
                    interpretation_parts.append(
                        f"The {model_name} model achieved excellent performance "
                        f"(F1-score: {f1_score:.3f}), indicating strong predictive capability."
                    )
                elif f1_score > 0.6:
                    interpretation_parts.append(
                        f"The {model_name} model achieved good performance "
                        f"(F1-score: {f1_score:.3f}), showing reasonable predictive accuracy."
                    )
                else:
                    interpretation_parts.append(
                        f"The best model ({model_name}) achieved moderate performance "
                        f"(F1-score: {f1_score:.3f}), suggesting challenging prediction task."
                    )
            
            # Graph vs traditional ML comparison
            if 'insights' in comparison:
                insights = comparison['insights']
                for insight in insights:
                    if 'graph-based' in insight.lower() or 'traditional' in insight.lower():
                        interpretation_parts.append(insight)
        
        # Feature importance analysis
        if 'feature_importance' in link_prediction_results:
            feature_importance = link_prediction_results['feature_importance']
            
            if 'top_features' in feature_importance:
                top_features = feature_importance['top_features'].get('features', [])[:3]
                if top_features:
                    interpretation_parts.append(
                        f"The most predictive features are {', '.join(top_features)}, "
                        "indicating these network properties are key for link formation."
                    )
        
        # Statistical significance
        if 'statistical_significance' in link_prediction_results:
            sig_results = link_prediction_results['statistical_significance']
            
            if 'summary' in sig_results:
                significant_differences = sig_results['summary'].get('significant_differences_found', False)
                if significant_differences:
                    interpretation_parts.append(
                        "Statistical analysis reveals significant performance differences "
                        "between models, validating the model comparison results."
                    )
                else:
                    interpretation_parts.append(
                        "No statistically significant differences were found between models, "
                        "suggesting similar predictive capabilities across approaches."
                    )
        
        return " ".join(interpretation_parts) if interpretation_parts else "Link prediction analysis completed successfully."
    
    def _analyze_network_structure(self, graph: nx.Graph) -> List[str]:
        """Analyze basic network structure and generate insights."""
        insights = []
        
        # Network size insights
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        if num_nodes > 10000:
            insights.append("This is a large-scale network requiring specialized analysis techniques.")
        elif num_nodes > 1000:
            insights.append("This is a medium-scale network suitable for comprehensive analysis.")
        else:
            insights.append("This is a small-scale network allowing for detailed individual node analysis.")
        
        # Density insights
        density = nx.density(graph)
        if density > 0.1:
            insights.append("The network is relatively dense with many connections between nodes.")
        elif density > 0.01:
            insights.append("The network has moderate connectivity with selective connections.")
        else:
            insights.append("The network is sparse with few connections relative to its size.")
        
        # Connectivity insights
        if nx.is_connected(graph):
            insights.append("The network is fully connected, allowing information flow between all nodes.")
        else:
            num_components = nx.number_connected_components(graph)
            insights.append(f"The network has {num_components} disconnected components, indicating fragmentation.")
        
        return insights
    
    def _extract_centrality_insights(self, centrality_results: Dict[str, Any]) -> List[str]:
        """Extract insights from centrality analysis."""
        insights = []
        
        if 'top_nodes' in centrality_results:
            top_nodes = centrality_results['top_nodes']
            
            # Check for consistent top performers
            if 'degree' in top_nodes and 'betweenness' in top_nodes:
                degree_top = [node for node, _ in top_nodes['degree'][:3]]
                betweenness_top = [node for node, _ in top_nodes['betweenness'][:3]]
                
                common_nodes = set(degree_top) & set(betweenness_top)
                if common_nodes:
                    insights.append(
                        f"Nodes {', '.join(map(str, common_nodes))} consistently rank high across "
                        "multiple centrality measures, indicating key network influencers."
                    )
        
        return insights
    
    def _extract_community_insights(self, community_results: Dict[str, Any]) -> List[str]:
        """Extract insights from community detection."""
        insights = []
        
        if 'summary' in community_results:
            summary = community_results['summary']
            num_communities = summary.get('total_communities', 0)
            
            if num_communities > 10:
                insights.append("The network exhibits high fragmentation with many small communities.")
            elif num_communities > 5:
                insights.append("The network shows moderate community structure with several distinct groups.")
            elif num_communities > 1:
                insights.append("The network has clear but limited community structure.")
        
        return insights
    
    def _extract_influence_insights(self, influence_results: Dict[str, Any]) -> List[str]:
        """Extract insights from influence propagation."""
        insights = []
        
        if 'best_strategy_results' in influence_results:
            best_results = influence_results['best_strategy_results']
            influence_ratio = best_results.get('influence_ratio', 0)
            
            if influence_ratio > 0.5:
                insights.append("Information spreads rapidly through the network, indicating high connectivity.")
            elif influence_ratio < 0.1:
                insights.append("Information spread is limited, suggesting network clustering or resistance.")
        
        return insights
    
    def _extract_prediction_insights(self, prediction_results: Dict[str, Any]) -> List[str]:
        """Extract insights from link prediction."""
        insights = []
        
        if 'best_model' in prediction_results:
            best_model = prediction_results['best_model']
            f1_score = best_model.get('f1_score', 0)
            
            if f1_score > 0.7:
                insights.append("Link prediction models show strong performance, indicating predictable network evolution.")
            elif f1_score < 0.5:
                insights.append("Link prediction is challenging, suggesting complex or random network dynamics.")
        
        return insights
    
    def _generate_executive_summary(self, graph: nx.Graph, analysis_results: Dict[str, Any], 
                                  insights: List[str]) -> str:
        """Generate executive summary from analysis results."""
        summary_parts = []
        
        # Network overview
        summary_parts.append(
            f"Analysis of a social network with {graph.number_of_nodes():,} nodes "
            f"and {graph.number_of_edges():,} edges reveals key structural patterns and dynamics."
        )
        
        # Key findings
        if insights:
            summary_parts.append("Key findings include:")
            for insight in insights[:5]:  # Top 5 insights
                summary_parts.append(f"• {insight}")
        
        # Component-specific highlights
        components_analyzed = len(analysis_results)
        summary_parts.append(
            f"The comprehensive analysis covered {components_analyzed} major components "
            "providing insights into network structure, community organization, and predictive patterns."
        )
        
        return " ".join(summary_parts)
    
    def _generate_recommendations(self, insights: List[str]) -> List[str]:
        """Generate actionable recommendations based on insights."""
        recommendations = []
        
        # Generic recommendations based on common patterns
        recommendations.append(
            "Focus on high-centrality nodes for maximum network impact and information dissemination."
        )
        
        recommendations.append(
            "Consider community structure when designing interventions or information campaigns."
        )
        
        recommendations.append(
            "Monitor bridge nodes and communities as they are critical for network connectivity."
        )
        
        recommendations.append(
            "Use predictive models to anticipate network evolution and plan strategic actions."
        )
        
        return recommendations