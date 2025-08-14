"""
DocumentationGenerator class for automated report creation.

This module provides comprehensive documentation generation for all social network
analysis components, creating detailed reports with interpretations, visualizations,
and statistical summaries.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging
import pandas as pd
import networkx as nx
from pathlib import Path

from .result_interpreter import ResultInterpreter


class DocumentationGenerator:
    """
    Generates comprehensive documentation for social network analysis results.
    
    Creates detailed reports combining analysis results, interpretations,
    visualizations, and statistical summaries in multiple formats including
    Markdown, HTML, and PDF.
    """
    
    def __init__(self, output_dir: str = "outputs/documentation"):
        """
        Initialize DocumentationGenerator.
        
        Args:
            output_dir: Directory to save generated documentation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.interpreter = ResultInterpreter()
        
        # Report metadata
        self.report_metadata = {
            'generated_at': None,
            'analysis_components': [],
            'total_nodes': 0,
            'total_edges': 0,
            'analysis_parameters': {}
        }
        
    def generate_comprehensive_report(self,
                                    graph: nx.Graph,
                                    analysis_results: Dict[str, Any],
                                    report_title: str = "Social Network Analysis Report",
                                    format: str = "markdown",
                                    include_visualizations: bool = True,
                                    save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive analysis report combining all components.
        
        Args:
            graph: NetworkX graph that was analyzed
            analysis_results: Dictionary containing all analysis results
            report_title: Title for the report
            format: Output format ('markdown', 'html', 'text')
            include_visualizations: Whether to include visualization references
            save_path: Path to save the figure (optional)
            
        Returns:
            Path to generated report file
        """
        self.logger.info(f"Generating comprehensive report in {format} format...")
        
        # Update metadata
        self._update_metadata(graph, analysis_results)
        
        # Generate report content
        if format.lower() == 'markdown':
            content = self._generate_markdown_report(graph, analysis_results, report_title, include_visualizations)
            filename = f"{self._sanitize_filename(report_title)}.md"
        elif format.lower() == 'html':
            content = self._generate_html_report(graph, analysis_results, report_title, include_visualizations)
            filename = f"{self._sanitize_filename(report_title)}.html"
        else:
            content = self._generate_text_report(graph, analysis_results, report_title, include_visualizations)
            filename = f"{self._sanitize_filename(report_title)}.txt"
        
        # Save report
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Comprehensive report saved to {report_path}")
        return str(report_path)
    
    def generate_component_report(self,
                                component_name: str,
                                component_results: Dict[str, Any],
                                graph: Optional[nx.Graph] = None,
                                format: str = "markdown") -> str:
        """
        Generate focused report for a specific analysis component.
        
        Args:
            component_name: Name of the analysis component
            component_results: Results from the specific component
            graph: Optional NetworkX graph for context
            format: Output format ('markdown', 'html', 'text')
            
        Returns:
            Path to generated component report
        """
        self.logger.info(f"Generating {component_name} component report...")
        
        if format.lower() == 'markdown':
            content = self._generate_component_markdown(component_name, component_results, graph)
            filename = f"{component_name}_analysis.md"
        elif format.lower() == 'html':
            content = self._generate_component_html(component_name, component_results, graph)
            filename = f"{component_name}_analysis.html"
        else:
            content = self._generate_component_text(component_name, component_results, graph)
            filename = f"{component_name}_analysis.txt"
        
        # Save component report
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Component report saved to {report_path}")
        return str(report_path)
    
    def generate_executive_summary(self,
                                 graph: nx.Graph,
                                 analysis_results: Dict[str, Any],
                                 format: str = "markdown") -> str:
        """
        Generate executive summary of key findings.
        
        Args:
            graph: NetworkX graph that was analyzed
            analysis_results: Dictionary containing all analysis results
            format: Output format ('markdown', 'html', 'text')
            
        Returns:
            Path to generated executive summary
        """
        self.logger.info("Generating executive summary...")
        
        # Extract key insights
        key_insights = self.interpreter.extract_key_insights(graph, analysis_results)
        
        if format.lower() == 'markdown':
            content = self._generate_executive_summary_markdown(graph, analysis_results, key_insights)
            filename = "executive_summary.md"
        elif format.lower() == 'html':
            content = self._generate_executive_summary_html(graph, analysis_results, key_insights)
            filename = "executive_summary.html"
        else:
            content = self._generate_executive_summary_text(graph, analysis_results, key_insights)
            filename = "executive_summary.txt"
        
        # Save summary
        summary_path = self.output_dir / filename
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Executive summary saved to {summary_path}")
        return str(summary_path)
    
    def _generate_markdown_report(self, graph: nx.Graph, analysis_results: Dict[str, Any], 
                                title: str, include_visualizations: bool) -> str:
        """Generate comprehensive report in Markdown format."""
        content = []
        
        # Header
        content.append(f"# {title}")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Table of Contents
        content.append("## Table of Contents")
        content.append("1. [Executive Summary](#executive-summary)")
        content.append("2. [Network Overview](#network-overview)")
        if 'centrality' in analysis_results:
            content.append("3. [Centrality Analysis](#centrality-analysis)")
        if 'communities' in analysis_results:
            content.append("4. [Community Detection](#community-detection)")
        if 'influence_propagation' in analysis_results:
            content.append("5. [Influence Propagation](#influence-propagation)")
        if 'link_prediction' in analysis_results:
            content.append("6. [Link Prediction](#link-prediction)")
        content.append("7. [Key Insights](#key-insights)")
        content.append("8. [Methodology](#methodology)")
        content.append("")
        
        # Executive Summary
        content.append("## Executive Summary")
        key_insights = self.interpreter.extract_key_insights(graph, analysis_results)
        content.append(key_insights.get('summary', 'Analysis completed successfully.'))
        content.append("")
        
        # Network Overview
        content.append("## Network Overview")
        content.append(f"- **Nodes:** {graph.number_of_nodes():,}")
        content.append(f"- **Edges:** {graph.number_of_edges():,}")
        content.append(f"- **Density:** {nx.density(graph):.4f}")
        content.append(f"- **Connected:** {'Yes' if nx.is_connected(graph) else 'No'}")
        if not nx.is_connected(graph):
            content.append(f"- **Connected Components:** {nx.number_connected_components(graph)}")
        content.append("")
        
        # Component-specific sections
        if 'centrality' in analysis_results:
            content.extend(self._generate_centrality_markdown_section(analysis_results['centrality']))
        
        if 'communities' in analysis_results:
            content.extend(self._generate_community_markdown_section(analysis_results['communities']))
        
        if 'influence_propagation' in analysis_results:
            content.extend(self._generate_influence_markdown_section(analysis_results['influence_propagation']))
        
        if 'link_prediction' in analysis_results:
            content.extend(self._generate_link_prediction_markdown_section(analysis_results['link_prediction']))
        
        # Key Insights
        content.append("## Key Insights")
        for insight in key_insights.get('insights', []):
            content.append(f"- {insight}")
        content.append("")
        
        # Methodology
        content.append("## Methodology")
        content.append("This analysis employed the following methods:")
        content.append("")
        
        if 'centrality' in analysis_results:
            content.append("### Centrality Analysis")
            content.append("- **Degree Centrality:** Measures the number of direct connections")
            content.append("- **Betweenness Centrality:** Measures how often a node lies on shortest paths")
            content.append("- **Closeness Centrality:** Measures how close a node is to all other nodes")
            content.append("- **Clustering Coefficient:** Measures local clustering around a node")
            content.append("")
        
        if 'communities' in analysis_results:
            content.append("### Community Detection")
            method = analysis_results['communities'].get('method', 'Unknown')
            content.append(f"- **Algorithm:** {method.title()}")
            content.append("- **Modularity:** Quality measure of community partition")
            content.append("")
        
        if 'influence_propagation' in analysis_results:
            content.append("### Influence Propagation")
            model = analysis_results['influence_propagation'].get('model', 'Unknown')
            content.append(f"- **Model:** {model.replace('_', ' ').title()}")
            content.append("- **Seed Selection:** Based on centrality metrics")
            content.append("")
        
        if 'link_prediction' in analysis_results:
            content.append("### Link Prediction")
            content.append("- **Graph Neural Networks:** GraphSAGE and GNN models")
            content.append("- **Traditional ML:** Random Forest and SVM baselines")
            content.append("- **Evaluation:** Cross-validation with multiple metrics")
            content.append("")
        
        return "\n".join(content)
    
    def _generate_executive_summary_markdown(self, graph: nx.Graph, analysis_results: Dict[str, Any], 
                                           key_insights: Dict[str, Any]) -> str:
        """Generate executive summary in Markdown format."""
        content = []
        
        # Header
        content.append("# Executive Summary")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Overview
        content.append("## Network Overview")
        content.append(f"This analysis examined a social network containing **{graph.number_of_nodes():,} nodes** "
                      f"and **{graph.number_of_edges():,} edges** with a density of {nx.density(graph):.4f}.")
        content.append("")
        
        # Key findings
        content.append("## Key Findings")
        summary = key_insights.get('summary', 'Analysis completed successfully.')
        content.append(summary)
        content.append("")
        
        # Insights
        if key_insights.get('insights'):
            content.append("## Major Insights")
            for insight in key_insights['insights'][:5]:  # Top 5 insights
                content.append(f"- {insight}")
            content.append("")
        
        # Recommendations
        if key_insights.get('recommendations'):
            content.append("## Recommendations")
            for recommendation in key_insights['recommendations'][:3]:  # Top 3 recommendations
                content.append(f"- {recommendation}")
            content.append("")
        
        # Analysis components
        content.append("## Analysis Components")
        components = list(analysis_results.keys())
        content.append(f"This comprehensive analysis covered **{len(components)} major components**:")
        for component in components:
            content.append(f"- **{component.replace('_', ' ').title()}**")
        content.append("")
        
        return "\n".join(content)
    
    def _generate_executive_summary_html(self, graph: nx.Graph, analysis_results: Dict[str, Any], 
                                       key_insights: Dict[str, Any]) -> str:
        """Generate executive summary in HTML format."""
        markdown_content = self._generate_executive_summary_markdown(graph, analysis_results, key_insights)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Summary - Social Network Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2 {{ color: #333; }}
        .summary {{ background-color: #f9f9f9; padding: 20px; border-left: 4px solid #007acc; }}
        .insight {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        ul {{ padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="content">
        {self._markdown_to_html(markdown_content)}
    </div>
</body>
</html>
"""
        return html_content
    
    def _generate_executive_summary_text(self, graph: nx.Graph, analysis_results: Dict[str, Any], 
                                       key_insights: Dict[str, Any]) -> str:
        """Generate executive summary in plain text format."""
        markdown_content = self._generate_executive_summary_markdown(graph, analysis_results, key_insights)
        
        # Simple markdown to text conversion
        text_content = markdown_content.replace('#', '').replace('**', '').replace('*', '')
        return text_content
    
    def _generate_component_markdown(self, component_name: str, component_results: Dict[str, Any], 
                                   graph: Optional[nx.Graph] = None) -> str:
        """Generate component-specific report in Markdown format."""
        content = []
        
        # Header
        content.append(f"# {component_name.replace('_', ' ').title()} Analysis Report")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Component-specific content
        if component_name == 'centrality':
            content.extend(self._generate_centrality_markdown_section(component_results))
        elif component_name == 'communities':
            content.extend(self._generate_community_markdown_section(component_results))
        elif component_name == 'influence_propagation':
            content.extend(self._generate_influence_markdown_section(component_results))
        elif component_name == 'link_prediction':
            content.extend(self._generate_link_prediction_markdown_section(component_results))
        else:
            content.append(f"## {component_name.replace('_', ' ').title()} Results")
            content.append("Analysis results are available in the comprehensive report.")
            content.append("")
        
        # Interpretation
        interpretation = self.interpreter.interpret_centrality_results(component_results) if component_name == 'centrality' else \
                        self.interpreter.interpret_community_results(component_results) if component_name == 'communities' else \
                        self.interpreter.interpret_influence_results(component_results) if component_name == 'influence_propagation' else \
                        self.interpreter.interpret_link_prediction_results(component_results) if component_name == 'link_prediction' else \
                        "Component analysis completed successfully."
        
        content.append("## Interpretation")
        content.append(interpretation)
        content.append("")
        
        return "\n".join(content)
    
    def _generate_component_html(self, component_name: str, component_results: Dict[str, Any], 
                               graph: Optional[nx.Graph] = None) -> str:
        """Generate component-specific report in HTML format."""
        markdown_content = self._generate_component_markdown(component_name, component_results, graph)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{component_name.replace('_', ' ').title()} Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f9f9f9; padding: 20px; border-left: 4px solid #007acc; }}
    </style>
</head>
<body>
    <div class="content">
        {self._markdown_to_html(markdown_content)}
    </div>
</body>
</html>
"""
        return html_content
    
    def _generate_component_text(self, component_name: str, component_results: Dict[str, Any], 
                               graph: Optional[nx.Graph] = None) -> str:
        """Generate component-specific report in plain text format."""
        markdown_content = self._generate_component_markdown(component_name, component_results, graph)
        
        # Simple markdown to text conversion
        text_content = markdown_content.replace('#', '').replace('**', '').replace('*', '')
        text_content = text_content.replace('|', ' | ')  # Keep table formatting readable
        
        return text_content
    
    def _generate_centrality_markdown_section(self, centrality_results: Dict[str, Any]) -> List[str]:
        """Generate centrality analysis section in Markdown."""
        content = []
        content.append("## Centrality Analysis")
        content.append("")
        
        # Summary statistics
        if 'summary_statistics' in centrality_results:
            content.append("### Summary Statistics")
            content.append("| Metric | Mean | Std | Min | Max |")
            content.append("|--------|------|-----|-----|-----|")
            
            for metric, stats in centrality_results['summary_statistics'].items():
                content.append(f"| {metric.replace('_', ' ').title()} | "
                             f"{stats['mean']:.4f} | {stats['std']:.4f} | "
                             f"{stats['min']:.4f} | {stats['max']:.4f} |")
            content.append("")
        
        # Top nodes
        if 'top_nodes' in centrality_results:
            content.append("### Most Central Nodes")
            for metric, top_nodes in centrality_results['top_nodes'].items():
                content.append(f"#### By {metric.replace('_', ' ').title()}")
                for i, (node, score) in enumerate(top_nodes[:5], 1):
                    content.append(f"{i}. **{node}**: {score:.4f}")
                content.append("")
        
        # Correlations
        if 'centrality_correlations' in centrality_results:
            content.append("### Centrality Correlations")
            content.append("Spearman correlation coefficients between centrality measures:")
            content.append("")
            for comparison, correlation in centrality_results['centrality_correlations'].items():
                if correlation is not None:
                    content.append(f"- **{comparison.replace('_vs_', ' vs ').title()}**: {correlation:.3f}")
            content.append("")
        
        # Interpretation
        interpretation = self.interpreter.interpret_centrality_results(centrality_results)
        content.append("### Interpretation")
        content.append(interpretation)
        content.append("")
        
        return content
    
    def _generate_community_markdown_section(self, community_results: Dict[str, Any]) -> List[str]:
        """Generate community detection section in Markdown."""
        content = []
        content.append("## Community Detection")
        content.append("")
        
        # Overview
        if 'summary' in community_results:
            summary = community_results['summary']
            content.append("### Overview")
            content.append(f"- **Communities Detected:** {summary.get('total_communities', 0)}")
            content.append(f"- **Coverage:** {summary.get('coverage', 0):.1%} of nodes")
            content.append(f"- **Average Size:** {summary.get('size_statistics', {}).get('mean_size', 0):.1f} nodes")
            content.append(f"- **Modularity:** {community_results.get('modularity', 'N/A')}")
            content.append("")
        
        # Community details - use analyzed communities if available, otherwise raw communities
        communities_data = None
        if 'communities' in community_results and isinstance(community_results['communities'], dict):
            # Check if this is analyzed community data
            first_value = next(iter(community_results['communities'].values()), None)
            if isinstance(first_value, dict) and 'size' in first_value:
                communities_data = community_results['communities']
        
        # Fall back to raw communities if analyzed data not available
        if communities_data is None and 'raw_communities' in community_results:
            communities_data = community_results['raw_communities']
        elif communities_data is None and 'communities' in community_results:
            communities_data = community_results['communities']
        
        if communities_data:
            content.append("### Community Details")
            
            # Check if this is analyzed community data or raw community data
            first_value = next(iter(communities_data.values()), None)
            
            if isinstance(first_value, dict) and 'size' in first_value:
                # This is analyzed community data
                sorted_communities = sorted(communities_data.items(), 
                                          key=lambda x: x[1].get('size', 0), reverse=True)
                
                for community_id, community_data in sorted_communities[:10]:  # Top 10
                    content.append(f"#### Community {community_id}")
                    content.append(f"- **Size:** {community_data.get('size', 0)} nodes")
                    content.append(f"- **Density:** {community_data.get('density', 0):.3f}")
                    
                    # Key members
                    if 'key_members' in community_data and 'by_local_degree' in community_data['key_members']:
                        top_members = community_data['key_members']['by_local_degree'][:3]
                        member_list = [f"{node} ({degree})" for node, degree in top_members]
                        content.append(f"- **Key Members:** {', '.join(member_list)}")
                    content.append("")
            else:
                # This is raw community data (community_id -> list of nodes)
                sorted_communities = sorted(communities_data.items(), 
                                          key=lambda x: len(x[1]), reverse=True)
                
                for community_id, nodes in sorted_communities[:10]:  # Top 10
                    content.append(f"#### Community {community_id}")
                    content.append(f"- **Size:** {len(nodes)} nodes")
                    content.append(f"- **Members:** {', '.join(map(str, nodes[:5]))}")
                    if len(nodes) > 5:
                        content.append(f"  (and {len(nodes) - 5} more)")
                    content.append("")
        
        # Interpretation
        interpretation = self.interpreter.interpret_community_results(community_results)
        content.append("### Interpretation")
        content.append(interpretation)
        content.append("")
        
        return content
    
    def _generate_influence_markdown_section(self, influence_results: Dict[str, Any]) -> List[str]:
        """Generate influence propagation section in Markdown."""
        content = []
        content.append("## Influence Propagation")
        content.append("")
        
        # Model parameters
        if 'analysis_parameters' in influence_results:
            params = influence_results['analysis_parameters']
            content.append("### Model Configuration")
            content.append(f"- **Propagation Model:** {params.get('propagation_model', 'Unknown').replace('_', ' ').title()}")
            content.append(f"- **Number of Seeds:** {params.get('num_seeds', 'Unknown')}")
            content.append(f"- **Simulations:** {params.get('num_simulations', 'Unknown')}")
            content.append("")
        
        # Strategy comparison
        if 'strategy_comparison' in influence_results:
            content.append("### Seed Selection Strategy Comparison")
            content.append("| Strategy | Mean Influence | Std Influence |")
            content.append("|----------|----------------|---------------|")
            
            for strategy, results in influence_results['strategy_comparison'].items():
                for num_seeds, metrics in results.items():
                    content.append(f"| {strategy.replace('_', ' ').title()} | "
                                 f"{metrics['mean_influence']:.4f} | "
                                 f"{metrics['std_influence']:.4f} |")
                    break  # Only show first entry for brevity
            content.append("")
        
        # Best strategy results
        if 'best_strategy_results' in influence_results:
            best_results = influence_results['best_strategy_results']
            content.append("### Best Strategy Results")
            content.append(f"- **Strategy:** {best_results.get('strategy', 'Unknown').replace('_', ' ').title()}")
            content.append(f"- **Final Influence:** {best_results.get('final_influence', 0)} nodes")
            content.append(f"- **Influence Ratio:** {best_results.get('influence_ratio', 0):.1%}")
            content.append(f"- **Steps Taken:** {best_results.get('steps_taken', 0)}")
            content.append("")
        
        # Interpretation
        interpretation = self.interpreter.interpret_influence_results(influence_results)
        content.append("### Interpretation")
        content.append(interpretation)
        content.append("")
        
        return content
    
    def _generate_link_prediction_markdown_section(self, link_prediction_results: Dict[str, Any]) -> List[str]:
        """Generate link prediction section in Markdown."""
        content = []
        content.append("## Link Prediction")
        content.append("")
        
        # Model comparison
        if 'model_comparison' in link_prediction_results:
            content.append("### Model Performance Comparison")
            content.append("| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |")
            content.append("|-------|----------|-----------|--------|----------|---------|")
            
            comparison = link_prediction_results['model_comparison']
            if 'performance_comparison' in comparison and 'raw_data' in comparison['performance_comparison']:
                for model, metrics in comparison['performance_comparison']['raw_data'].items():
                    content.append(f"| {model} | "
                                 f"{metrics.get('accuracy', 0):.4f} | "
                                 f"{metrics.get('precision', 0):.4f} | "
                                 f"{metrics.get('recall', 0):.4f} | "
                                 f"{metrics.get('f1_score', 0):.4f} | "
                                 f"{metrics.get('auc_roc', 0):.4f} |")
            content.append("")
        
        # Best model
        if 'best_model' in link_prediction_results:
            best_model = link_prediction_results['best_model']
            content.append("### Best Performing Model")
            content.append(f"- **Model:** {best_model.get('name', 'Unknown')}")
            content.append(f"- **F1-Score:** {best_model.get('f1_score', 0):.4f}")
            content.append(f"- **AUC-ROC:** {best_model.get('auc_roc', 0):.4f}")
            content.append("")
        
        # Feature importance
        if 'feature_importance' in link_prediction_results:
            feature_importance = link_prediction_results['feature_importance']
            if 'top_features' in feature_importance:
                content.append("### Most Important Features")
                top_features = feature_importance['top_features']
                for i, (feature, importance) in enumerate(zip(
                    top_features.get('features', [])[:5],
                    top_features.get('importance_scores', [])[:5]
                ), 1):
                    content.append(f"{i}. **{feature}**: {importance:.4f}")
                content.append("")
        
        # Statistical significance
        if 'statistical_significance' in link_prediction_results:
            sig_results = link_prediction_results['statistical_significance']
            if 'summary' in sig_results:
                content.append("### Statistical Significance")
                content.append(f"- **Significant Differences Found:** {'Yes' if sig_results['summary'].get('significant_differences_found', False) else 'No'}")
                content.append("")
        
        # Interpretation
        interpretation = self.interpreter.interpret_link_prediction_results(link_prediction_results)
        content.append("### Interpretation")
        content.append(interpretation)
        content.append("")
        
        return content
    
    def _generate_html_report(self, graph: nx.Graph, analysis_results: Dict[str, Any], 
                            title: str, include_visualizations: bool) -> str:
        """Generate comprehensive report in HTML format."""
        # Convert markdown to HTML (simplified version)
        markdown_content = self._generate_markdown_report(graph, analysis_results, title, include_visualizations)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f9f9f9; padding: 20px; border-left: 4px solid #007acc; }}
        .insight {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="content">
        {self._markdown_to_html(markdown_content)}
    </div>
</body>
</html>
"""
        return html_content
    
    def _generate_text_report(self, graph: nx.Graph, analysis_results: Dict[str, Any], 
                            title: str, include_visualizations: bool) -> str:
        """Generate comprehensive report in plain text format."""
        # Convert markdown to plain text
        markdown_content = self._generate_markdown_report(graph, analysis_results, title, include_visualizations)
        
        # Simple markdown to text conversion
        text_content = markdown_content.replace('#', '').replace('**', '').replace('*', '')
        text_content = text_content.replace('|', ' | ')  # Keep table formatting readable
        
        return text_content
    
    def _update_metadata(self, graph: nx.Graph, analysis_results: Dict[str, Any]):
        """Update report metadata."""
        self.report_metadata.update({
            'generated_at': datetime.now().isoformat(),
            'analysis_components': list(analysis_results.keys()),
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'analysis_parameters': {
                component: results.get('parameters', {}) if isinstance(results, dict) else {}
                for component, results in analysis_results.items()
            }
        })
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        import re
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = filename.replace(' ', '_').lower()
        return filename
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Simple markdown to HTML conversion."""
        html = markdown_text
        
        # Headers
        html = html.replace('### ', '<h3>').replace('\n', '</h3>\n', 1) if '### ' in html else html
        html = html.replace('## ', '<h2>').replace('\n', '</h2>\n', 1) if '## ' in html else html  
        html = html.replace('# ', '<h1>').replace('\n', '</h1>\n', 1) if '# ' in html else html
        
        # Bold text
        html = html.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
        
        # Lists
        lines = html.split('\n')
        in_list = False
        result_lines = []
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    result_lines.append('<ul>')
                    in_list = True
                result_lines.append(f'<li>{line.strip()[2:]}</li>')
            else:
                if in_list:
                    result_lines.append('</ul>')
                    in_list = False
                result_lines.append(line)
        
        if in_list:
            result_lines.append('</ul>')
        
        # Paragraphs
        html = '\n'.join(result_lines)
        html = html.replace('\n\n', '</p>\n<p>')
        html = f'<p>{html}</p>'
        
        return html
    
    def save_metadata(self, filename: str = "report_metadata.json"):
        """Save report metadata to JSON file."""
        metadata_path = self.output_dir / filename
        with open(metadata_path, 'w') as f:
            json.dump(self.report_metadata, f, indent=2)
        
        self.logger.info(f"Report metadata saved to {metadata_path}")
        return str(metadata_path)
    
    def create_report_index(self, reports: List[str]) -> str:
        """Create an index file linking to all generated reports."""
        index_content = []
        index_content.append("# Social Network Analysis Reports Index")
        index_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        index_content.append("")
        
        index_content.append("## Available Reports")
        for report_path in reports:
            report_name = Path(report_path).stem.replace('_', ' ').title()
            relative_path = Path(report_path).name
            index_content.append(f"- [{report_name}]({relative_path})")
        
        index_content.append("")
        index_content.append("## Analysis Summary")
        index_content.append(f"- **Total Nodes:** {self.report_metadata.get('total_nodes', 0):,}")
        index_content.append(f"- **Total Edges:** {self.report_metadata.get('total_edges', 0):,}")
        index_content.append(f"- **Components Analyzed:** {', '.join(self.report_metadata.get('analysis_components', []))}")
        
        # Save index
        index_path = self.output_dir / "index.md"
        with open(index_path, 'w') as f:
            f.write('\n'.join(index_content))
        
        self.logger.info(f"Report index created at {index_path}")
        return str(index_path)