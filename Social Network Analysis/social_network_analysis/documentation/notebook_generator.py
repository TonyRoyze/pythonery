"""
NotebookGenerator class for creating Jupyter notebook templates.

This module provides functionality to generate reproducible Jupyter notebook
templates for social network analysis workflows, including data loading,
analysis execution, and result visualization.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging


class NotebookGenerator:
    """
    Generates Jupyter notebook templates for reproducible social network analysis.
    
    Creates structured notebooks with code cells, markdown documentation,
    and visualization templates for different analysis workflows.
    """
    
    def __init__(self, output_dir: str = "analysis_notebooks"):
        """
        Initialize NotebookGenerator.
        
        Args:
            output_dir: Directory to save generated notebooks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Notebook metadata
        self.notebook_metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        }
    
    def generate_comprehensive_analysis_notebook(self, 
                                               data_files: Dict[str, str],
                                               analysis_components: List[str] = None) -> str:
        """
        Generate comprehensive analysis notebook template.
        
        Args:
            data_files: Dictionary mapping data types to file paths
            analysis_components: List of analysis components to include
            
        Returns:
            Path to generated notebook file
        """
        if analysis_components is None:
            analysis_components = ['centrality', 'communities', 'influence_propagation', 'link_prediction']
        
        self.logger.info("Generating comprehensive analysis notebook...")
        
        cells = []
        
        # Title and introduction
        cells.extend(self._create_introduction_cells())
        
        # Setup and imports
        cells.extend(self._create_setup_cells())
        
        # Data loading
        cells.extend(self._create_data_loading_cells(data_files))
        
        # Graph construction
        cells.extend(self._create_graph_construction_cells())
        
        # Analysis components
        for component in analysis_components:
            if component == 'centrality':
                cells.extend(self._create_centrality_analysis_cells())
            elif component == 'communities':
                cells.extend(self._create_community_analysis_cells())
            elif component == 'influence_propagation':
                cells.extend(self._create_influence_analysis_cells())
            elif component == 'link_prediction':
                cells.extend(self._create_link_prediction_cells())
        
        # Results compilation and documentation
        cells.extend(self._create_results_compilation_cells())
        
        # Create notebook
        notebook = {
            "cells": cells,
            "metadata": self.notebook_metadata,
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook
        notebook_path = self.output_dir / "comprehensive_social_network_analysis.ipynb"
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        self.logger.info(f"Comprehensive analysis notebook saved to {notebook_path}")
        return str(notebook_path)
    
    def generate_component_notebook(self, 
                                  component_name: str,
                                  data_files: Dict[str, str]) -> str:
        """
        Generate focused notebook for specific analysis component.
        
        Args:
            component_name: Name of the analysis component
            data_files: Dictionary mapping data types to file paths
            
        Returns:
            Path to generated notebook file
        """
        self.logger.info(f"Generating {component_name} analysis notebook...")
        
        cells = []
        
        # Component-specific introduction
        cells.extend(self._create_component_introduction_cells(component_name))
        
        # Setup and imports
        cells.extend(self._create_setup_cells())
        
        # Data loading
        cells.extend(self._create_data_loading_cells(data_files))
        
        # Graph construction
        cells.extend(self._create_graph_construction_cells())
        
        # Component-specific analysis
        if component_name == 'centrality':
            cells.extend(self._create_centrality_analysis_cells())
        elif component_name == 'communities':
            cells.extend(self._create_community_analysis_cells())
        elif component_name == 'influence_propagation':
            cells.extend(self._create_influence_analysis_cells())
        elif component_name == 'link_prediction':
            cells.extend(self._create_link_prediction_cells())
        
        # Component-specific results
        cells.extend(self._create_component_results_cells(component_name))
        
        # Create notebook
        notebook = {
            "cells": cells,
            "metadata": self.notebook_metadata,
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook
        notebook_path = self.output_dir / f"{component_name}_analysis.ipynb"
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        self.logger.info(f"{component_name} analysis notebook saved to {notebook_path}")
        return str(notebook_path)
    
    def generate_visualization_notebook(self, 
                                      analysis_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate notebook focused on visualization and interpretation.
        
        Args:
            analysis_results: Optional pre-computed analysis results
            
        Returns:
            Path to generated visualization notebook
        """
        self.logger.info("Generating visualization notebook...")
        
        cells = []
        
        # Visualization introduction
        cells.extend(self._create_visualization_introduction_cells())
        
        # Setup for visualization
        cells.extend(self._create_visualization_setup_cells())
        
        # Load or compute results
        if analysis_results:
            cells.extend(self._create_results_loading_cells())
        else:
            cells.extend(self._create_quick_analysis_cells())
        
        # Visualization sections
        cells.extend(self._create_network_visualization_cells())
        cells.extend(self._create_centrality_visualization_cells())
        cells.extend(self._create_community_visualization_cells())
        cells.extend(self._create_interactive_visualization_cells())
        
        # Create notebook
        notebook = {
            "cells": cells,
            "metadata": self.notebook_metadata,
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook
        notebook_path = self.output_dir / "network_visualization.ipynb"
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        self.logger.info(f"Visualization notebook saved to {notebook_path}")
        return str(notebook_path)
    
    def _create_introduction_cells(self) -> List[Dict]:
        """Create introduction cells for comprehensive analysis."""
        return [
            self._create_markdown_cell(
                "# Comprehensive Social Network Analysis\n\n"
                "This notebook provides a complete analysis workflow for social network data, "
                "including centrality analysis, community detection, influence propagation modeling, "
                "and link prediction.\n\n"
                "## Analysis Components\n"
                "1. **Data Loading and Preprocessing**\n"
                "2. **Graph Construction**\n"
                "3. **Centrality Analysis**\n"
                "4. **Community Detection**\n"
                "5. **Influence Propagation**\n"
                "6. **Link Prediction**\n"
                "7. **Results Documentation**\n\n"
                "## Requirements\n"
                "- NetworkX for graph analysis\n"
                "- PyTorch Geometric for graph neural networks\n"
                "- Scikit-learn for traditional ML models\n"
                "- Matplotlib and Plotly for visualization"
            )
        ]
    
    def _create_component_introduction_cells(self, component_name: str) -> List[Dict]:
        """Create introduction cells for component-specific analysis."""
        component_descriptions = {
            'centrality': "This notebook focuses on centrality analysis, measuring the importance "
                         "and influence of nodes within the social network using various centrality metrics.",
            'communities': "This notebook focuses on community detection, identifying groups of "
                          "closely connected nodes and analyzing their characteristics.",
            'influence_propagation': "This notebook focuses on influence propagation modeling, "
                                   "simulating how information spreads through the network.",
            'link_prediction': "This notebook focuses on link prediction, using machine learning "
                              "models to predict future connections in the network."
        }
        
        description = component_descriptions.get(component_name, f"Analysis of {component_name} component.")
        
        return [
            self._create_markdown_cell(
                f"# {component_name.replace('_', ' ').title()} Analysis\n\n"
                f"{description}\n\n"
                "## Objectives\n"
                f"- Perform comprehensive {component_name.replace('_', ' ')} analysis\n"
                "- Generate interpretable results and visualizations\n"
                "- Document findings and insights\n"
                "- Create reproducible analysis workflow"
            )
        ]
    
    def _create_setup_cells(self) -> List[Dict]:
        """Create setup and import cells."""
        return [
            self._create_markdown_cell("## Setup and Imports"),
            self._create_code_cell(
                "# Import required libraries\n"
                "import numpy as np\n"
                "import pandas as pd\n"
                "import networkx as nx\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "import plotly.graph_objects as go\n"
                "import plotly.express as px\n"
                "from pathlib import Path\n"
                "import json\n"
                "import warnings\n"
                "warnings.filterwarnings('ignore')\n\n"
                "# Import social network analysis modules\n"
                "from social_network_analysis.data.data_loader import DataLoader\n"
                "from social_network_analysis.graph.graph_builder import GraphBuilder\n"
                "from social_network_analysis.analysis.centrality_calculator import CentralityCalculator\n"
                "from social_network_analysis.analysis.community_detector import CommunityDetector\n"
                "from social_network_analysis.analysis.community_analyzer import CommunityAnalyzer\n"
                "from social_network_analysis.analysis.influence_propagator import InfluencePropagator\n"
                "from social_network_analysis.ml.model_comparator import ModelComparator\n"
                "from social_network_analysis.visualization.network_visualizer import NetworkVisualizer\n"
                "from social_network_analysis.visualization.interactive_visualizer import InteractiveVisualizer\n"
                "from social_network_analysis.documentation.documentation_generator import DocumentationGenerator\n\n"
                "# Set up plotting\n"
                "plt.style.use('default')\n"
                "sns.set_palette('husl')\n"
                "%matplotlib inline\n\n"
                "print('Setup complete!')"
            )
        ]
    
    def _create_data_loading_cells(self, data_files: Dict[str, str]) -> List[Dict]:
        """Create data loading cells."""
        cells = [self._create_markdown_cell("## Data Loading")]
        
        # Data file paths
        file_paths_code = "# Define data file paths\ndata_files = {\n"
        for data_type, file_path in data_files.items():
            file_paths_code += f"    '{data_type}': '{file_path}',\n"
        file_paths_code += "}\n\nprint('Data file paths defined:')\nfor key, path in data_files.items():\n    print(f'  {key}: {path}')"
        
        cells.append(self._create_code_cell(file_paths_code))
        
        # Load data
        cells.append(self._create_code_cell(
            "# Initialize data loader\n"
            "data_loader = DataLoader()\n\n"
            "# Load data files\n"
            "print('Loading data files...')\n"
            "loaded_data = {}\n\n"
            "if 'comments' in data_files:\n"
            "    loaded_data['comments'] = data_loader.load_comments(data_files['comments'])\n"
            "    print(f'Loaded {len(loaded_data[\"comments\"])} comments')\n\n"
            "if 'relationships' in data_files:\n"
            "    loaded_data['relationships'] = data_loader.load_relationships(data_files['relationships'])\n"
            "    print(f'Loaded {len(loaded_data[\"relationships\"])} relationships')\n\n"
            "if 'reply_counts' in data_files:\n"
            "    loaded_data['reply_counts'] = data_loader.load_reply_counts(data_files['reply_counts'])\n"
            "    print(f'Loaded reply counts for {len(loaded_data[\"reply_counts\"])} users')\n\n"
            "print('Data loading complete!')"
        ))
        
        return cells
    
    def _create_graph_construction_cells(self) -> List[Dict]:
        """Create graph construction cells."""
        return [
            self._create_markdown_cell("## Graph Construction"),
            self._create_code_cell(
                "# Initialize graph builder\n"
                "graph_builder = GraphBuilder()\n\n"
                "# Build graph from relationship data\n"
                "print('Building network graph...')\n"
                "graph = graph_builder.build_graph(\n"
                "    relationships=loaded_data.get('relationships'),\n"
                "    comments=loaded_data.get('comments')\n"
                ")\n\n"
                "# Add node attributes if available\n"
                "if 'reply_counts' in loaded_data:\n"
                "    graph = graph_builder.add_node_attributes(\n"
                "        graph, \n"
                "        loaded_data['reply_counts']\n"
                "    )\n\n"
                "# Display graph statistics\n"
                "print(f'Graph constructed successfully!')\n"
                "print(f'  Nodes: {graph.number_of_nodes():,}')\n"
                "print(f'  Edges: {graph.number_of_edges():,}')\n"
                "print(f'  Density: {nx.density(graph):.4f}')\n"
                "print(f'  Connected: {nx.is_connected(graph)}')\n"
                "if not nx.is_connected(graph):\n"
                "    print(f'  Connected components: {nx.number_connected_components(graph)}')"
            )
        ]
    
    def _create_centrality_analysis_cells(self) -> List[Dict]:
        """Create centrality analysis cells."""
        return [
            self._create_markdown_cell("## Centrality Analysis"),
            self._create_code_cell(
                "# Initialize centrality calculator\n"
                "centrality_calc = CentralityCalculator()\n\n"
                "# Calculate all centrality metrics\n"
                "print('Calculating centrality metrics...')\n"
                "centrality_results = centrality_calc.generate_centrality_report(graph)\n\n"
                "print('Centrality analysis complete!')\n"
                "print(f'Analyzed {len(centrality_results[\"centrality_scores\"][\"degree\"])} nodes')"
            ),
            self._create_code_cell(
                "# Display top nodes by different centrality measures\n"
                "top_nodes = centrality_results['top_nodes']\n\n"
                "for metric, nodes in top_nodes.items():\n"
                "    print(f'\\nTop 5 nodes by {metric.replace(\"_\", \" \").title()}:')\n"
                "    for i, (node, score) in enumerate(nodes[:5], 1):\n"
                "        print(f'  {i}. {node}: {score:.4f}')"
            ),
            self._create_code_cell(
                "# Visualize centrality distributions\n"
                "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n"
                "axes = axes.flatten()\n\n"
                "centrality_scores = centrality_results['centrality_scores']\n"
                "metrics = ['degree', 'betweenness', 'closeness', 'clustering']\n\n"
                "for i, metric in enumerate(metrics):\n"
                "    if metric in centrality_scores:\n"
                "        values = list(centrality_scores[metric].values())\n"
                "        axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')\n"
                "        axes[i].set_title(f'{metric.replace(\"_\", \" \").title()} Distribution')\n"
                "        axes[i].set_xlabel('Centrality Score')\n"
                "        axes[i].set_ylabel('Frequency')\n"
                "        axes[i].grid(True, alpha=0.3)\n\n"
                "plt.tight_layout()\n"
                "plt.show()"
            )
        ]
    
    def _create_community_analysis_cells(self) -> List[Dict]:
        """Create community analysis cells."""
        return [
            self._create_markdown_cell("## Community Detection and Analysis"),
            self._create_code_cell(
                "# Initialize community detector and analyzer\n"
                "community_detector = CommunityDetector()\n"
                "community_analyzer = CommunityAnalyzer()\n\n"
                "# Detect communities using Louvain algorithm\n"
                "print('Detecting communities...')\n"
                "louvain_communities = community_detector.detect_louvain_communities(graph)\n"
                "louvain_modularity = community_detector.calculate_modularity(graph, louvain_communities)\n\n"
                "print(f'Louvain algorithm detected {len(louvain_communities)} communities')\n"
                "print(f'Modularity score: {louvain_modularity:.4f}')"
            ),
            self._create_code_cell(
                "# Analyze community characteristics\n"
                "community_analysis = community_analyzer.analyze_community_characteristics(\n"
                "    graph, louvain_communities\n"
                ")\n\n"
                "# Display community summary\n"
                "summary = community_analysis['summary']\n"
                "print('Community Analysis Summary:')\n"
                "print(f'  Total communities: {summary[\"total_communities\"]}')\n"
                "print(f'  Coverage: {summary[\"coverage\"]:.1%}')\n"
                "print(f'  Average size: {summary[\"size_statistics\"][\"mean_size\"]:.1f} nodes')\n"
                "print(f'  Average density: {summary[\"density_statistics\"][\"mean_density\"]:.3f}')"
            ),
            self._create_code_cell(
                "# Visualize community size distribution\n"
                "community_sizes = [len(nodes) for nodes in louvain_communities.values()]\n\n"
                "plt.figure(figsize=(12, 5))\n\n"
                "plt.subplot(1, 2, 1)\n"
                "plt.hist(community_sizes, bins=20, alpha=0.7, edgecolor='black')\n"
                "plt.title('Community Size Distribution')\n"
                "plt.xlabel('Community Size (nodes)')\n"
                "plt.ylabel('Frequency')\n"
                "plt.grid(True, alpha=0.3)\n\n"
                "plt.subplot(1, 2, 2)\n"
                "plt.boxplot(community_sizes)\n"
                "plt.title('Community Size Box Plot')\n"
                "plt.ylabel('Community Size (nodes)')\n"
                "plt.grid(True, alpha=0.3)\n\n"
                "plt.tight_layout()\n"
                "plt.show()"
            )
        ]
    
    def _create_influence_analysis_cells(self) -> List[Dict]:
        """Create influence propagation analysis cells."""
        return [
            self._create_markdown_cell("## Influence Propagation Analysis"),
            self._create_code_cell(
                "# Initialize influence propagator\n"
                "influence_propagator = InfluencePropagator()\n\n"
                "# Calculate centrality scores for seed selection\n"
                "if 'centrality_results' not in locals():\n"
                "    centrality_calc = CentralityCalculator()\n"
                "    centrality_results = centrality_calc.generate_centrality_report(graph)\n\n"
                "centrality_scores = centrality_results['centrality_scores']\n"
                "print('Centrality scores available for seed selection')"
            ),
            self._create_code_cell(
                "# Generate comprehensive influence propagation report\n"
                "print('Generating influence propagation analysis...')\n"
                "influence_report = influence_propagator.generate_propagation_report(\n"
                "    graph=graph,\n"
                "    centrality_scores=centrality_scores,\n"
                "    num_seeds=5,\n"
                "    num_simulations=50\n"
                ")\n\n"
                "print('Influence propagation analysis complete!')\n"
                "print(f'Analyzed {influence_report[\"graph_info\"][\"nodes\"]} nodes')\n"
                "print(f'Tested {len(influence_report[\"strategy_comparison\"])} seed selection strategies')"
            ),
            self._create_code_cell(
                "# Display strategy comparison results\n"
                "strategy_comparison = influence_report['strategy_comparison']\n\n"
                "print('Seed Selection Strategy Comparison:')\n"
                "print('Strategy\\t\\tMean Influence\\tStd Influence')\n"
                "print('-' * 50)\n\n"
                "for strategy, results in strategy_comparison.items():\n"
                "    for num_seeds, metrics in results.items():\n"
                "        mean_inf = metrics['mean_influence']\n"
                "        std_inf = metrics['std_influence']\n"
                "        print(f'{strategy:<15}\\t{mean_inf:.4f}\\t\\t{std_inf:.4f}')\n"
                "        break  # Only show first entry"
            )
        ]
    
    def _create_link_prediction_cells(self) -> List[Dict]:
        """Create link prediction analysis cells."""
        return [
            self._create_markdown_cell("## Link Prediction Analysis"),
            self._create_code_cell(
                "# Import link prediction modules\n"
                "from social_network_analysis.ml.link_prediction_data import LinkPredictionData\n"
                "from social_network_analysis.ml.graphsage_predictor import GraphSAGEPredictor\n"
                "from social_network_analysis.ml.traditional_ml_predictor import TraditionalMLPredictor\n\n"
                "print('Link prediction modules imported')"
            ),
            self._create_code_cell(
                "# Prepare data for link prediction\n"
                "print('Preparing link prediction data...')\n"
                "link_data = LinkPredictionData()\n"
                "train_data, test_data = link_data.prepare_temporal_split(graph)\n\n"
                "print(f'Training edges: {len(train_data[\"edges\"])}')\n"
                "print(f'Test edges: {len(test_data[\"edges\"])}')\n"
                "print('Link prediction data prepared!')"
            ),
            self._create_code_cell(
                "# Initialize model comparator\n"
                "model_comparator = ModelComparator()\n\n"
                "# Train and evaluate GraphSAGE model\n"
                "print('Training GraphSAGE model...')\n"
                "graphsage_predictor = GraphSAGEPredictor()\n"
                "graphsage_results = graphsage_predictor.train_and_evaluate(\n"
                "    train_data, test_data\n"
                ")\n\n"
                "# Add results to comparator\n"
                "model_comparator.add_model_results(\n"
                "    model_name='GraphSAGE',\n"
                "    model_type='graph',\n"
                "    test_predictions=graphsage_results['predictions'],\n"
                "    test_probabilities=graphsage_results['probabilities'],\n"
                "    test_labels=graphsage_results['labels'],\n"
                "    training_time=graphsage_results.get('training_time')\n"
                ")\n\n"
                "print('GraphSAGE model evaluation complete!')"
            ),
            self._create_code_cell(
                "# Train and evaluate traditional ML models\n"
                "print('Training traditional ML models...')\n"
                "traditional_predictor = TraditionalMLPredictor()\n"
                "traditional_results = traditional_predictor.train_and_evaluate(\n"
                "    train_data, test_data\n"
                ")\n\n"
                "# Add traditional ML results\n"
                "for model_name, results in traditional_results.items():\n"
                "    model_comparator.add_model_results(\n"
                "        model_name=model_name,\n"
                "        model_type='traditional',\n"
                "        test_predictions=results['predictions'],\n"
                "        test_probabilities=results['probabilities'],\n"
                "        test_labels=results['labels'],\n"
                "        feature_importance=results.get('feature_importance', {})\n"
                "    )\n\n"
                "print('Traditional ML model evaluation complete!')"
            ),
            self._create_code_cell(
                "# Generate comprehensive model comparison\n"
                "print('Generating model comparison analysis...')\n"
                "comparison_results = model_comparator.generate_comprehensive_comparison()\n\n"
                "# Display performance comparison\n"
                "performance_data = comparison_results['performance_comparison']['raw_data']\n"
                "print('\\nModel Performance Comparison:')\n"
                "print('Model\\t\\tAccuracy\\tPrecision\\tRecall\\t\\tF1-Score\\tAUC-ROC')\n"
                "print('-' * 80)\n\n"
                "for model, metrics in performance_data.items():\n"
                "    print(f'{model:<15}\\t{metrics[\"accuracy\"]:.4f}\\t\\t{metrics[\"precision\"]:.4f}\\t\\t'\n"
                "          f'{metrics[\"recall\"]:.4f}\\t\\t{metrics[\"f1_score\"]:.4f}\\t\\t{metrics[\"auc_roc\"]:.4f}')"
            )
        ]
    
    def _create_results_compilation_cells(self) -> List[Dict]:
        """Create results compilation and documentation cells."""
        return [
            self._create_markdown_cell("## Results Compilation and Documentation"),
            self._create_code_cell(
                "# Compile all analysis results\n"
                "analysis_results = {}\n\n"
                "if 'centrality_results' in locals():\n"
                "    analysis_results['centrality'] = centrality_results\n"
                "    print('✓ Centrality results compiled')\n\n"
                "if 'community_analysis' in locals():\n"
                "    analysis_results['communities'] = {\n"
                "        **community_analysis,\n"
                "        'communities': louvain_communities,\n"
                "        'modularity': louvain_modularity,\n"
                "        'method': 'louvain'\n"
                "    }\n"
                "    print('✓ Community analysis results compiled')\n\n"
                "if 'influence_report' in locals():\n"
                "    analysis_results['influence_propagation'] = influence_report\n"
                "    print('✓ Influence propagation results compiled')\n\n"
                "if 'comparison_results' in locals():\n"
                "    analysis_results['link_prediction'] = {\n"
                "        'model_comparison': comparison_results,\n"
                "        'best_model': comparison_results['model_ranking'].get('overall_best', {})\n"
                "    }\n"
                "    print('✓ Link prediction results compiled')\n\n"
                "print(f'\\nTotal analysis components: {len(analysis_results)}')"
            ),
            self._create_code_cell(
                "# Generate comprehensive documentation\n"
                "doc_generator = DocumentationGenerator()\n\n"
                "# Generate comprehensive report\n"
                "print('Generating comprehensive analysis report...')\n"
                "report_path = doc_generator.generate_comprehensive_report(\n"
                "    graph=graph,\n"
                "    analysis_results=analysis_results,\n"
                "    report_title='Social Network Analysis Report',\n"
                "    format='markdown'\n"
                ")\n\n"
                "print(f'Comprehensive report saved to: {report_path}')\n\n"
                "# Generate executive summary\n"
                "summary_path = doc_generator.generate_executive_summary(\n"
                "    graph=graph,\n"
                "    analysis_results=analysis_results\n"
                ")\n\n"
                "print(f'Executive summary saved to: {summary_path}')"
            ),
            self._create_code_cell(
                "# Save analysis results for future use\n"
                "import pickle\n"
                "from datetime import datetime\n\n"
                "# Create results package\n"
                "results_package = {\n"
                "    'graph': graph,\n"
                "    'analysis_results': analysis_results,\n"
                "    'metadata': {\n"
                "        'generated_at': datetime.now().isoformat(),\n"
                "        'nodes': graph.number_of_nodes(),\n"
                "        'edges': graph.number_of_edges(),\n"
                "        'components': list(analysis_results.keys())\n"
                "    }\n"
                "}\n\n"
                "# Save results\n"
                "results_file = 'social_network_analysis_results.pkl'\n"
                "with open(results_file, 'wb') as f:\n"
                "    pickle.dump(results_package, f)\n\n"
                "print(f'Analysis results saved to: {results_file}')\n"
                "print('\\n🎉 Analysis complete! All results have been documented and saved.')"
            )
        ]
    
    def _create_visualization_introduction_cells(self) -> List[Dict]:
        """Create introduction cells for visualization notebook."""
        return [
            self._create_markdown_cell(
                "# Social Network Visualization and Interpretation\n\n"
                "This notebook focuses on creating comprehensive visualizations and "
                "interpretations of social network analysis results.\n\n"
                "## Visualization Components\n"
                "1. **Network Overview Visualizations**\n"
                "2. **Centrality-based Visualizations**\n"
                "3. **Community Structure Visualizations**\n"
                "4. **Interactive Dashboards**\n"
                "5. **Results Interpretation**"
            )
        ]
    
    def _create_visualization_setup_cells(self) -> List[Dict]:
        """Create setup cells for visualization notebook."""
        return [
            self._create_markdown_cell("## Setup and Imports"),
            self._create_code_cell(
                "# Import visualization libraries\n"
                "import numpy as np\n"
                "import pandas as pd\n"
                "import networkx as nx\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "import plotly.graph_objects as go\n"
                "import plotly.express as px\n"
                "from plotly.subplots import make_subplots\n"
                "import warnings\n"
                "warnings.filterwarnings('ignore')\n\n"
                "# Import analysis modules\n"
                "from social_network_analysis.visualization.network_visualizer import NetworkVisualizer\n"
                "from social_network_analysis.visualization.interactive_visualizer import InteractiveVisualizer\n"
                "from social_network_analysis.documentation.result_interpreter import ResultInterpreter\n\n"
                "# Set up plotting\n"
                "plt.style.use('default')\n"
                "sns.set_palette('husl')\n"
                "%matplotlib inline\n\n"
                "print('Visualization setup complete!')"
            )
        ]
    
    def _create_results_loading_cells(self) -> List[Dict]:
        """Create cells for loading pre-computed results."""
        return [
            self._create_markdown_cell("## Load Analysis Results"),
            self._create_code_cell(
                "# Load pre-computed analysis results\n"
                "import pickle\n\n"
                "results_file = 'social_network_analysis_results.pkl'\n"
                "with open(results_file, 'rb') as f:\n"
                "    results_package = pickle.load(f)\n\n"
                "graph = results_package['graph']\n"
                "analysis_results = results_package['analysis_results']\n"
                "metadata = results_package['metadata']\n\n"
                "print('Analysis results loaded successfully!')\n"
                "print(f'Graph: {metadata[\"nodes\"]} nodes, {metadata[\"edges\"]} edges')\n"
                "print(f'Components: {metadata[\"components\"]}')"
            )
        ]
    
    def _create_network_visualization_cells(self) -> List[Dict]:
        """Create network visualization cells."""
        return [
            self._create_markdown_cell("## Network Overview Visualizations"),
            self._create_code_cell(
                "# Initialize visualizers\n"
                "network_viz = NetworkVisualizer()\n"
                "interactive_viz = InteractiveVisualizer()\n\n"
                "# Create basic network visualization\n"
                "print('Creating network overview visualization...')\n"
                "network_viz.visualize_network(graph, layout='spring', figsize=(12, 10))\n"
                "plt.title('Social Network Overview')\n"
                "plt.show()"
            )
        ]
    
    def _create_centrality_visualization_cells(self) -> List[Dict]:
        """Create centrality visualization cells."""
        return [
            self._create_markdown_cell("## Centrality Visualizations"),
            self._create_code_cell(
                "# Visualize centrality metrics\n"
                "if 'centrality' in analysis_results:\n"
                "    centrality_scores = analysis_results['centrality']['centrality_scores']\n"
                "    \n"
                "    # Create centrality-based network visualization\n"
                "    network_viz.visualize_centrality(\n"
                "        graph, centrality_scores['degree'], \n"
                "        title='Network colored by Degree Centrality'\n"
                "    )\n"
                "    plt.show()\n"
                "    \n"
                "    print('Centrality visualizations created!')\n"
                "else:\n"
                "    print('Centrality results not available')"
            )
        ]
    
    def _create_community_visualization_cells(self) -> List[Dict]:
        """Create community visualization cells."""
        return [
            self._create_markdown_cell("## Community Visualizations"),
            self._create_code_cell(
                "# Visualize community structure\n"
                "if 'communities' in analysis_results:\n"
                "    communities = analysis_results['communities']['communities']\n"
                "    \n"
                "    # Create community visualization\n"
                "    network_viz.visualize_communities(\n"
                "        graph, communities,\n"
                "        title='Network Community Structure'\n"
                "    )\n"
                "    plt.show()\n"
                "    \n"
                "    print('Community visualizations created!')\n"
                "else:\n"
                "    print('Community results not available')"
            )
        ]
    
    def _create_interactive_visualization_cells(self) -> List[Dict]:
        """Create interactive visualization cells."""
        return [
            self._create_markdown_cell("## Interactive Visualizations"),
            self._create_code_cell(
                "# Create interactive network visualization\n"
                "print('Creating interactive network visualization...')\n"
                "interactive_fig = interactive_viz.create_interactive_network(\n"
                "    graph, layout='spring',\n"
                "    title='Interactive Social Network'\n"
                ")\n"
                "interactive_fig.show()\n\n"
                "print('Interactive visualization created!')"
            )
        ]
    
    def _create_component_results_cells(self, component_name: str) -> List[Dict]:
        """Create component-specific results cells."""
        return [
            self._create_markdown_cell(f"## {component_name.replace('_', ' ').title()} Results Summary"),
            self._create_code_cell(
                f"# Summarize {component_name} analysis results\n"
                f"print('{component_name.replace('_', ' ').title()} Analysis Complete!')\n"
                "print('Key findings and insights have been generated.')\n"
                "print('Refer to the generated documentation for detailed results.')"
            )
        ]
    
    def _create_quick_analysis_cells(self) -> List[Dict]:
        """Create quick analysis cells for visualization notebook."""
        return [
            self._create_markdown_cell("## Quick Analysis for Visualization"),
            self._create_code_cell(
                "# Perform quick analysis for visualization purposes\n"
                "# Note: For comprehensive analysis, use the full analysis notebook\n"
                "print('Performing quick analysis for visualization...')\n"
                "# Add minimal analysis code here if needed\n"
                "print('Quick analysis complete!')"
            )
        ]
    
    def _create_markdown_cell(self, content: str) -> Dict:
        """Create a markdown cell."""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": content.split('\n')
        }
    
    def _create_code_cell(self, content: str) -> Dict:
        """Create a code cell."""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": content.split('\n')
        }
    
    def create_notebook_index(self, notebooks: List[str]) -> str:
        """Create an index notebook linking to all generated notebooks."""
        cells = [
            self._create_markdown_cell(
                "# Social Network Analysis Notebooks Index\n\n"
                "This index provides links to all available analysis notebooks.\n\n"
                "## Available Notebooks"
            )
        ]
        
        for notebook_path in notebooks:
            notebook_name = Path(notebook_path).stem.replace('_', ' ').title()
            relative_path = Path(notebook_path).name
            cells.append(self._create_markdown_cell(f"- [{notebook_name}]({relative_path})"))
        
        cells.append(self._create_markdown_cell(
            "\n## Usage Instructions\n"
            "1. Start with the **Comprehensive Analysis** notebook for complete workflow\n"
            "2. Use **Component-specific** notebooks for focused analysis\n"
            "3. Use the **Visualization** notebook for creating plots and interpretations\n\n"
            "## Requirements\n"
            "Ensure all required packages are installed:\n"
            "```bash\n"
            "pip install networkx torch torch-geometric scikit-learn matplotlib plotly seaborn pandas numpy\n"
            "```"
        ))
        
        # Create notebook
        notebook = {
            "cells": cells,
            "metadata": self.notebook_metadata,
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save index notebook
        index_path = self.output_dir / "index.ipynb"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        self.logger.info(f"Notebook index created at {index_path}")
        return str(index_path)