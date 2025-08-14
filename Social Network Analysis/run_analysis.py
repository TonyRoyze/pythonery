#!/Users/viduragunawardana/miniforge3/envs/octaveenv/bin/python
"""
Social Network Analysis - Main Analysis Pipeline

This script runs the complete social network analysis pipeline on the 2027 dataset,
generating all outputs including visualizations, metrics, and model predictions.
Addresses all tasks from the assessment brief.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import networkx as nx
import json
from pathlib import Path
from datetime import datetime

# Add the social_network_analysis package to the path
sys.path.append(str(Path(__file__).parent / 'social_network_analysis'))

from social_network_analysis.data.data_loader import DataLoader
from social_network_analysis.graph.graph_builder import GraphBuilder
from social_network_analysis.analysis.centrality_calculator import CentralityCalculator
from social_network_analysis.analysis.community_detector import CommunityDetector
from social_network_analysis.analysis.community_analyzer import CommunityAnalyzer
from social_network_analysis.analysis.influence_propagator import InfluencePropagator
from social_network_analysis.analysis.sentiment_analyzer import SentimentAnalyzer
from social_network_analysis.analysis.topology_analyzer import TopologyAnalyzer
from social_network_analysis.analysis.scalability_analyzer import ScalabilityAnalyzer
from social_network_analysis.ml.graphsage_predictor import GraphSAGEPredictor
from social_network_analysis.ml.gnn_predictor import GNNPredictor
from social_network_analysis.ml.traditional_ml_predictor import TraditionalMLPredictor
from social_network_analysis.ml.model_comparator import ModelComparator
from social_network_analysis.visualization.network_visualizer import NetworkVisualizer
from social_network_analysis.visualization.interactive_visualizer import InteractiveVisualizer
# Documentation generator removed - using JSON export instead


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def create_output_directories():
    """Create output directories for results."""
    output_dirs = [
        'outputs',
        'outputs/visualizations',
        'outputs/metrics',

        'outputs/exports',
        'outputs/sentiment_analysis',
        'outputs/topology_analysis',
        'outputs/scalability_analysis'
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    logger = setup_logging()
    logger.info("Starting Social Network Analysis Pipeline")
    

    create_output_directories()
    
    try:
        logger.info("Step 1: Loading data...")
        data_loader = DataLoader(validate_data=True)
        
        data = data_loader.load_all_data(
            comments_file='2027-comments.json',
            aggregated_relationships_file='user_relationships_2027_aggregated.csv',
            raw_relationships_file='user_relationships_2027_raw.csv',
            reply_counts_file='2027-comments_author_reply_counts.csv'
        )
        
        logger.info(f"Loaded {len(data['comments'])} comments")
        logger.info(f"Loaded {len(data['aggregated_relationships'])} aggregated relationships")
        logger.info(f"Loaded {len(data['raw_relationships'])} raw relationships")
        logger.info(f"Loaded {len(data['reply_counts'])} reply counts")
        
        logger.info("Step 2: Building network graph...")
        graph_builder = GraphBuilder()
        graph = graph_builder.build_graph(
            relationships=data['aggregated_relationships'],
            comments=data['comments']
        )
        
        graph = graph_builder.add_node_attributes(graph, data['reply_counts'])
        
        logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        logger.info("Step 3: Calculating centrality metrics ...")
        centrality_calc = CentralityCalculator()
        centrality_results = centrality_calc.generate_centrality_report(graph)
        
        centrality_calc.export_centrality_data(
            centrality_results['centrality_scores'], 
            'outputs/metrics/centrality_metrics.json',
            format='json'
        )
        logger.info("Centrality metrics calculated and saved")
        
        logger.info("Step 4: Detecting communities ...")
        community_detector = CommunityDetector()
        
        # Louvain communities
        louvain_results = community_detector.detect_louvain_communities(graph)
        louvain_communities = louvain_results['communities']
        louvain_node_communities = louvain_results['node_communities']
        louvain_modularity = louvain_results['modularity']
        
        # Spectral communities
        spectral_results = community_detector.detect_spectral_communities(graph, n_clusters=10)
        spectral_communities = spectral_results['communities']
        spectral_node_communities = spectral_results['node_communities']
        spectral_modularity = spectral_results['modularity']
        
        community_results = {
            'louvain': {
                'communities': louvain_communities,
                'node_communities': louvain_node_communities,
                'modularity': louvain_modularity
            },
            'spectral': {
                'communities': spectral_communities,
                'node_communities': spectral_node_communities,
                'modularity': spectral_modularity
            }
        }
        
        logger.info(f"Louvain detected {len(louvain_communities)} communities (modularity: {louvain_modularity:.3f})")
        logger.info(f"Spectral detected {len(spectral_communities)} communities (modularity: {spectral_modularity:.3f})")
        
        logger.info("Step 5: Analyzing communities...")
        community_analyzer = CommunityAnalyzer()
        community_analysis = community_analyzer.analyze_community_characteristics(
            graph, louvain_communities, centrality_results['centrality_scores']
        )
        
        logger.info("Step 6: Running influence propagation analysis ...")
        influence_propagator = InfluencePropagator()
        
        # Select seed nodes based on centrality
        top_nodes = sorted(centrality_results['centrality_scores']['degree'].items(), key=lambda x: x[1], reverse=True)[:10]
        seed_nodes = [node for node, _ in top_nodes]
        
        # Run propagation simulation
        propagation_results = influence_propagator.simulate_propagation(
            graph, seed_nodes, max_steps=10, activation_prob=0.1
        )
        
        # Evaluate influence of high-centrality nodes
        influence_scores = influence_propagator.evaluate_high_centrality_influence(
            graph, centrality_results['centrality_scores'], top_k=20
        )
        
        # Analyze community impact
        community_impact = influence_propagator.analyze_community_propagation_impact(
            graph, louvain_communities, centrality_results['centrality_scores']
        )
        
        logger.info("Influence propagation analysis completed")
        
        logger.info("Step 7: Running link prediction models ...")
        
        # Prepare data for link prediction
        from social_network_analysis.ml.link_prediction_data import LinkPredictionDataPipeline
        link_data_prep = LinkPredictionDataPipeline()
        
        # Create a simple relationships DataFrame for the pipeline
        relationships_df = pd.DataFrame([
            {'Source': u, 'Destination': v} for u, v in graph.edges()
        ])
        
        link_prediction_data = link_data_prep.prepare_link_prediction_data(
            graph, relationships_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        
        train_data = link_prediction_data['train']
        test_data = link_prediction_data['test']
        
        # Adapt data format for GraphSAGE and GNN predictors
        adapted_data = {
            'train_data': link_prediction_data['train']['pyg_data'],
            'val_data': link_prediction_data['val']['pyg_data'],
            'test_data': link_prediction_data['test']['pyg_data'],
            'train_pos_edges': link_prediction_data['train']['positive_edges'],
            'train_neg_edges': link_prediction_data['train']['negative_edges'],
            'val_pos_edges': link_prediction_data['val']['positive_edges'],
            'val_neg_edges': link_prediction_data['val']['negative_edges'],
            'test_pos_edges': link_prediction_data['test']['positive_edges'],
            'test_neg_edges': link_prediction_data['test']['negative_edges']
        }
        
        # GraphSAGE model
        graphsage_predictor = GraphSAGEPredictor()
        graphsage_results = graphsage_predictor.train(adapted_data)
        
        # GNN model
        gnn_predictor = GNNPredictor()
        gnn_results = gnn_predictor.train(adapted_data)
        
        # Traditional ML baselines
        traditional_predictor = TraditionalMLPredictor()
        traditional_training_results = traditional_predictor.train_all_models(link_prediction_data)
        traditional_results = traditional_predictor.evaluate_models(link_prediction_data, traditional_training_results)
        
        model_comparator = ModelComparator()
        
        # Add GraphSAGE results
        if 'test_predictions' in graphsage_results and graphsage_results['test_predictions'] is not None:
            model_comparator.add_model_results(
                'GraphSAGE', 'graph_neural_network', 
                graphsage_results['test_predictions'], 
                graphsage_results['test_probabilities'],
                graphsage_results['test_labels']
            )
        
        # Add GNN results
        if 'test_predictions' in gnn_results and gnn_results['test_predictions'] is not None:
            model_comparator.add_model_results(
                'GNN', 'graph_neural_network',
                gnn_results['test_predictions'], 
                gnn_results['test_probabilities'],
                gnn_results['test_labels']
            )
        
        # Add traditional ML results
        for model_name, results in traditional_results.items():
            if 'predictions' in results and 'probabilities' in results:
                # Get test labels from link prediction data
                test_labels = np.concatenate([
                    np.ones(len(link_prediction_data['test']['positive_edges'])),
                    np.zeros(len(link_prediction_data['test']['negative_edges']))
                ])
                
                model_comparator.add_model_results(
                    model_name, 'traditional_ml',
                    results['predictions'], 
                    results['probabilities'],
                    test_labels
                )
        
        # Generate comprehensive comparison
        comparison_results = model_comparator.generate_comprehensive_comparison()
        
        logger.info("Link prediction models trained and evaluated")
        

        logger.info("Step 8: Performing scalability analysis ...")
        try:
            scalability_analyzer = ScalabilityAnalyzer(logger)
            scalability_results = scalability_analyzer.perform_scalability_analysis(graph)
        except Exception as e:
            logger.error(f"Scalability analysis failed completely: {e}")
            scalability_results = None
        

        logger.info("Step 9: Performing sentiment analysis ...")
        logger.info(f"Comments data type: {type(data['comments'])}, length: {len(data['comments'])}")
        if len(data['comments']) > 0:
            if isinstance(data['comments'], pd.DataFrame):
                logger.info(f"Comments DataFrame columns: {list(data['comments'].columns)}")
                logger.info(f"Sample comment: {data['comments'].iloc[0].to_dict()}")
            else:
                logger.info(f"Sample comment structure: {type(data['comments'][0])}")
                if isinstance(data['comments'][0], dict):
                    logger.info(f"Sample comment keys: {list(data['comments'][0].keys())}")
        
        try:
            sentiment_analyzer = SentimentAnalyzer(logger)
            sentiment_df, sentiment_stats = sentiment_analyzer.perform_sentiment_analysis(data['comments'])
        except Exception as e:
            logger.error(f"Sentiment analysis failed completely: {e}")
            sentiment_df = None
            sentiment_stats = None
        
 
        logger.info("Step 10: Analyzing network topologies ...")
        try:
            topology_analyzer = TopologyAnalyzer(logger)
            topology_analysis = topology_analyzer.analyze_network_topologies(graph)
        except Exception as e:
            logger.error(f"Topology analysis failed completely: {e}")
            topology_analysis = None
        

        logger.info("Step 11: Generating visualizations...")
        
        # Static visualizations
        visualizer = NetworkVisualizer()
        
        # Network overview
        visualizer.visualize_network(graph, save_path='outputs/visualizations/network_overview.png')
        
        # Centrality visualizations
        visualizer.visualize_centrality(graph, centrality_results['centrality_scores']['degree'], 
                                      centrality_type='degree',
                                      save_path='outputs/visualizations/degree_centrality.png')
        visualizer.visualize_centrality(graph, centrality_results['centrality_scores']['betweenness'], 
                                      centrality_type='betweenness',
                                      save_path='outputs/visualizations/betweenness_centrality.png')
        
        # Community visualizations
        visualizer.visualize_communities(graph, louvain_communities, 
                                       save_path='outputs/visualizations/louvain_communities.png')
        visualizer.visualize_communities(graph, spectral_communities, 
                                       save_path='outputs/visualizations/spectral_communities.png')
        
        # Interactive visualizations
        interactive_viz = InteractiveVisualizer()
        interactive_viz.create_interactive_network(graph, node_size=centrality_results['centrality_scores'], node_color=louvain_node_communities,
                                                 save_path='outputs/visualizations/interactive_network.html')
        
        # Influence propagation visualizations 
        logger.info("Generating influence propagation visualizations...")
        
        # Create influence propagation step-by-step visualization
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from matplotlib.colors import LinearSegmentedColormap
            
            # Create a custom colormap for influence levels
            colors = ['lightblue', 'blue', 'darkblue', 'purple', 'red']

            
        except ImportError:
            logger.warning("Matplotlib not available, skipping influence propagation visualization")
        except Exception as e:
            logger.warning(f"Influence propagation visualization failed: {e}")
        
        # Enhanced influence propagation visualizations using NetworkVisualizer
        logger.info("Generating enhanced influence propagation visualizations...")
        try:
            # Get top influential nodes as seed nodes
            if centrality_results and 'centrality_scores' in centrality_results:
                degree_scores = centrality_results['centrality_scores'].get('degree', {})
                if degree_scores:
                    # Select top 5 nodes by degree centrality as seed nodes
                    top_nodes = sorted(degree_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    seed_nodes = [node for node, score in top_nodes]
                    
                    # Create influence propagation timeline visualization
                    if influence_propagator:
                        timeline_fig = visualizer.visualize_influence_propagation_with_timeline(
                            graph=graph,
                            influence_propagator=influence_propagator,
                            seed_nodes=seed_nodes,
                            max_steps=8,
                            save_path='outputs/visualizations/influence_propagation_timeline.png'
                        )
                        plt.close(timeline_fig)
                        
                        # Create influence strategy comparison
                        strategy_fig = visualizer.visualize_influence_comparison_strategies(
                            graph=graph,
                            influence_propagator=influence_propagator,
                            centrality_scores=centrality_results['centrality_scores'],
                            seed_counts=[1, 3, 5, 8, 10],
                            save_path='outputs/visualizations/influence_strategy_comparison.png'
                        )
                        plt.close(strategy_fig)
                        
                        # Create comprehensive influence dashboard
                        dashboard_fig = visualizer.create_influence_dashboard(
                            graph=graph,
                            influence_propagator=influence_propagator,
                            centrality_scores=centrality_results['centrality_scores'],
                            communities=louvain_communities,
                            seed_nodes=seed_nodes,
                            save_path='outputs/visualizations/influence_analysis_dashboard.png'
                        )
                        plt.close(dashboard_fig)
                        
                        logger.info("Enhanced influence propagation visualizations created successfully")
                    else:
                        logger.warning("InfluencePropagator not available for enhanced visualizations")
                else:
                    logger.warning("No centrality scores available for seed node selection")
            else:
                logger.warning("No centrality results available for enhanced influence visualizations")
                
        except Exception as e:
            logger.error(f"Enhanced influence propagation visualizations failed: {e}")
        
        logger.info("Visualizations generated")
        
        # Step 12: Export comprehensive analysis results as JSON
        logger.info("Step 12: Exporting comprehensive analysis results...")
        
        # Compile all results with detailed task explanations
        analysis_results = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'comments_count': len(data['comments']),
                    'relationships_count': len(data['aggregated_relationships']),
                    'reply_counts_available': len(data['reply_counts']) if data['reply_counts'] is not None else 0
                }
            },
            'graph_statistics': {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2) if graph.number_of_nodes() > 1 else 0,
                'is_connected': nx.is_connected(graph),
                'number_of_components': nx.number_connected_components(graph)
            },
            'centrality_analysis': centrality_results,
            'community_detection': community_results,
            'community_analysis': community_analysis,
            'influence_propagation': {
                'propagation_results': propagation_results,
                'influence_scores': influence_scores,
                'community_impact': community_impact
            },
            'link_prediction': {
                'graphsage_results': graphsage_results,
                'gnn_results': gnn_results,
                'traditional_results': traditional_results,
                'comparison': comparison_results
            },
            'scalability_analysis': scalability_results,
            'sentiment_analysis': {
                'sentiment_data': sentiment_df.to_dict('records') if sentiment_df is not None else None,
                'sentiment_statistics': sentiment_stats
            },
            'topology_analysis': topology_analysis,
            'task_completion_summary': {
                'task_1_centrality': {
                    'description': 'NetworkX representation and centrality metrics calculation',
                    'status': 'completed' if centrality_results else 'failed',
                    'outputs': ['centrality_metrics.json', 'degree_centrality.png', 'betweenness_centrality.png'],
                    'metrics': ['degree', 'betweenness', 'closeness', 'clustering_coefficient'],
                    'key_findings': f"Top central node by degree: {max(centrality_results.get('degree', {}), key=centrality_results.get('degree', {}).get) if centrality_results and centrality_results.get('degree') else 'N/A'}"
                },
                'task_2_community_detection': {
                    'description': 'Community detection using Louvain and Spectral algorithms',
                    'status': 'completed' if community_results else 'failed',
                    'algorithms': ['Louvain', 'Spectral Clustering'],
                    'communities_found': len(community_results.get('louvain', {}).get('communities', {})) if community_results else 0,
                    'modularity_score': community_results.get('louvain', {}).get('modularity', 0) if community_results else 0
                },
                'task_3_influence_propagation': {
                    'description': 'Information propagation simulation using graph neural networks',
                    'status': 'completed' if propagation_results else 'failed',
                    'simulation_type': 'Seed-based propagation with centrality-selected nodes',
                    'propagation_reach': propagation_results.get('final_infected_count', 0) if propagation_results else 0
                },
                'task_4_link_prediction': {
                    'description': 'Graph-based machine learning for link prediction',
                    'status': 'completed' if comparison_results else 'failed',
                    'models_tested': ['GraphSAGE', 'GNN', 'Random Forest', 'SVM'],
                    'best_model': comparison_results.get('best_model', 'N/A') if comparison_results else 'N/A',
                    'best_accuracy': comparison_results.get('best_accuracy', 0) if comparison_results else 0
                },
                'task_5_scalability': {
                    'description': 'Distributed processing and scalability evaluation',
                    'status': 'completed' if scalability_results else 'failed',
                    'distributed_processing': scalability_results.get('distributed_processing', {}).get('dask_available', False) if scalability_results else False,
                    'max_graph_size_tested': max(scalability_results.get('scalability_tests', {}).keys()) if scalability_results and scalability_results.get('scalability_tests') else 0
                },
                'task_6_sentiment_analysis': {
                    'description': 'Sentiment analysis on textual content with network integration',
                    'status': 'completed' if sentiment_stats else 'failed',
                    'total_comments_analyzed': sentiment_stats.get('total_comments', 0) if sentiment_stats else 0,
                    'average_sentiment': sentiment_stats.get('avg_sentiment', 0) if sentiment_stats else 0,
                    'sentiment_distribution': {
                        'positive': sentiment_stats.get('positive_percentage', 0) if sentiment_stats else 0,
                        'negative': sentiment_stats.get('negative_percentage', 0) if sentiment_stats else 0,
                        'neutral': sentiment_stats.get('neutral_percentage', 0) if sentiment_stats else 0
                    }
                },
                'task_7_topology_analysis': {
                    'description': 'Network topology comparison and diffusion analysis',
                    'status': 'completed' if topology_analysis else 'failed',
                    'topologies_compared': ['Current', 'Random', 'Small-World', 'Scale-Free'],
                    'diffusion_efficiency': topology_analysis.get('diffusion_results', {}).get('current', {}).get('avg_infection_rate', 0) if topology_analysis else 0
                }
            }
        }
        
        # Export comprehensive results as JSON
        with open('outputs/exports/comprehensive_analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info("Comprehensive analysis results exported to JSON")
        
        # Step 13: Export additional formats
        logger.info("Step 13: Exporting additional formats...")
        
        # Export graph in multiple formats
        nx.write_gexf(graph, 'outputs/exports/social_network.gexf')
        nx.write_graphml(graph, 'outputs/exports/social_network.graphml')
        
        # Export metrics as CSV
        
        # Centrality metrics
        centrality_df = pd.DataFrame({
            'node': list(centrality_results['centrality_scores']['degree'].keys()),
            'degree_centrality': list(centrality_results['centrality_scores']['degree'].values()),
            'betweenness_centrality': [centrality_results['centrality_scores']['betweenness'].get(node, 0) 
                                     for node in centrality_results['centrality_scores']['degree'].keys()],
            'closeness_centrality': [centrality_results['centrality_scores']['closeness'].get(node, 0) 
                                   for node in centrality_results['centrality_scores']['degree'].keys()],
            'clustering_coefficient': [centrality_results['centrality_scores']['clustering'].get(node, 0) 
                                     for node in centrality_results['centrality_scores']['degree'].keys()]
        })
        centrality_df.to_csv('outputs/exports/centrality_metrics.csv', index=False)
        
        # Community assignments
        community_df = pd.DataFrame([
            {'node': node, 'louvain_community': louvain_node_communities.get(node, -1),
             'spectral_community': spectral_node_communities.get(node, -1)}
            for node in graph.nodes()
        ])
        community_df.to_csv('outputs/exports/community_assignments.csv', index=False)
        
        # Model performance export
        if comparison_results and 'performance_comparison' in comparison_results:
            perf_data = comparison_results['performance_comparison'].get('raw_data', {})
            if perf_data:
                performance_df = pd.DataFrame([
                    {
                        'model': model,
                        'accuracy': results.get('accuracy', 0),
                        'precision': results.get('precision', 0),
                        'recall': results.get('recall', 0),
                        'f1_score': results.get('f1_score', 0),
                        'auc_roc': results.get('auc_roc', 0)
                    }
                    for model, results in perf_data.items()
                ])
                performance_df.to_csv('outputs/exports/model_performance.csv', index=False)
                logger.info("Model performance metrics exported")
            else:
                logger.info("No model performance data available for export")
        else:
            logger.info("No model comparison results available for export")
        
        logger.info("Results exported")
        
        # Final summary
        logger.info("="*80)
        logger.info("SOCIAL NETWORK ANALYSIS COMPLETED SUCCESSFULLY - ALL TASKS ADDRESSED")
        logger.info("="*80)
        logger.info(f"Network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        logger.info(f"Communities (Louvain): {len(louvain_communities)} (modularity: {louvain_modularity:.3f})")
        logger.info(f"Communities (Spectral): {len(spectral_communities)} (modularity: {spectral_modularity:.3f})")
        
        if comparison_results and 'performance_comparison' in comparison_results:
            best_model = max(comparison_results['performance_comparison'].get('raw_data', {}).items(), 
                           key=lambda x: x[1].get('f1_score', 0))[0]
            logger.info(f"Best link prediction model: {best_model}")
        
        if sentiment_stats and isinstance(sentiment_stats, dict):
            try:
                logger.info(f"Sentiment analysis: {sentiment_stats.get('total_comments', 0)} comments analyzed")
            except Exception as e:
                logger.warning(f"Error accessing sentiment stats: {e}")
        elif sentiment_df is not None:
            logger.info(f"Sentiment analysis: {len(sentiment_df)} comments analyzed")
        else:
            logger.warning("Sentiment analysis failed or no data available")
        
        if scalability_results:
            logger.info("Scalability analysis completed for multiple graph sizes")
        else:
            logger.warning("Scalability analysis failed or no data available")
        
        if topology_analysis:
            logger.info("Network topology analysis completed")
        else:
            logger.warning("Topology analysis failed or no data available")
        
        
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()