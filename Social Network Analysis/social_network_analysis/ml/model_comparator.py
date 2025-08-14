"""
Comprehensive model evaluation and comparison system for social network analysis.

This module implements systematic performance comparison between different models,
statistical significance analysis, and feature importance analysis for understanding
prediction factors in link prediction tasks.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


class ModelComparator:
    """
    Comprehensive model comparison system for link prediction models.
    
    Provides systematic performance comparison, statistical significance testing,
    and feature importance analysis across different model types including
    traditional ML and graph-based approaches.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model comparator.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Storage for comparison results
        self.model_results = {}
        self.comparison_results = {}
        self.statistical_tests = {}
        self.feature_importance_analysis = {}
        
        # Metrics to compare
        self.metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def add_model_results(self, 
                         model_name: str,
                         model_type: str,
                         test_predictions: np.ndarray,
                         test_probabilities: np.ndarray,
                         test_labels: np.ndarray,
                         training_time: Optional[float] = None,
                         model_params: Optional[Dict] = None,
                         feature_importance: Optional[Dict] = None,
                         cross_val_scores: Optional[Dict] = None) -> None:
        """
        Add model results for comparison.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('traditional', 'graph')
            test_predictions: Binary predictions on test set
            test_probabilities: Prediction probabilities on test set
            test_labels: True labels for test set
            training_time: Time taken to train the model (seconds)
            model_params: Model hyperparameters
            feature_importance: Feature importance scores (for traditional ML)
            cross_val_scores: Cross-validation scores for different metrics
        """
        # Calculate test metrics
        test_metrics = {
            'accuracy': accuracy_score(test_labels, test_predictions),
            'precision': precision_score(test_labels, test_predictions, zero_division=0),
            'recall': recall_score(test_labels, test_predictions, zero_division=0),
            'f1_score': f1_score(test_labels, test_predictions, zero_division=0),
            'auc_roc': roc_auc_score(test_labels, test_probabilities) if len(np.unique(test_labels)) > 1 else 0.0
        }
        
        # Store model results
        self.model_results[model_name] = {
            'model_type': model_type,
            'test_metrics': test_metrics,
            'test_predictions': test_predictions,
            'test_probabilities': test_probabilities,
            'test_labels': test_labels,
            'training_time': training_time,
            'model_params': model_params or {},
            'feature_importance': feature_importance or {},
            'cross_val_scores': cross_val_scores or {},
            'confusion_matrix': confusion_matrix(test_labels, test_predictions)
        }
        
        self.logger.info(f"Added results for {model_name}: F1={test_metrics['f1_score']:.4f}, "
                        f"AUC={test_metrics['auc_roc']:.4f}")
    
    def perform_statistical_significance_analysis(self, 
                                                 significance_level: float = 0.05,
                                                 test_type: str = 'auto') -> Dict:
        """
        Perform statistical significance analysis between models.
        
        Args:
            significance_level: Significance level for hypothesis testing
            test_type: Type of statistical test ('auto', 'paired_ttest', 'wilcoxon', 'mannwhitney')
            
        Returns:
            Dictionary containing statistical test results
        """
        if len(self.model_results) < 2:
            self.logger.warning("Need at least 2 models for statistical comparison")
            return {}
        
        self.logger.info("Performing statistical significance analysis...")
        
        statistical_results = {
            'pairwise_comparisons': {},
            'overall_comparison': {},
            'significance_level': significance_level,
            'test_type': test_type
        }
        
        model_names = list(self.model_results.keys())
        
        # Pairwise comparisons for each metric
        for metric in self.metrics:
            statistical_results['pairwise_comparisons'][metric] = {}
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    
                    # Get cross-validation scores if available, otherwise use test metrics
                    if (self.model_results[model1]['cross_val_scores'] and 
                        self.model_results[model2]['cross_val_scores'] and
                        metric in self.model_results[model1]['cross_val_scores'] and
                        metric in self.model_results[model2]['cross_val_scores']):
                        
                        scores1 = self.model_results[model1]['cross_val_scores'][metric]
                        scores2 = self.model_results[model2]['cross_val_scores'][metric]
                        
                        # Ensure scores are arrays
                        if not isinstance(scores1, (list, np.ndarray)):
                            scores1 = [scores1]
                        if not isinstance(scores2, (list, np.ndarray)):
                            scores2 = [scores2]
                        
                        scores1 = np.array(scores1)
                        scores2 = np.array(scores2)
                        
                        # Perform statistical test
                        if test_type == 'auto':
                            # Use paired t-test if we have paired samples, otherwise Mann-Whitney U
                            if len(scores1) == len(scores2) and len(scores1) > 1:
                                test_stat, p_value = ttest_rel(scores1, scores2)
                                test_used = 'paired_ttest'
                            else:
                                test_stat, p_value = mannwhitneyu(scores1, scores2, alternative='two-sided')
                                test_used = 'mannwhitney'
                        elif test_type == 'paired_ttest':
                            if len(scores1) == len(scores2):
                                test_stat, p_value = ttest_rel(scores1, scores2)
                                test_used = 'paired_ttest'
                            else:
                                self.logger.warning(f"Cannot perform paired t-test for {model1} vs {model2}: different sample sizes")
                                continue
                        elif test_type == 'wilcoxon':
                            if len(scores1) == len(scores2):
                                test_stat, p_value = wilcoxon(scores1, scores2)
                                test_used = 'wilcoxon'
                            else:
                                self.logger.warning(f"Cannot perform Wilcoxon test for {model1} vs {model2}: different sample sizes")
                                continue
                        elif test_type == 'mannwhitney':
                            test_stat, p_value = mannwhitneyu(scores1, scores2, alternative='two-sided')
                            test_used = 'mannwhitney'
                        
                    else:
                        # Use single test metrics (less reliable for statistical testing)
                        score1 = self.model_results[model1]['test_metrics'][metric]
                        score2 = self.model_results[model2]['test_metrics'][metric]
                        
                        # Create artificial samples for testing (not ideal, but provides some comparison)
                        scores1 = np.array([score1])
                        scores2 = np.array([score2])
                        
                        # Simple difference test
                        diff = abs(score1 - score2)
                        test_stat = diff
                        p_value = 1.0 if diff < 0.01 else 0.5  # Rough approximation
                        test_used = 'simple_difference'
                    
                    # Store results
                    comparison_key = f"{model1}_vs_{model2}"
                    statistical_results['pairwise_comparisons'][metric][comparison_key] = {
                        'model1': model1,
                        'model2': model2,
                        'model1_mean': np.mean(scores1),
                        'model2_mean': np.mean(scores2),
                        'difference': np.mean(scores1) - np.mean(scores2),
                        'test_statistic': test_stat,
                        'p_value': p_value,
                        'significant': p_value < significance_level,
                        'test_used': test_used,
                        'effect_size': self._calculate_effect_size(scores1, scores2)
                    }
        
        # Overall comparison using Friedman test if we have multiple models and CV scores
        if len(model_names) > 2:
            for metric in self.metrics:
                cv_scores_available = all(
                    self.model_results[model]['cross_val_scores'] and 
                    metric in self.model_results[model]['cross_val_scores']
                    for model in model_names
                )
                
                if cv_scores_available:
                    # Collect all CV scores
                    all_scores = []
                    for model in model_names:
                        scores = self.model_results[model]['cross_val_scores'][metric]
                        if not isinstance(scores, (list, np.ndarray)):
                            scores = [scores]
                        all_scores.append(np.array(scores))
                    
                    # Check if all have same length
                    if len(set(len(scores) for scores in all_scores)) == 1:
                        try:
                            test_stat, p_value = friedmanchisquare(*all_scores)
                            statistical_results['overall_comparison'][metric] = {
                                'test_statistic': test_stat,
                                'p_value': p_value,
                                'significant': p_value < significance_level,
                                'test_used': 'friedman'
                            }
                        except Exception as e:
                            self.logger.warning(f"Failed to perform Friedman test for {metric}: {str(e)}")
        
        self.statistical_tests = statistical_results
        self.logger.info("Statistical significance analysis completed!")
        
        return statistical_results
    
    def _calculate_effect_size(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        if len(scores1) <= 1 or len(scores2) <= 1:
            return 0.0
        
        pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                             (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                            (len(scores1) + len(scores2) - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(scores1) - np.mean(scores2)) / pooled_std
    
    def analyze_feature_importance(self, 
                                  feature_names: Optional[List[str]] = None,
                                  top_k: int = 20) -> Dict:
        """
        Analyze and compare feature importance across traditional ML models.
        
        Args:
            feature_names: Names of features (if available)
            top_k: Number of top features to analyze
            
        Returns:
            Dictionary containing feature importance analysis
        """
        self.logger.info("Analyzing feature importance...")
        
        # Collect feature importance from traditional ML models
        traditional_models = {
            name: data for name, data in self.model_results.items()
            if data['model_type'] == 'traditional' and data['feature_importance']
        }
        
        if not traditional_models:
            self.logger.warning("No traditional ML models with feature importance found")
            return {}
        
        feature_analysis = {
            'individual_importance': {},
            'consensus_ranking': {},
            'importance_correlation': {},
            'top_features': {}
        }
        
        # Collect all feature importance scores
        all_importances = {}
        for model_name, model_data in traditional_models.items():
            importance_dict = model_data['feature_importance']
            feature_analysis['individual_importance'][model_name] = importance_dict
            
            # Store in matrix format for correlation analysis
            for feature, importance in importance_dict.items():
                if feature not in all_importances:
                    all_importances[feature] = {}
                all_importances[feature][model_name] = importance
        
        # Create importance matrix
        if all_importances:
            importance_df = pd.DataFrame(all_importances).T.fillna(0)
            
            # Calculate consensus ranking (average importance across models)
            consensus_importance = importance_df.mean(axis=1).sort_values(ascending=False)
            feature_analysis['consensus_ranking'] = consensus_importance.to_dict()
            
            # Get top features
            top_features = consensus_importance.head(top_k)
            feature_analysis['top_features'] = {
                'features': top_features.index.tolist(),
                'importance_scores': top_features.values.tolist(),
                'feature_names': feature_names[:len(top_features)] if feature_names else None
            }
            
            # Calculate correlation between models' feature rankings
            if len(traditional_models) > 1:
                correlation_matrix = importance_df.corr()
                feature_analysis['importance_correlation'] = correlation_matrix.to_dict()
            
            # Feature stability analysis
            feature_std = importance_df.std(axis=1)
            feature_analysis['feature_stability'] = {
                'most_stable': feature_std.nsmallest(10).to_dict(),
                'least_stable': feature_std.nlargest(10).to_dict()
            }
        
        self.feature_importance_analysis = feature_analysis
        self.logger.info("Feature importance analysis completed!")
        
        return feature_analysis
    
    def generate_comprehensive_comparison(self) -> Dict:
        """
        Generate comprehensive comparison results combining all analyses.
        
        Returns:
            Dictionary containing complete comparison results
        """
        self.logger.info("Generating comprehensive model comparison...")
        
        if not self.model_results:
            self.logger.error("No model results available for comparison")
            return {}
        
        # Basic performance comparison
        performance_comparison = self._create_performance_comparison()
        
        # Model ranking
        model_ranking = self._rank_models()
        
        # Generate insights
        insights = self._generate_insights()
        
        # Compile comprehensive results
        comprehensive_results = {
            'performance_comparison': performance_comparison,
            'model_ranking': model_ranking,
            'statistical_significance': self.statistical_tests,
            'feature_importance': self.feature_importance_analysis,
            'insights': insights,
            'summary': {
                'total_models': len(self.model_results),
                'traditional_models': len([m for m in self.model_results.values() if m['model_type'] == 'traditional']),
                'graph_models': len([m for m in self.model_results.values() if m['model_type'] == 'graph']),
                'best_overall_model': model_ranking['overall_best']['model'] if model_ranking.get('overall_best') else None,
                'significant_differences_found': any(
                    any(comp['significant'] for comp in metric_comps.values())
                    for metric_comps in self.statistical_tests.get('pairwise_comparisons', {}).values()
                ) if self.statistical_tests else False
            }
        }
        
        self.comparison_results = comprehensive_results
        self.logger.info("Comprehensive comparison completed!")
        
        return comprehensive_results
    
    def _create_performance_comparison(self) -> Dict:
        """Create detailed performance comparison table."""
        performance_data = {}
        
        for model_name, model_data in self.model_results.items():
            performance_data[model_name] = {
                'model_type': model_data['model_type'],
                **model_data['test_metrics'],
                'training_time': model_data.get('training_time', 'N/A'),
                'parameters': len(model_data.get('model_params', {}))
            }
        
        # Convert to DataFrame for easier analysis
        performance_df = pd.DataFrame(performance_data).T
        
        return {
            'raw_data': performance_data,
            'dataframe': performance_df.to_dict(),
            'best_per_metric': {
                metric: {
                    'model': performance_df[metric].idxmax(),
                    'score': performance_df[metric].max()
                }
                for metric in self.metrics if metric in performance_df.columns
            },
            'worst_per_metric': {
                metric: {
                    'model': performance_df[metric].idxmin(),
                    'score': performance_df[metric].min()
                }
                for metric in self.metrics if metric in performance_df.columns
            }
        }
    
    def _rank_models(self) -> Dict:
        """Rank models based on multiple criteria."""
        if not self.model_results:
            return {}
        
        # Create ranking based on different criteria
        rankings = {}
        
        # Rank by individual metrics
        for metric in self.metrics:
            metric_scores = {
                name: data['test_metrics'][metric] 
                for name, data in self.model_results.items()
            }
            sorted_models = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
            rankings[f'by_{metric}'] = [
                {'model': model, 'score': score} for model, score in sorted_models
            ]
        
        # Overall ranking (weighted average of metrics)
        weights = {'f1_score': 0.4, 'auc_roc': 0.3, 'precision': 0.15, 'recall': 0.15}
        overall_scores = {}
        
        for model_name, model_data in self.model_results.items():
            weighted_score = sum(
                model_data['test_metrics'].get(metric, 0) * weight
                for metric, weight in weights.items()
            )
            overall_scores[model_name] = weighted_score
        
        sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        rankings['overall_weighted'] = [
            {'model': model, 'score': score} for model, score in sorted_overall
        ]
        
        # Best overall model
        if sorted_overall:
            rankings['overall_best'] = {
                'model': sorted_overall[0][0],
                'score': sorted_overall[0][1],
                'metrics': self.model_results[sorted_overall[0][0]]['test_metrics']
            }
        
        return rankings
    
    def _generate_insights(self) -> List[str]:
        """Generate insights from the comparison results."""
        insights = []
        
        if not self.model_results:
            return insights
        
        # Model type comparison
        traditional_models = [m for m in self.model_results.values() if m['model_type'] == 'traditional']
        graph_models = [m for m in self.model_results.values() if m['model_type'] == 'graph']
        
        if traditional_models and graph_models:
            traditional_f1 = np.mean([m['test_metrics']['f1_score'] for m in traditional_models])
            graph_f1 = np.mean([m['test_metrics']['f1_score'] for m in graph_models])
            
            if graph_f1 > traditional_f1 + 0.05:
                insights.append(f"Graph-based models significantly outperform traditional ML "
                              f"(avg F1: {graph_f1:.3f} vs {traditional_f1:.3f})")
            elif traditional_f1 > graph_f1 + 0.05:
                insights.append(f"Traditional ML models outperform graph-based models "
                              f"(avg F1: {traditional_f1:.3f} vs {graph_f1:.3f})")
            else:
                insights.append(f"Traditional ML and graph-based models perform similarly "
                              f"(F1 difference: {abs(graph_f1 - traditional_f1):.3f})")
        
        # Performance insights
        f1_scores = [data['test_metrics']['f1_score'] for data in self.model_results.values()]
        if f1_scores:
            best_f1 = max(f1_scores)
            worst_f1 = min(f1_scores)
            
            if best_f1 - worst_f1 > 0.1:
                insights.append(f"Large performance variation across models "
                              f"(F1 range: {worst_f1:.3f} - {best_f1:.3f})")
            else:
                insights.append(f"Models show consistent performance "
                              f"(F1 range: {worst_f1:.3f} - {best_f1:.3f})")
        
        # Statistical significance insights
        if self.statistical_tests and 'pairwise_comparisons' in self.statistical_tests:
            significant_pairs = 0
            total_pairs = 0
            
            for metric_comps in self.statistical_tests['pairwise_comparisons'].values():
                for comp in metric_comps.values():
                    total_pairs += 1
                    if comp.get('significant', False):
                        significant_pairs += 1
            
            if total_pairs > 0:
                sig_percentage = (significant_pairs / total_pairs) * 100
                insights.append(f"{sig_percentage:.1f}% of model comparisons show "
                              f"statistically significant differences")
        
        # Feature importance insights
        if self.feature_importance_analysis and 'top_features' in self.feature_importance_analysis:
            top_features = self.feature_importance_analysis['top_features'].get('features', [])
            if top_features:
                insights.append(f"Top predictive features: {', '.join(top_features[:3])}")
        
        return insights
    
    def create_visualization_plots(self, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualization plots for model comparison.
        
        Args:
            save_path: Directory to save plots (optional)
            
        Returns:
            Dictionary of matplotlib figures
        """
        if not self.model_results:
            self.logger.error("No model results available for visualization")
            return {}
        
        figures = {}
        
        # 1. Performance comparison bar plot
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        models = list(self.model_results.keys())
        metrics_data = {metric: [] for metric in self.metrics}
        
        for model in models:
            for metric in self.metrics:
                metrics_data[metric].append(self.model_results[model]['test_metrics'][metric])
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(self.metrics):
            ax1.bar(x + i * width, metrics_data[metric], width, label=metric.replace('_', ' ').title())
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['performance_comparison'] = fig1
        
        # 2. ROC curves comparison
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        for model_name, model_data in self.model_results.items():
            if len(np.unique(model_data['test_labels'])) > 1:
                fpr, tpr, _ = roc_curve(model_data['test_labels'], model_data['test_probabilities'])
                auc_score = model_data['test_metrics']['auc_roc']
                ax2.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        figures['roc_comparison'] = fig2
        
        # 3. Precision-Recall curves comparison
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        
        for model_name, model_data in self.model_results.items():
            precision, recall, _ = precision_recall_curve(
                model_data['test_labels'], model_data['test_probabilities']
            )
            ax3.plot(recall, precision, label=model_name)
        
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curves Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        figures['pr_comparison'] = fig3
        
        # 4. Feature importance plot (if available)
        if self.feature_importance_analysis and 'top_features' in self.feature_importance_analysis:
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            
            top_features = self.feature_importance_analysis['top_features']
            features = top_features['features'][:15]  # Top 15 features
            scores = top_features['importance_scores'][:15]
            
            y_pos = np.arange(len(features))
            ax4.barh(y_pos, scores)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(features)
            ax4.set_xlabel('Importance Score')
            ax4.set_title('Top Feature Importance (Consensus Ranking)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            figures['feature_importance'] = fig4
        
        # Save plots if path provided
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            for plot_name, fig in figures.items():
                fig.savefig(f"{save_path}/{plot_name}.png", dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved {plot_name} plot to {save_path}")
        
        return figures
    
    def export_results(self, filepath: str, format: str = 'json') -> None:
        """
        Export comparison results to file.
        
        Args:
            filepath: Path to save results
            format: Export format ('json', 'csv', 'excel')
        """
        if not self.comparison_results:
            self.logger.error("No comparison results to export. Run generate_comprehensive_comparison() first.")
            return
        
        if format == 'json':
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            exportable_results = self._make_json_serializable(self.comparison_results)
            
            with open(filepath, 'w') as f:
                json.dump(exportable_results, f, indent=2)
                
        elif format == 'csv':
            # Export performance comparison as CSV
            if 'performance_comparison' in self.comparison_results:
                perf_df = pd.DataFrame(self.comparison_results['performance_comparison']['raw_data']).T
                perf_df.to_csv(filepath)
                
        elif format == 'excel':
            with pd.ExcelWriter(filepath) as writer:
                # Performance comparison
                if 'performance_comparison' in self.comparison_results:
                    perf_df = pd.DataFrame(self.comparison_results['performance_comparison']['raw_data']).T
                    perf_df.to_excel(writer, sheet_name='Performance')
                
                # Model ranking
                if 'model_ranking' in self.comparison_results:
                    for ranking_type, ranking_data in self.comparison_results['model_ranking'].items():
                        if isinstance(ranking_data, list):
                            ranking_df = pd.DataFrame(ranking_data)
                            ranking_df.to_excel(writer, sheet_name=f'Ranking_{ranking_type}', index=False)
        
        self.logger.info(f"Results exported to {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        else:
            return obj
    
    def get_summary_report(self) -> str:
        """
        Generate a text summary report of the comparison results.
        
        Returns:
            Formatted string report
        """
        if not self.comparison_results:
            return "No comparison results available. Run generate_comprehensive_comparison() first."
        
        report = []
        report.append("=" * 80)
        report.append("MODEL COMPARISON SUMMARY REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        summary = self.comparison_results['summary']
        report.append(f"\nTotal Models Compared: {summary['total_models']}")
        report.append(f"Traditional ML Models: {summary['traditional_models']}")
        report.append(f"Graph-based Models: {summary['graph_models']}")
        report.append(f"Best Overall Model: {summary['best_overall_model']}")
        report.append(f"Significant Differences Found: {summary['significant_differences_found']}")
        
        # Performance comparison
        if 'performance_comparison' in self.comparison_results:
            report.append("\n" + "-" * 50)
            report.append("PERFORMANCE COMPARISON")
            report.append("-" * 50)
            
            best_per_metric = self.comparison_results['performance_comparison']['best_per_metric']
            for metric, data in best_per_metric.items():
                report.append(f"Best {metric.replace('_', ' ').title()}: {data['model']} ({data['score']:.4f})")
        
        # Key insights
        if 'insights' in self.comparison_results:
            report.append("\n" + "-" * 50)
            report.append("KEY INSIGHTS")
            report.append("-" * 50)
            
            for insight in self.comparison_results['insights']:
                report.append(f"• {insight}")
        
        # Statistical significance
        if self.statistical_tests and 'pairwise_comparisons' in self.statistical_tests:
            report.append("\n" + "-" * 50)
            report.append("STATISTICAL SIGNIFICANCE (F1 Score)")
            report.append("-" * 50)
            
            f1_comparisons = self.statistical_tests['pairwise_comparisons'].get('f1_score', {})
            for comp_name, comp_data in f1_comparisons.items():
                significance = "***" if comp_data['significant'] else "n.s."
                report.append(f"{comp_data['model1']} vs {comp_data['model2']}: "
                            f"p={comp_data['p_value']:.4f} {significance}")
        
        # Feature importance
        if self.feature_importance_analysis and 'top_features' in self.feature_importance_analysis:
            report.append("\n" + "-" * 50)
            report.append("TOP PREDICTIVE FEATURES")
            report.append("-" * 50)
            
            top_features = self.feature_importance_analysis['top_features']
            for i, (feature, score) in enumerate(zip(top_features['features'][:10], 
                                                   top_features['importance_scores'][:10])):
                report.append(f"{i+1:2d}. {feature}: {score:.4f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)