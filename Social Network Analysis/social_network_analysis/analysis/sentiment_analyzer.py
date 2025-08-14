"""
Sentiment Analysis Module

This module provides sentiment analysis capabilities for social network text content,
integrating sentiment scores with network structure analysis.
"""

import logging
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


class SentimentAnalyzer:
    """
    Analyzes sentiment of textual content in social network data.
    
    This class provides methods to perform sentiment analysis on comments and posts,
    calculate sentiment statistics, and integrate sentiment scores with network analysis.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            logger: Optional logger instance for logging operations
        """
        self.logger = logger or logging.getLogger(__name__)
        self._sentiment_analyzer = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """Initialize the sentiment analysis backend."""
        try:
            from textblob import TextBlob
            self._sentiment_analyzer = TextBlob
            self.logger.info("TextBlob sentiment analyzer initialized")
        except ImportError:
            self.logger.warning("TextBlob not available, sentiment analysis will be limited")
            self._sentiment_analyzer = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for sentiment analysis.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text ready for sentiment analysis
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs, mentions, hashtags, and special characters
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a single text using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment polarity score (-1 to 1, where -1 is negative, 1 is positive)
        """
        if not self._sentiment_analyzer:
            return 0.0
        
        try:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return 0.0
            
            blob = self._sentiment_analyzer(cleaned_text)
            return blob.sentiment.polarity
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed for text: {e}")
            return 0.0
    
    def get_sentiment_label(self, score: float) -> str:
        """
        Convert sentiment score to categorical label.
        
        Args:
            score: Sentiment polarity score
            
        Returns:
            Sentiment label ('positive', 'negative', or 'neutral')
        """
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def perform_sentiment_analysis(self, comments_data, output_dir: str = 'outputs/sentiment_analysis') -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Perform comprehensive sentiment analysis on textual content.
        
        Args:
            comments_data: List of comments (can be dicts or strings)
            output_dir: Directory to save results
            
        Returns:
            Tuple of (sentiment_df, sentiment_stats) or (None, None) if failed
        """
        self.logger.info("Performing sentiment analysis on textual content...")
        
        if not self._sentiment_analyzer:
            self.logger.error("No sentiment analyzer available")
            return None, None
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Handle DataFrame input
            if isinstance(comments_data, pd.DataFrame):
                self.logger.info(f"Converting DataFrame with {len(comments_data)} rows to list")
                comments_list = comments_data.to_dict('records')
            else:
                comments_list = comments_data
            
            self.logger.info(f"Processing {len(comments_list)} comments for sentiment analysis")
            
            # Analyze sentiment for each comment
            sentiment_results = []
            processed_count = 0
            skipped_count = 0
            
            for i, comment in enumerate(comments_list):
                # Handle different comment formats
                if isinstance(comment, dict):
                    text_content = comment.get('text', comment.get('content', ''))
                    comment_id = comment.get('id', comment.get('comment_id', str(i)))
                    author = comment.get('author', comment.get('user', comment.get('username', 'unknown')))
                elif isinstance(comment, str):
                    text_content = comment
                    comment_id = str(i)
                    author = 'unknown'
                else:
                    self.logger.warning(f"Skipping comment {i}: unsupported format")
                    continue
                
                # Skip empty content
                if not text_content or not isinstance(text_content, str):
                    skipped_count += 1
                    continue
                
                # Analyze sentiment
                sentiment_score = self.analyze_sentiment(text_content)
                sentiment_label = self.get_sentiment_label(sentiment_score)
                
                sentiment_results.append({
                    'comment_id': comment_id,
                    'author': author,
                    'text': text_content[:200] + '...' if len(text_content) > 200 else text_content,
                    'text_length': len(text_content),
                    'sentiment_score': sentiment_score,
                    'sentiment_label': sentiment_label
                })
                processed_count += 1
            
            self.logger.info(f"Processed {processed_count} comments, skipped {skipped_count} comments")
            
            if not sentiment_results:
                self.logger.warning("No valid comments found for sentiment analysis")
                return None, None
            
            # Create sentiment DataFrame
            sentiment_df = pd.DataFrame(sentiment_results)
            
            # Save detailed results
            sentiment_df.to_csv(f'{output_dir}/sentiment_scores.csv', index=False)
            
            # Calculate comprehensive statistics
            sentiment_stats = self._calculate_sentiment_statistics(sentiment_df)
            
            # Save statistics
            with open(f'{output_dir}/sentiment_statistics.json', 'w') as f:
                json.dump(sentiment_stats, f, indent=2)
            
            # Generate sentiment distribution analysis
            self._generate_sentiment_distribution_analysis(sentiment_df, output_dir)
            
            self.logger.info(f"Sentiment analysis completed: {len(sentiment_results)} comments analyzed")
            self.logger.info(f"Average sentiment: {sentiment_stats['avg_sentiment']:.3f}")
            self.logger.info(f"Sentiment distribution - Positive: {sentiment_stats['positive_percentage']:.1f}%, "
                           f"Negative: {sentiment_stats['negative_percentage']:.1f}%, "
                           f"Neutral: {sentiment_stats['neutral_percentage']:.1f}%")
            
            return sentiment_df, sentiment_stats
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return None, None
    
    def _calculate_sentiment_statistics(self, sentiment_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive sentiment statistics."""
        try:
            total_comments = len(sentiment_df)
            positive_comments = len(sentiment_df[sentiment_df['sentiment_score'] > 0.1])
            negative_comments = len(sentiment_df[sentiment_df['sentiment_score'] < -0.1])
            neutral_comments = total_comments - positive_comments - negative_comments
            
            sentiment_stats = {
                'total_comments': total_comments,
                'positive_comments': positive_comments,
                'negative_comments': negative_comments,
                'neutral_comments': neutral_comments,
                'positive_percentage': (positive_comments / total_comments) * 100 if total_comments > 0 else 0,
                'negative_percentage': (negative_comments / total_comments) * 100 if total_comments > 0 else 0,
                'neutral_percentage': (neutral_comments / total_comments) * 100 if total_comments > 0 else 0,
                'avg_sentiment': float(sentiment_df['sentiment_score'].mean()),
                'sentiment_std': float(sentiment_df['sentiment_score'].std()),
                'sentiment_median': float(sentiment_df['sentiment_score'].median()),
                'sentiment_min': float(sentiment_df['sentiment_score'].min()),
                'sentiment_max': float(sentiment_df['sentiment_score'].max()),
                'avg_text_length': float(sentiment_df['text_length'].mean()),
                'sentiment_by_length': self._analyze_sentiment_by_length(sentiment_df)
            }
            
            return sentiment_stats
            
        except Exception as e:
            self.logger.warning(f"Error calculating sentiment statistics: {e}")
            return {
                'total_comments': len(sentiment_df),
                'positive_comments': 0,
                'negative_comments': 0,
                'neutral_comments': 0,
                'positive_percentage': 0.0,
                'negative_percentage': 0.0,
                'neutral_percentage': 0.0,
                'avg_sentiment': 0.0,
                'sentiment_std': 0.0,
                'sentiment_median': 0.0,
                'sentiment_min': 0.0,
                'sentiment_max': 0.0,
                'avg_text_length': 0.0,
                'sentiment_by_length': {}
            }
    
    def _analyze_sentiment_by_length(self, sentiment_df: pd.DataFrame) -> Dict:
        """Analyze sentiment patterns by text length."""
        try:
            # Create length bins
            sentiment_df['length_bin'] = pd.cut(sentiment_df['text_length'], 
                                              bins=[0, 50, 100, 200, float('inf')], 
                                              labels=['short', 'medium', 'long', 'very_long'])
            
            length_analysis = {}
            for length_bin in sentiment_df['length_bin'].unique():
                if pd.isna(length_bin):
                    continue
                    
                bin_data = sentiment_df[sentiment_df['length_bin'] == length_bin]
                length_analysis[str(length_bin)] = {
                    'count': len(bin_data),
                    'avg_sentiment': float(bin_data['sentiment_score'].mean()),
                    'positive_ratio': len(bin_data[bin_data['sentiment_score'] > 0.1]) / len(bin_data)
                }
            
            return length_analysis
            
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment by length: {e}")
            return {}
    
    def _generate_sentiment_distribution_analysis(self, sentiment_df: pd.DataFrame, output_dir: str):
        """Generate additional sentiment distribution analysis."""
        try:
            # Author sentiment analysis
            if 'author' in sentiment_df.columns:
                author_sentiment = sentiment_df.groupby('author').agg({
                    'sentiment_score': ['mean', 'count', 'std']
                }).round(3)
                
                author_sentiment.columns = ['avg_sentiment', 'comment_count', 'sentiment_std']
                author_sentiment = author_sentiment.reset_index()
                author_sentiment.to_csv(f'{output_dir}/author_sentiment_analysis.csv', index=False)
            
            # Sentiment score distribution
            sentiment_distribution = {
                'very_negative': len(sentiment_df[sentiment_df['sentiment_score'] <= -0.5]),
                'negative': len(sentiment_df[(sentiment_df['sentiment_score'] > -0.5) & (sentiment_df['sentiment_score'] <= -0.1)]),
                'neutral': len(sentiment_df[(sentiment_df['sentiment_score'] > -0.1) & (sentiment_df['sentiment_score'] <= 0.1)]),
                'positive': len(sentiment_df[(sentiment_df['sentiment_score'] > 0.1) & (sentiment_df['sentiment_score'] <= 0.5)]),
                'very_positive': len(sentiment_df[sentiment_df['sentiment_score'] > 0.5])
            }
            
            with open(f'{output_dir}/sentiment_distribution.json', 'w') as f:
                json.dump(sentiment_distribution, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Error generating sentiment distribution analysis: {e}")
    
    def integrate_with_network(self, sentiment_df: pd.DataFrame, graph, centrality_scores: Dict) -> Dict:
        """
        Integrate sentiment analysis with network structure analysis.
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
            graph: NetworkX graph
            centrality_scores: Dictionary of centrality scores
            
        Returns:
            Dictionary with integrated analysis results
        """
        try:
            # Create author-level sentiment aggregation
            author_sentiment = sentiment_df.groupby('author').agg({
                'sentiment_score': ['mean', 'count', 'std']
            }).round(3)
            
            author_sentiment.columns = ['avg_sentiment', 'comment_count', 'sentiment_std']
            author_sentiment = author_sentiment.reset_index()
            
            # Integrate with centrality scores
            integrated_analysis = {}
            
            for author in author_sentiment['author']:
                if author in graph.nodes():
                    author_data = author_sentiment[author_sentiment['author'] == author].iloc[0]
                    
                    integrated_analysis[author] = {
                        'avg_sentiment': author_data['avg_sentiment'],
                        'comment_count': author_data['comment_count'],
                        'sentiment_std': author_data['sentiment_std'],
                        'degree_centrality': centrality_scores.get('degree', {}).get(author, 0),
                        'betweenness_centrality': centrality_scores.get('betweenness', {}).get(author, 0),
                        'closeness_centrality': centrality_scores.get('closeness', {}).get(author, 0)
                    }
            
            # Calculate correlations between sentiment and centrality
            if integrated_analysis:
                sentiment_values = [data['avg_sentiment'] for data in integrated_analysis.values()]
                degree_values = [data['degree_centrality'] for data in integrated_analysis.values()]
                betweenness_values = [data['betweenness_centrality'] for data in integrated_analysis.values()]
                
                correlations = {
                    'sentiment_degree_correlation': np.corrcoef(sentiment_values, degree_values)[0, 1] if len(sentiment_values) > 1 else 0,
                    'sentiment_betweenness_correlation': np.corrcoef(sentiment_values, betweenness_values)[0, 1] if len(sentiment_values) > 1 else 0
                }
            else:
                correlations = {'sentiment_degree_correlation': 0, 'sentiment_betweenness_correlation': 0}
            
            return {
                'integrated_analysis': integrated_analysis,
                'correlations': correlations,
                'summary': {
                    'total_authors_analyzed': len(integrated_analysis),
                    'avg_network_sentiment': np.mean(sentiment_values) if sentiment_values else 0,
                    'sentiment_centrality_relationship': 'positive' if correlations['sentiment_degree_correlation'] > 0.1 else 'negative' if correlations['sentiment_degree_correlation'] < -0.1 else 'neutral'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error integrating sentiment with network analysis: {e}")
            return {}