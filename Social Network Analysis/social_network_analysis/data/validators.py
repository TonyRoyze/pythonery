"""
Data validation utilities for social network analysis.

This module provides comprehensive validation for different data types
to ensure data quality and consistency.
"""

import pandas as pd
import numpy as np
from typing import List, Set, Optional
import logging
import re


class DataValidator:
    """
    Validates social network data to ensure quality and consistency.
    
    Provides validation methods for comments, relationships, and reply count data.
    """
    
    def __init__(self):
        """Initialize DataValidator with logging."""
        self.logger = logging.getLogger(__name__)
        
    def validate_comments_data(self, df: pd.DataFrame) -> bool:
        """
        Validate comments DataFrame.
        
        Args:
            df: DataFrame containing comment data
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check required columns
            required_columns = ['text', 'author', 'author_id', 'like_count', 'timestamp']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns in comments data")
                return False
                
            # Check for empty DataFrame
            if df.empty:
                self.logger.warning("Comments DataFrame is empty")
                return True
                
            # Validate text column
            if df['text'].isnull().any():
                self.logger.warning("Found null values in text column")
                
            # Validate author and author_id
            if df['author'].isnull().any() or df['author_id'].isnull().any():
                self.logger.error("Found null values in author or author_id columns")
                return False
                
            # Check for duplicate author_id entries (should be unique per comment)
            # Note: Same author can have multiple comments, so this checks for exact duplicates
            duplicate_rows = df.duplicated(subset=['author_id', 'text', 'timestamp'])
            if duplicate_rows.any():
                self.logger.warning(f"Found {duplicate_rows.sum()} duplicate comment entries")
                
            # Validate like_count is non-negative
            if (df['like_count'] < 0).any():
                self.logger.error("Found negative like_count values")
                return False
                
            # Validate timestamp format
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                self.logger.error("Timestamp column is not datetime type")
                return False
                
            # Check for reasonable timestamp range (not too far in future/past)
            current_year = pd.Timestamp.now().year
            min_year = df['timestamp'].dt.year.min()
            max_year = df['timestamp'].dt.year.max()
            
            if min_year < 2000 or max_year > current_year + 1:
                self.logger.warning(f"Unusual timestamp range: {min_year} to {max_year}")
                
            self.logger.info("Comments data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating comments data: {e}")
            return False
            
    def validate_relationships_data(self, df: pd.DataFrame, aggregated: bool = True) -> bool:
        """
        Validate relationships DataFrame.
        
        Args:
            df: DataFrame containing relationship data
            aggregated: Whether this is aggregated or raw relationship data
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check required columns
            required_columns = ['Source', 'Destination', 'Weight', 'Source_Name', 'Destination_Name']
            if not all(col in df.columns for col in required_columns):
                self.logger.error("Missing required columns in relationships data")
                return False
                
            # Check for empty DataFrame
            if df.empty:
                self.logger.warning("Relationships DataFrame is empty")
                return True
                
            # Validate Source and Destination are not null
            if df['Source'].isnull().any() or df['Destination'].isnull().any():
                self.logger.error("Found null values in Source or Destination columns")
                return False
                
            # Check for self-loops (Source == Destination)
            self_loops = df['Source'] == df['Destination']
            if self_loops.any():
                self.logger.warning(f"Found {self_loops.sum()} self-loop relationships")
                
            # Validate Weight is positive
            if (df['Weight'] <= 0).any():
                self.logger.error("Found non-positive Weight values")
                return False
                
            # For aggregated data, weights should typically be > 1
            if aggregated and (df['Weight'] == 1).all():
                self.logger.warning("All weights are 1 in aggregated data - this may indicate raw data")
                
            # For raw data, weights should typically be 1
            if not aggregated and not (df['Weight'] == 1).all():
                self.logger.warning("Non-unit weights found in raw data")
                
            # Check for consistency between ID and Name columns
            source_mapping = df.groupby('Source')['Source_Name'].nunique()
            dest_mapping = df.groupby('Destination')['Destination_Name'].nunique()
            
            if (source_mapping > 1).any():
                inconsistent_sources = source_mapping[source_mapping > 1].index.tolist()
                self.logger.warning(f"Inconsistent Source_Name mappings for IDs: {inconsistent_sources[:5]}")
                
            if (dest_mapping > 1).any():
                inconsistent_dests = dest_mapping[dest_mapping > 1].index.tolist()
                self.logger.warning(f"Inconsistent Destination_Name mappings for IDs: {inconsistent_dests[:5]}")
                
            # Check for duplicate relationships
            duplicate_pairs = df.duplicated(subset=['Source', 'Destination'])
            if duplicate_pairs.any():
                self.logger.warning(f"Found {duplicate_pairs.sum()} duplicate relationship pairs")
                
            self.logger.info(f"Relationships data validation passed ({'aggregated' if aggregated else 'raw'})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating relationships data: {e}")
            return False
            
    def validate_reply_counts_data(self, df: pd.DataFrame) -> bool:
        """
        Validate reply counts DataFrame.
        
        Args:
            df: DataFrame containing reply count data
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check required columns
            required_columns = ['author_id', 'author_name', 'reply_count']
            if not all(col in df.columns for col in required_columns):
                self.logger.error("Missing required columns in reply counts data")
                return False
                
            # Check for empty DataFrame
            if df.empty:
                self.logger.warning("Reply counts DataFrame is empty")
                return True
                
            # Validate author_id is not null
            if df['author_id'].isnull().any():
                self.logger.error("Found null values in author_id column")
                return False
                
            # Check for duplicate author_ids
            duplicate_authors = df['author_id'].duplicated()
            if duplicate_authors.any():
                self.logger.warning(f"Found {duplicate_authors.sum()} duplicate author_id entries")
                
            # Validate reply_count is non-negative integer
            if (df['reply_count'] < 0).any():
                self.logger.error("Found negative reply_count values")
                return False
                
            if not df['reply_count'].dtype.kind in 'iuf':  # integer, unsigned, float
                self.logger.error("reply_count column is not numeric")
                return False
                
            # Check for reasonable reply count range
            max_replies = df['reply_count'].max()
            if max_replies > 10000:  # Arbitrary threshold for very high reply counts
                self.logger.warning(f"Very high maximum reply count: {max_replies}")
                
            # Validate author_name format (should start with @)
            if df['author_name'].notna().any():
                invalid_names = ~df['author_name'].str.startswith('@', na=False)
                if invalid_names.any():
                    self.logger.warning(f"Found {invalid_names.sum()} author names not starting with @")
                    
            self.logger.info("Reply counts data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating reply counts data: {e}")
            return False
            
    def validate_data_consistency(self, 
                                comments_df: pd.DataFrame,
                                relationships_df: pd.DataFrame,
                                reply_counts_df: pd.DataFrame) -> bool:
        """
        Validate consistency across different data sources.
        
        Args:
            comments_df: Comments DataFrame
            relationships_df: Relationships DataFrame  
            reply_counts_df: Reply counts DataFrame
            
        Returns:
            True if cross-validation passes, False otherwise
        """
        try:
            # Get unique author IDs from each dataset
            comment_authors = set(comments_df['author_id'].unique())
            relationship_authors = set(relationships_df['Source'].unique()) | set(relationships_df['Destination'].unique())
            reply_count_authors = set(reply_counts_df['author_id'].unique())
            
            # Check overlap between datasets
            comment_relationship_overlap = len(comment_authors & relationship_authors)
            comment_reply_overlap = len(comment_authors & reply_count_authors)
            relationship_reply_overlap = len(relationship_authors & reply_count_authors)
            
            self.logger.info(f"Author overlap - Comments/Relationships: {comment_relationship_overlap}")
            self.logger.info(f"Author overlap - Comments/Reply counts: {comment_reply_overlap}")
            self.logger.info(f"Author overlap - Relationships/Reply counts: {relationship_reply_overlap}")
            
            # Warn if there's very little overlap
            if comment_relationship_overlap < min(len(comment_authors), len(relationship_authors)) * 0.1:
                self.logger.warning("Very low overlap between comments and relationships data")
                
            # Check for consistency in author names
            if not comments_df.empty and not reply_counts_df.empty:
                # Create mapping from comments data
                comment_name_mapping = dict(zip(comments_df['author_id'], comments_df['author']))
                
                # Check consistency with reply counts
                for _, row in reply_counts_df.iterrows():
                    author_id = row['author_id']
                    if author_id in comment_name_mapping:
                        comment_name = comment_name_mapping[author_id]
                        reply_name = row['author_name']
                        if comment_name != reply_name:
                            self.logger.warning(f"Name mismatch for {author_id}: '{comment_name}' vs '{reply_name}'")
                            
            self.logger.info("Cross-dataset validation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in cross-dataset validation: {e}")
            return False