"""
DataLoader class for loading and preprocessing social network data.

This module provides functionality to load JSON comments data and CSV
relationship data, with proper error handling and data validation.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

from .validators import DataValidator


class DataLoader:
    """
    Main interface for loading different social network data formats.
    
    Handles loading of:
    - JSON comment data with user interactions
    - CSV relationship data (both aggregated and raw)
    - CSV reply count data
    """
    
    def __init__(self, validate_data: bool = True):
        """
        Initialize DataLoader.
        
        Args:
            validate_data: Whether to validate loaded data automatically
        """
        self.validate_data = validate_data
        self.validator = DataValidator() if validate_data else None
        self.logger = logging.getLogger(__name__)
        
    def load_comments(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load JSON comment data and convert to DataFrame.
        
        Args:
            filepath: Path to JSON file containing comment data
            
        Returns:
            DataFrame with columns: text, author, author_id, like_count, timestamp
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is malformed or data validation fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Comment file not found: {filepath}")
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                comments_data = json.load(f)
                
            if not isinstance(comments_data, list):
                raise ValueError("JSON data must be a list of comment objects")
                
            df = pd.DataFrame(comments_data)
            
            # Ensure required columns exist
            required_columns = ['text', 'author', 'author_id', 'like_count', 'timestamp']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure like_count is numeric
            df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0)
            
            if self.validate_data:
                if not self.validator.validate_comments_data(df):
                    raise ValueError("Comment data validation failed")
                    
            self.logger.info(f"Loaded {len(df)} comments from {filepath}")
            return df
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {filepath}: {e}")
        except Exception as e:
            self.logger.error(f"Error loading comments from {filepath}: {e}")
            raise
            
    def load_relationships(self, filepath: Union[str, Path], aggregated: bool = True) -> pd.DataFrame:
        """
        Load CSV relationship data.
        
        Args:
            filepath: Path to CSV file containing relationship data
            aggregated: Whether this is aggregated (True) or raw (False) relationship data
            
        Returns:
            DataFrame with columns: Source, Destination, Weight, Source_Name, Destination_Name
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is malformed or data validation fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Relationship file not found: {filepath}")
            
        try:
            df = pd.read_csv(filepath)
            
            # Ensure required columns exist
            required_columns = ['Source', 'Destination', 'Weight', 'Source_Name', 'Destination_Name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Ensure Weight is numeric
            df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(1)
            
            # Remove rows with missing source or destination
            df = df.dropna(subset=['Source', 'Destination'])
            
            if self.validate_data:
                if not self.validator.validate_relationships_data(df, aggregated):
                    raise ValueError("Relationship data validation failed")
                    
            self.logger.info(f"Loaded {len(df)} relationships from {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading relationships from {filepath}: {e}")
            raise
            
    def load_reply_counts(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load CSV reply count data.
        
        Args:
            filepath: Path to CSV file containing reply count data
            
        Returns:
            DataFrame with columns: Author ID, Author Name, Number of Replies Received
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is malformed or data validation fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Reply counts file not found: {filepath}")
            
        try:
            df = pd.read_csv(filepath)
            
            # Standardize column names
            column_mapping = {
                'Author ID': 'author_id',
                'Author Name': 'author_name', 
                'Number of Replies Received': 'reply_count'
            }
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist after renaming
            required_columns = ['author_id', 'author_name', 'reply_count']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Ensure reply_count is numeric
            df['reply_count'] = pd.to_numeric(df['reply_count'], errors='coerce').fillna(0)
            
            # Remove rows with missing author_id
            df = df.dropna(subset=['author_id'])
            
            if self.validate_data:
                if not self.validator.validate_reply_counts_data(df):
                    raise ValueError("Reply counts data validation failed")
                    
            self.logger.info(f"Loaded reply counts for {len(df)} authors from {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading reply counts from {filepath}: {e}")
            raise
            
    def load_all_data(self, 
                     comments_file: Union[str, Path],
                     aggregated_relationships_file: Union[str, Path],
                     raw_relationships_file: Union[str, Path],
                     reply_counts_file: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Load all data files at once.
        
        Args:
            comments_file: Path to JSON comments file
            aggregated_relationships_file: Path to aggregated relationships CSV
            raw_relationships_file: Path to raw relationships CSV  
            reply_counts_file: Path to reply counts CSV
            
        Returns:
            Dictionary containing all loaded DataFrames with keys:
            'comments', 'aggregated_relationships', 'raw_relationships', 'reply_counts'
        """
        data = {}
        
        try:
            data['comments'] = self.load_comments(comments_file)
            data['aggregated_relationships'] = self.load_relationships(aggregated_relationships_file, aggregated=True)
            data['raw_relationships'] = self.load_relationships(raw_relationships_file, aggregated=False)
            data['reply_counts'] = self.load_reply_counts(reply_counts_file)
            
            self.logger.info("Successfully loaded all data files")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data files: {e}")
            raise