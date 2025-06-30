"""
Data Loader Module for Google Play Store EDA
- Loads and explores app and review data efficiently.
- Designed for use on both CPU and GPU (if available).
- Uses chunking and vectorized operations for large datasets.
- Compatible with pandas, numpy, and can be extended for GPU with cuDF or RAPIDS.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Optional
warnings.filterwarnings('ignore')

class DataLoader:
    """Class for loading and exploring Google Play Store datasets."""
    
    def __init__(self, data_path="data"):
        """
        Initialize DataLoader
        
        Parameters:
        -----------
        data_path : str
            Path to the data directory
        """
        self.data_path = Path(data_path)
        self.apps_df = None
        self.reviews_df = None
        
    def load_csv(self, path: str, chunksize: Optional[int] = None) -> pd.DataFrame:
        """
        Load a CSV file efficiently, with optional chunking for large files.
        Args:
            path (str): Path to the CSV file.
            chunksize (Optional[int]): Number of rows per chunk (for large files).
        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        if chunksize:
            # For very large files, read in chunks and concatenate
            chunks = pd.read_csv(path, chunksize=chunksize)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path)
        return df

    def basic_info(self, df: pd.DataFrame) -> None:
        """
        Print basic information about the dataframe.
        Args:
            df (pd.DataFrame): Dataframe to describe.
        """
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.info())
        print(df.describe(include='all'))

    def missing_summary(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a summary of missing values in the dataframe.
        Args:
            df (pd.DataFrame): Dataframe to check.
        Returns:
            pd.Series: Missing value counts per column.
        """
        return df.isnull().sum()

    def load_apps_data(self, filename="googleplaystore.csv"):
        """
        Load Google Play Store apps dataset
        
        Parameters:
        -----------
        filename : str
            Name of the apps CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded apps dataset
        """
        file_path = self.data_path / filename
        try:
            self.apps_df = pd.read_csv(file_path)
            print(f"Successfully loaded apps data: {self.apps_df.shape[0]} rows, {self.apps_df.shape[1]} columns")
            return self.apps_df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return None
        except Exception as e:
            print(f"Error loading apps data: {e}")
            return None
    
    def load_reviews_data(self, filename="googleplaystore_user_reviews.csv"):
        """
        Load Google Play Store user reviews dataset
        
        Parameters:
        -----------
        filename : str
            Name of the reviews CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded reviews dataset
        """
        file_path = self.data_path / filename
        try:
            self.reviews_df = pd.read_csv(file_path)
            print(f"Successfully loaded reviews data: {self.reviews_df.shape[0]} rows, {self.reviews_df.shape[1]} columns")
            return self.reviews_df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return None
        except Exception as e:
            print(f"Error loading reviews data: {e}")
            return None
    
    def load_all_data(self):
        """
        Load both datasets
        
        Returns:
        --------
        tuple
            (apps_df, reviews_df)
        """
        apps_df = self.load_apps_data()
        reviews_df = self.load_reviews_data()
        return apps_df, reviews_df
    
    def get_basic_info(self, df, dataset_name=""):
        """
        Get basic information about a dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to analyze
        dataset_name : str
            Name of the dataset for display purposes
            
        Returns:
        --------
        dict
            Basic information about the dataset
        """
        if df is None:
            return None
            
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        print(f"\nBasic Information - {dataset_name}")
        print(f"Shape: {info['shape']}")
        print(f"Memory Usage: {info['memory_usage'] / 1024**2:.2f} MB")
        print(f"Duplicates: {info['duplicates']}")
        print(f"\nColumns: {info['columns']}")
        
        return info
    
    def get_numeric_summary(self, df):
        """
        Get summary statistics for numeric columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to analyze
            
        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        if df is None:
            return None
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary = df[numeric_cols].describe()
            print(f"\nNumeric Summary Statistics")
            print(summary)
            return summary
        else:
            print("No numeric columns found")
            return None
    
    def get_categorical_summary(self, df, max_categories=10):
        """
        Get summary statistics for categorical columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to analyze
        max_categories : int
            Maximum number of categories to display
            
        Returns:
        --------
        dict
            Summary statistics for categorical columns
        """
        if df is None:
            return None
            
        categorical_cols = df.select_dtypes(include=['object']).columns
        summary = {}
        
        print(f"\nCategorical Summary Statistics")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"\n{col}:")
            print(f"  Unique values: {unique_count}")
            if unique_count <= max_categories:
                value_counts = df[col].value_counts()
                print(f"  Value counts:")
                for value, count in value_counts.items():
                    print(f"    {value}: {count}")
            else:
                print(f"  Top {max_categories} values:")
                top_values = df[col].value_counts().head(max_categories)
                for value, count in top_values.items():
                    print(f"    {value}: {count}")
            
            summary[col] = {
                'unique_count': unique_count,
                'value_counts': df[col].value_counts().to_dict()
            }
        
        return summary
    
    def explore_datasets(self):
        """
        Comprehensive exploration of both datasets
        
        Returns:
        --------
        dict
            Exploration results
        """
        print("Exploring Google Play Store Datasets")
        print("=" * 50)
        
        # Load data if not already loaded
        if self.apps_df is None:
            self.load_apps_data()
        if self.reviews_df is None:
            self.load_reviews_data()
        
        # Basic info
        apps_info = self.get_basic_info(self.apps_df, "Apps Dataset")
        reviews_info = self.get_basic_info(self.reviews_df, "Reviews Dataset")
        
        # Numeric summaries
        apps_numeric = self.get_numeric_summary(self.apps_df)
        reviews_numeric = self.get_numeric_summary(self.reviews_df)
        
        # Categorical summaries
        apps_categorical = self.get_categorical_summary(self.apps_df)
        reviews_categorical = self.get_categorical_summary(self.reviews_df)
        
        return {
            'apps_info': apps_info,
            'reviews_info': reviews_info,
            'apps_numeric': apps_numeric,
            'reviews_numeric': reviews_numeric,
            'apps_categorical': apps_categorical,
            'reviews_categorical': reviews_categorical
        }

# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    loader.explore_datasets() 