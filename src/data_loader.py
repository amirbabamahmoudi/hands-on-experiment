import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_data(self) -> pd.DataFrame:

        try:
            df = pd.read_excel(self.data_path)
            print(f"Successfully loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
            
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

        # Capacity is in the second column (index 1)
        y = df.iloc[:, 1]
        # Features are all columns except the capacity column and the Cell ID column
        print(df.columns[0:2])
        X = df.drop(df.columns[0:2], axis=1)
        return X, y
        
    def get_feature_names(self, df: pd.DataFrame) -> list:

        # Return all column names except the capacity column and Cell ID column
        columns = df.columns.tolist()
        return columns[2:]  # Exclude the capacity column (index 1)
        
    def explore_data(self, df: pd.DataFrame) -> None:

        #Perform initial data exploration and print summary statistics.
        print("\nDataset Overview:")
        print("-" * 50)
        print(f"Number of samples: {df.shape[0]}")
        print(f"Number of features: {df.shape[1] - 2}")  # Excluding capacity column
        print("\nFeature Statistics:")
        print(df.describe())
        print("\nMissing Values:")
        print(df.isnull().sum())
