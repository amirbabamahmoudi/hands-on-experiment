import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Union

class Preprocessor:
    def __init__(self, scaler_type: str = 'standard'):
        # normalizing features with standard scalar as default
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.feature_selector = None
        
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, 
                       fit=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # Handle missing values if any
        X = self._handle_missing_values(X)
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        if y is not None:
            return X_scaled, np.array(y)
        return X_scaled
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        # For numerical data, fill missing values with median
        return X.fillna(X.median())
