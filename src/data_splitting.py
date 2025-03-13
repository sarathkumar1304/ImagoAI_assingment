import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple

class DataSplitter:
    def split_data(self, X: np.ndarray, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Splits the dataset into train and test sets.

        :param X: Features after PCA
        :param y: Target variable
        :param test_size: Fraction of data to be used for testing
        :param random_state: Random seed for reproducibility
        :return: X_train, X_test, y_train, y_test
        """
        logging.info("Starting train-test split...")
        
        # Perform Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Logging shape of the datasets
        logging.info(f"Train shape: X_train={X_train.shape}, y_train={y_train.shape}")
        logging.info(f"Test shape: X_test={X_test.shape}, y_test={y_test.shape}")
        logging.info("Train-test split completed successfully.")

        return X_train, X_test, y_train, y_test



class DataSplittingForCNN:
    def split_data(self,df:pd.DataFrame,target_column:str,test_size:float=0.2,random_state:int=42)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        logging.info("Data Splitting Started")
        X = df.drop(target_column,axis=1)
        y = df[target_column]
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

# Train-Test Split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    