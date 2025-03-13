from src.data_splitting import DataSplitter
import pandas as pd
import numpy as np
import logging
from typing import Tuple
from zenml import step

@step
def split_data_step(X: np.ndarray, y: pd.Series, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    ZenML step to split data into training and testing sets.

    :param X: Features after PCA
    :param y: Target variable
    :param test_size: Fraction of data for testing (default: 20%)
    :return: X_train, X_test, y_train, y_test
    """
    logging.info("Starting Train-Test Split Step...")
    
    # Initialize DataSplitter
    splitter = DataSplitter()

    # Perform Train-Test Split
    X_train, X_test, y_train, y_test = splitter.split_data(X, y, test_size)

    logging.info("Train-Test Split Step Completed Successfully.")
    return X_train, X_test, y_train, y_test
