from zenml import step
from src.outlier_detection import OutlierDetection
import pandas as pd
import logging
from typing import Tuple

@step(enable_cache=False)
def outlier_detection_step(df:pd.DataFrame,target_column:str,scaling:str="min-max-scaling")->Tuple[pd.DataFrame,pd.Series]:
    
    """
    Perform outlier detection and scaling on the input DataFrame.

    This function initiates an outlier detection process on the given DataFrame, 
    removes the outliers from the specified columns, and then applies the specified 
    scaling method to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to process.
        target_column (str): The name of the target column that should not be 
            considered for outlier detection and scaling.
        scaling (str, optional): The scaling method to apply. Options are 
            "min-max-scaling" and "standard-scaling". Defaults to "min-max-scaling".

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the scaled DataFrame and 
        the target series.
    """

    logging.info("Outlier Detection step initatied ..")
    detection = OutlierDetection(df)
    columns = df.drop(target_column,axis=1).columns.tolist()
    cleaned_df = detection.remove_outliers(columns=columns)
    logging.info(f"cleaned_df shape : {cleaned_df.shape}")
    logging.info("Outlier detection completed and outliers removed successfully.")
    logging.info(f"Applying {scaling} to the Dataframe")
    X = cleaned_df.drop([target_column],axis=1)
    y = cleaned_df[target_column]
    if scaling =="min-max-scaling":
        X_scaled = detection.apply_minmax_scaling(columns=X.columns.tolist())
    elif scaling == "standard-scaling":
        X_scaled = detection.apply_standard_scaling(columns=X.columns.tolist())
    logging.info(f"Applied {scaling} successfully")
    return X_scaled, y
