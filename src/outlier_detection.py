import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List
import joblib
import logging
import os

class OutlierDetection:
    def __init__(self, df:pd.DataFrame):
        """
        Initializes the DataProcessor class with a DataFrame.

        :param df: Input DataFrame
        """
        self.df = df

    def detect_outliers_iqr(self, column:List[str])->List[str]:
        """
        Detects outliers in a given column using the IQR method.

        :param column: Column name to check for outliers
        :return: List of indices of outliers
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()

    def remove_outliers(self, columns:List[str])->pd.DataFrame:
        """
        Removes outliers from specified columns using the IQR method.

        :param columns: List of numerical columns to remove outliers from
        :return: DataFrame without outliers
        """
        for col in columns:
            outlier_indices = self.detect_outliers_iqr(col)
            self.df = self.df.drop(index=outlier_indices)

        self.df.reset_index(drop=True, inplace=True)
        
        return self.df

    def apply_minmax_scaling(self, columns:pd.DataFrame)->pd.DataFrame:
        """
        Applies Min-Max Scaling to specified columns.

        :param columns: List of numerical columns to normalize
        :return: Scaled DataFrame
        """
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        # model_dir = "models"
        # os.makedirs(model_dir, exist_ok=True)
        # model_path = os.path.join(model_dir, "min_max_scaler.pkl")
        # joblib.dump(scaler, model_path)
        # logging.info(f"Scaler saved locally at {model_path}")
        return self.df

    def apply_standard_scaling(self, columns:pd.DataFrame)->pd.DataFrame:
        """
        Applies Standard Scaling (Z-score normalization) to specified columns.

        :param columns: List of numerical columns to standardize
        :return: Scaled DataFrame
        """
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        # model_dir = "models"
        # os.makedirs(model_dir, exist_ok=True)
        # model_path = os.path.join(model_dir, "standard_scaler.pkl")
        # joblib.dump(scaler, model_path)
        # logging.info(f"Model saved locally at {model_path}")
        return self.df

# Example Usage
if __name__ == "__main__":
    
    # data ="data/TASK-ML-INTERN.csv" 
    # df = pd.read_csv(data)
    # df.drop("hsi_id",axis=1,inplace=True)
    # columns = df.drop("vomitoxin_ppb",axis=1).columns.tolist()

    # print("Original DataFrame:")
    # print(df)
    
    # # Create an instance of DataProcessor
    # processor = OutlierDetection(df)

    # # Remove Outliers
    # df_cleaned = processor.remove_outliers(columns=columns)
    # print("\nDataFrame After Outlier Removal:")
    # print(df_cleaned)

    # # Apply Min-Max Scaling
    # df_minmax_scaled = processor.apply_minmax_scaling(columns=columns)
    # print("\nDataFrame After Min-Max Scaling:")
    # print(df_minmax_scaled)

    # # Apply Standard Scaling
    # df_standard_scaled = processor.apply_standard_scaling(columns=columns)
    # print("\nDataFrame After Standard Scaling:")
    # print(df_standard_scaled)
    pass