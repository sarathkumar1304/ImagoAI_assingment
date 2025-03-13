import pandas as pd
import logging
from typing import List

# Configure Logging
logging.basicConfig(
    filename="logging.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataPreprocessing:
    """
    A class for handling common data preprocessing tasks such as handling missing values, 
    removing duplicate rows, and dropping unnecessary columns from a Pandas DataFrame.
    """

    def __init__(self):
        """Initializes the DataPreprocessing class and sets up logging."""
        logging.info("DataPreprocessing instance created.")

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Handles missing values in the given DataFrame using the specified strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame with missing values.
        strategy (str): The strategy to handle missing values. Options:
            - 'mean': Fill missing values with the mean of the column.
            - 'median': Fill missing values with the median of the column.
            - 'mode': Fill missing values with the most frequent value in the column.
            - 'drop': Remove rows with missing values.

        Returns:
        pd.DataFrame: The DataFrame after handling missing values.
        """
        try:
            if df.isnull().sum().sum() == 0:
                logging.info("No missing values found.")
                return df

            logging.info(f"Handling missing values using strategy: {strategy}")

            if strategy == "mean":
                df.fillna(df.mean(), inplace=True)
            elif strategy == "median":
                df.fillna(df.median(), inplace=True)
            elif strategy == "mode":
                df.fillna(df.mode().iloc[0], inplace=True)
            elif strategy == "drop":
                df.dropna(inplace=True)
            else:
                logging.error(f"Invalid strategy: {strategy}")
                raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', 'drop'.")

            logging.info("Missing values handled successfully.")
            return df

        except Exception as e:
            logging.exception("Error in handling missing values: %s", str(e))
            return df

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows from the given DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing duplicate rows.

        Returns:
        pd.DataFrame: The DataFrame after removing duplicates.
        """
        try:
            initial_shape = df.shape
            df.drop_duplicates(inplace=True)
            final_shape = df.shape

            logging.info(f"Dropped {initial_shape[0] - final_shape[0]} duplicate rows.")
            return df

        except Exception as e:
            logging.exception("Error in dropping duplicates: %s", str(e))
            return df

    def drop_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Drops specified columns from the given DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (List[str]): A list of column names to drop.

        Returns:
        pd.DataFrame: The DataFrame after dropping the specified columns.
        """
        try:
            logging.info(f"Columns to drop: {columns}")
            df.drop(columns=columns, axis=1, inplace=True)
            logging.info(f"Dropped columns: {columns}")
            return df

        except Exception as e:
            logging.exception("Error in dropping columns: %s", str(e))
            return df




# Example Usage
if __name__ == "__main__":
    # df = pd.read_csv("data/TASK-ML-INTERN.csv")

    # # Initialize Preprocessing Class
    # processor = DataPreprocessing()

    # # Handle Missing Values
    # df = processor.handle_missing_values(df, strategy="mean")

    # # Drop Duplicates
    # df = processor.drop_duplicates(df)

    # # Drop Unwanted Columns
    # df = processor.drop_columns(df, columns="hsi_id") 

    # print(df)
    pass
