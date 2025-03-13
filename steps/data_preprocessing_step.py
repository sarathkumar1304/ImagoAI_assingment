from src.data_preprocessing import DataPreprocessing
import logging
from zenml import step
import pandas as pd
from typing import List


@step(enable_cache=False)
def data_preprocessing_step(df:pd.DataFrame,strategy:str ="mean",columns_to_drop= List[str])->pd.DataFrame:
    try:
        logging.info("Data Preprocessing Step initiated ")
        processing = DataPreprocessing()
        logging.info(f"Implementing handle missing values to the dataframe with the {strategy}")
        df  = processing.handle_missing_values(df,strategy=strategy)
        logging.info("Handle missing values successfully")
        logging.info("Dropping the Duplicated values in the Data Frame")
        df = processing.drop_duplicates(df)
        logging.info("Drop duplicate values in the dataframe")
        logging.info(f"Dropping the unwanted column : {columns_to_drop}")
        df = processing.drop_columns(df,columns=columns_to_drop)
        logging.info(f"{columns_to_drop} dropped successfully.")

        return df
    except Exception as e:
        logging.error(f"Error occure in data splitting step {e}")
        raise e

