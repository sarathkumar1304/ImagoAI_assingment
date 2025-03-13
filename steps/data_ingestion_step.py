from zenml import step
from src.data_ingestion import DataIngestion
import pandas as pd
import logging
from typing import Optional

# Configure Logging
logging.basicConfig(
    filename="logging.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@step
def data_ingestion_step(path: str) -> pd.DataFrame:
    """
    ZenML step for data ingestion with error handling and logging.

    :param path: Path to the dataset file
    :return: Pandas DataFrame if successful, None otherwise
    """
    try:
        logging.info("Data ingestion step initiated.")
        logging.info(f"Reading data from path: {path}")

        # Initialize Data Ingestion
        data_ingest = DataIngestion()
        df = data_ingest.ingest_data(path=path)

        # Validate if data is loaded correctly
        if df is None:
            logging.error("Data ingestion failed: No data returned.")
            return None

        if df.empty:
            logging.warning("Data ingestion warning: The dataset is empty.")
            return None

        logging.info(f"Data successfully ingested. Shape: {df.shape}")

        # Quick Data Validation
        if df.isnull().sum().sum() > 0:
            logging.warning("Data contains missing values. Consider handling them.")

        return df

    except FileNotFoundError as e:
        logging.exception(f"File not found: {path}. Error: {e}")
    except ValueError as e:
        logging.exception(f"Value error in data ingestion: {e}")
    except Exception as e:
        logging.exception(f"Unexpected error during data ingestion: {e}")

    return None
