import os
import logging
import pandas as pd

# Configure Logging
logging.basicConfig(
    filename="logging.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataIngestion:
    def __init__(self):
        """
        This is the constructor of DataIngestion class.
        It gets called whenever an instance of DataIngestion class is created.
        """
        logging.info("DataIngestion instance created.")

    def ingest_data(self, path: str):
        """
        Ingests data from a CSV file.

        Args:
            path (str): The path to the CSV file to ingest.

        Returns:
            pd.DataFrame: The ingested data as a Pandas DataFrame.

        Raises:
            FileNotFoundError: If the file at the given path does not exist.
            Exception: If any unexpected error occurs during data ingestion.
        """
        try:
            if not os.path.exists(path):
                logging.error("File not found: %s", path)
                raise FileNotFoundError(f"File not found: {path}")

            logging.info("Read the CVS from the path : %s", path)
            df = pd.read_csv(path)
            return df

        except FileNotFoundError as e:
            logging.exception("Error: %s", str(e))
            raise e
        except Exception as e:
            logging.exception("Unexpected error during data ingestion: %s", str(e))
            raise e

# Example Usage
if __name__ == "__main__":
    # ingestion = DataIngestion()
    # file_path = "data/TASK-ML-INTERN.csv"  # Replace with actual path
    # result = ingestion.ingest_data(file_path)
    # print(result.head())
    pass