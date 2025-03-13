from pipelines.training_pipeline import training_pipeline
import os
import click
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

# Run the pipeline
def run_pipeline():
    training_pipeline()
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )
if __name__ == '__main__':  
    run_pipeline()