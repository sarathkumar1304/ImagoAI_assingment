import json
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step
def predictor(service: MLFlowDeploymentService, input_data: str) -> np.ndarray:
    """
    Loads PCA-transformed input data and makes predictions using the MLflow model.
    
    Args:
        service (MLFlowDeploymentService): MLflow deployment service for inference.
        input_data (str): JSON string of PCA-transformed input data.
        
    Returns:
        np.ndarray: Model predictions.
    """
    service.start(timeout=60)

    #  Parse input JSON (using orient="records" format)
    data = json.loads(input_data)  # Convert JSON string to list of dicts

    #  Convert to DataFrame
    df_pca = pd.DataFrame(data)

    #  Expected PCA column names
    expected_columns = ['PC1', 'PC2']

    #  Ensure only PCA-transformed features are present
    df_pca = df_pca[expected_columns]

    #  Convert DataFrame to NumPy array for prediction
    data_array = df_pca.to_numpy()

    #  Make predictions using the deployed MLflow model
    prediction = service.predict(data_array)
    print(prediction)

    return prediction
