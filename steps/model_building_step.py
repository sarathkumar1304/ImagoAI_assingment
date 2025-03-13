# from src.model_building import ModelBuilding
# import os
# import logging
# import mlflow
# import joblib
# import pandas as pd
# from sklearn.pipeline import Pipeline

# from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
# from zenml.client import Client
# from zenml import ArtifactConfig, step, save_artifact
# from zenml.enums import ArtifactType
# from typing import Annotated
# import numpy as np

# # Get the active experiment tracker from ZenML
# experiment_tracker = Client().active_stack.experiment_tracker

# @step(enable_cache=False, experiment_tracker=experiment_tracker.name)
# def model_building_step(
#     model_name: str, 
#     X_train: np.ndarray, 
#     y_train: pd.Series
# ) -> Annotated[Pipeline, ArtifactConfig(artifact_type=ArtifactType.MODEL)]:
#     """
#     ZenML step to build and train a model, log to MLflow, and save it as an artifact.

#     :param model_name: Name of the model
#     :param X_train: Training features
#     :param y_train: Training labels
#     :return: Trained pipeline model
#     """
#     logging.info(f"Model Building Step Initiated for: {model_name}")
    
#     try:
#         # Ensure an MLflow run is active
#         if not mlflow.active_run():
#             mlflow.start_run(run_name=model_name, nested=True,log_system_metrics=True)
        
#         # Enable MLflow autologging
#         mlflow.sklearn.autolog()

#         #  Initialize ModelBuilding
#         model_builder = ModelBuilding()
#         model = model_builder.get_model(model_name, X_train, y_train)
#         logging.info(f"Model '{model_name}' has been successfully created.")

#         # Create the pipeline and train
#         pipeline = Pipeline(steps=[("model", model)])
#         y_train = np.array(y_train).ravel() 
#         pipeline.fit(X_train, y_train)
#         logging.info("Model training completed successfully.")

#         # Save model using MLflow
#         mlflow.sklearn.log_model(pipeline, artifact_path="model_pipeline")

#         # Save model as ZenML artifact
#         save_artifact(pipeline, name="trained_model", artifact_type=ArtifactType.MODEL)

#         # Save model locally for backup
#         model_dir = "models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, "model.pkl")
#         joblib.dump(pipeline, model_path)
#         logging.info(f"Model saved locally at {model_path}")

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         raise e

#     finally:
#         mlflow.end_run()

#     return pipeline

# # # zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow
# # # zenml stack register custom_stack -e mlflow_experiment_tracker ... --set

from src.model_building import ModelBuilding
import os
import logging
import mlflow
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.client import Client
from zenml import ArtifactConfig, step, save_artifact
from zenml.enums import ArtifactType
from typing import Annotated

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def model_building_step(
    model_name: str, 
    X_train: np.ndarray, 
    y_train: pd.Series,
    tune_hyperparameters: bool = False  # Enable or disable hyperparameter tuning
) -> Annotated[Pipeline, ArtifactConfig(artifact_type=ArtifactType.MODEL)]:
    """
    ZenML step to build and train a model, perform optional hyperparameter tuning, 
    log to MLflow, and save it as an artifact.

    :param model_name: Name of the model
    :param X_train: Training features
    :param y_train: Training labels
    :param tune_hyperparameters: Boolean flag for enabling hyperparameter tuning
    :return: Trained pipeline model
    """
    logging.info(f"Model Building Step Initiated for: {model_name}")

    try:
        # Ensure an MLflow run is active
        if not mlflow.active_run():
            mlflow.start_run(run_name=model_name, nested=True, log_system_metrics=True)

        # Enable MLflow autologging
        mlflow.sklearn.autolog()

        # Initialize ModelBuilding
        model_builder = ModelBuilding()
        model = model_builder.get_model(model_name, X_train, y_train)

        if tune_hyperparameters:
            logging.info(f"Performing hyperparameter tuning for {model_name}.")

            # Define hyperparameter grids for different models
            param_grids = {

                "linear_regression": {},  # No hyperparameters for basic Linear Regression

                "decision_tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"],
                },

                "random_forest": {
                    "n_estimators": [50, 100, 200, 500],
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"],
                    "bootstrap": [True, False],
                },

                "adaboost": {
                    "n_estimators": [50, 100, 200, 500],
                    "learning_rate": [0.001, 0.01, 0.1, 1],
                    "loss": ["linear", "square", "exponential"],  # For regression tasks
                },

                "gradient_boosting": {
                    "n_estimators": [50, 100, 200, 500],
                    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.5],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"],
                    "subsample": [0.8, 0.9, 1.0],  # Helps prevent overfitting
                },

                "xgboost": {  # If you decide to use XGBoost
                    "n_estimators": [50, 100, 200, 500],
                    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.5],
                    "max_depth": [3, 5, 10],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "gamma": [0, 0.1, 0.2, 0.3],  # Regularization
                    "lambda": [0, 0.01, 0.1, 1],  # L2 Regularization
                }
            }


            param_grid = param_grids.get(model_name, {})

            if param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                logging.info(f"Best Parameters for {model_name}: {grid_search.best_params_}")
            else:
                logging.info(f"No hyperparameter tuning performed for {model_name} (no hyperparameters available).")

        logging.info(f"Model '{model_name}' has been successfully created.")

        # Create the pipeline and train
        pipeline = Pipeline(steps=[("model", model)])
        y_train = np.array(y_train).ravel()
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed successfully.")

        # Save model using MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model_pipeline")

        # Save model as ZenML artifact
        save_artifact(pipeline, name="trained_model", artifact_type=ArtifactType.MODEL)

        # Save model locally for backup
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name+"_model.pkl")
        joblib.dump(pipeline, model_path)
        logging.info(f"Model saved locally at {model_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e

    finally:
        mlflow.end_run()

    return pipeline
