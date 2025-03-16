

# # # zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow
# # # zenml stack register custom_stack -e mlflow_experiment_tracker ... --set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import mlflow
import joblib
import optuna
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from zenml.client import Client
from zenml import ArtifactConfig, step, save_artifact
from zenml.enums import ArtifactType
from typing import Annotated
from zenml import Model

experiment_tracker = Client().active_stack.experiment_tracker
model = Model(
    name= "ImagoAI-ML-Model",
    description = 'Model for predicting vomitoxin levels in corn samples'
)
@step(enable_cache=False, experiment_tracker=experiment_tracker.name,model=model)
def model_building_step(
    model_name: str, 
    X_train: np.ndarray|pd.DataFrame, 
    y_train: np.ndarray|pd.Series,
    X_test :np.ndarray|pd.DataFrame,
    y_test:np.ndarray|pd.Series,
    tune_hyperparameters: bool = True,
    n_trials:int=30
) -> Annotated[Pipeline, ArtifactConfig(artifact_type=ArtifactType.MODEL)]:
    """
    ZenML step to build and train a model with Optuna hyperparameter tuning.
    """
    logging.info(f"Model Building Step Initiated for: {model_name}")
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    model_mapping = {
        "linear_regression": LinearRegression,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "adaboost": AdaBoostRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "xgboost": xgb.XGBRegressor,
        "svm": SVR
    }
    # mlflow.set_experiment(model_name)
    if not mlflow.active_run():
        mlflow.start_run(run_name=model_name, nested=True, log_system_metrics=True)
    
    # Enable MLflow autologging
    mlflow.sklearn.autolog()
    
    def objective(trial):
    # Define the parameter grid ONLY for the selected model
        if model_name == "decision_tree":
            param_grid = {
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
        elif model_name == "linear_regression":
            # param_grid = {}
            return 0
        elif model_name == "random_forest":
            param_grid = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        elif model_name == "adaboost":
            param_grid = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
                'estimator': DecisionTreeRegressor()
            }
        elif model_name == "gradient_boosting":
            param_grid = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error'])
            }
        elif model_name == "xgboost":
            param_grid =  {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Added correct param
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),  # Regularization
                'lambda': trial.suggest_float('lambda', 0.0, 10.0)  # L2 Regularization
            }
        elif model_name == "svm":
            param_grid = {
                'C': trial.suggest_float('C', 0.1, 100.0),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'degree': trial.suggest_int('degree', 2, 5),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            }
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        # Train model with selected parameters
        model = model_mapping[model_name](**param_grid)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Return R² score for Optuna to optimize
        return r2_score(y_test, y_pred)

    if tune_hyperparameters:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        logging.info(f"Best parameters for {model_name}: {best_params}")
    else:
        best_params = {}

    # valid_params = {k: v for k, v in best_params.items() if v is not None}
    try:

    #  Initialize the final model with only valid parameters
        final_model = model_mapping[model_name](**best_params)
        # final_model = model_mapping[model_name](**best_params)
        final_model.fit(X_train, y_train)

        pipeline = Pipeline(steps=[("model", final_model)])
        pipeline.fit(X_train, y_train)

        mlflow.sklearn.log_model(pipeline, artifact_path="model_pipeline")
        save_artifact(pipeline, name="trained_model", artifact_type=ArtifactType.MODEL)
        
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(pipeline, model_path)
        logging.info(f"Model saved locally at {model_path}")

        # return pipeline
    except Exception as e:
        logging.error(f"An error occurred during model training: {str(e)}")
        # mlflow.end_run(status=mlflow.entities.RunStatus.FAILED)
        raise e
    finally:
        # end the mlflow run
        mlflow.end_run()

    return pipeline
