from src.model_evaluation import ModelEvaluation
from zenml import step
import mlflow
import logging

@step(enable_cache=False)
def model_evaluation_step(model,X_test,y_test):
    """
    Evaluates a model using regression metrics.

    :param model: Trained model
    :param X_test: Test data
    :param y_test: Test labels
    :return: Mean Absolute Error, Root Mean Squared Error, R-Squared score
    """
    logging.info("Model Evaluation started")
    evaluate = ModelEvaluation()
    results = evaluate.evaluate_model(model= model,X_test = X_test,y_test = y_test)
    logging.info("Model Evaluation completed")
    
    return results

