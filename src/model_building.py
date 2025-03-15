import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xg

class ModelBuilding:
    """
    A class for building different regression models.
    Supports Linear Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, and XGBoost.
    """

    def linear_regression(self, X_train, y_train):
        """
        Trains a Linear Regression model.
        
        Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target variable.
        
        Returns:
        model (LinearRegression): Trained Linear Regression model.
        """
        logging.info("Model Linear Regression initiated")
        regressor = LinearRegression()
        model = regressor.fit(X_train, y_train)
        logging.info("Model built successfully")
        return model
    
    def decision_tree(self, X_train, y_train):
        """
        Trains a Decision Tree Regressor.
        """
        regressor = DecisionTreeRegressor()
        model = regressor.fit(X_train, y_train)
        return model
    
    def random_forest(self, X_train, y_train):
        """
        Trains a Random Forest Regressor.
        """
        regressor = RandomForestRegressor()
        model = regressor.fit(X_train, y_train)
        return model
    
    def ada_boost(self, X_train, y_train):
        """
        Trains an AdaBoost Regressor.
        """
        regressor = AdaBoostRegressor()
        model = regressor.fit(X_train, y_train)
        return model
    
    def gradient_boosting(self, X_train, y_train):
        """
        Trains a Gradient Boosting Regressor.
        """
        regressor = GradientBoostingRegressor()
        model = regressor.fit(X_train, y_train)
        return model
    
    def xgboost(self, X_train, y_train):
        """
        Trains an XGBoost Regressor.
        """
        regressor = xg.XGBRegressor()
        model = regressor.fit(X_train, y_train)
        return model
    
    def get_model(self, model_name, X_train, y_train):
        """
        Retrieves and trains the specified regression model.
        
        Parameters:
        model_name (str): Name of the model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target variable.
        
        Returns:
        Trained model instance.
        """
        if model_name == "linear_regression":
            return self.linear_regression(X_train, y_train)
        elif model_name == "decision_tree":
            return self.decision_tree(X_train, y_train)
        elif model_name == "random_forest":
            return self.random_forest(X_train, y_train)
        elif model_name == "adaboost":
            return self.ada_boost(X_train, y_train)
        elif model_name == "gradient_boosting":
            return self.gradient_boosting(X_train, y_train)
        elif model_name == "xgboost":
            return self.xgboost(X_train, y_train)
        else:
            logging.error(f"Model '{model_name}' not recognized.")
            raise ValueError(f"Model '{model_name}' not recognized.")