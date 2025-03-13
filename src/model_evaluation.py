# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# import pandas as pd
# import logging
# import mlflow

# class ModelEvaluation:

#     def evaluate_model(self,model, X_test, y_test):
#         """
#         Evaluate the model using regression metrics.

#         :param model: Trained model
#         :param X_test: Test data
#         :param y_test: Test labels
#         """
#         y_pred = model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)

#         print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")
#         logging.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")
#         return mae, rmse ,r2


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import logging
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluation:

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model using regression metrics and plot graphs.

        :param model: Trained model
        :param X_test: Test data
        :param y_test: Test labels
        """
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")
        logging.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

        # Create Plots
        self.plot_results(y_test, y_pred)

    def plot_results(self, y_test, y_pred):
        """
        Generate and save plots for actual vs predicted values and residuals.

        :param y_test: True values
        :param y_pred: Predicted values
        """

        # --- 1. Actual vs. Predicted Scatter Plot ---
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor='k')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red")  # Identity line
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.savefig("actual_vs_predicted.png")  # Save plot
        # plt.show()

        # --- 2. Residual Plot ---
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, edgecolor='k')
        plt.axhline(y=0, linestyle="--", color="red")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title("Residuals vs Predicted Values")
        plt.savefig("residual_plot.png")  # Save plot
        # plt.show()
