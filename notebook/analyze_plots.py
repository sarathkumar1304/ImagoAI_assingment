import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

class Plots:

    # Function to plot Actual vs. Predicted values (with error lines)
    def plot_actual_vs_predicted(self, y_test, y_pred):
        """Plots a scatter plot of actual vs. predicted values with red error lines."""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, edgecolor="k", color="blue", label="Predicted")

        # Red Error Lines
        for i in range(len(y_test)):
            plt.plot([y_test[i], y_test[i]], [y_test[i], y_pred[i]], color="red", alpha=0.6)

        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="orange", linestyle="--", lw=2, label="Actual")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values with Error Lines")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()

    # Function to perform Residual Analysis (with red error lines)
    def plot_residual_analysis(self, y_test, y_pred):
        """Plots residual distribution with red error lines."""
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, bins=30, kde=True, color="blue", alpha=0.6)
        plt.axvline(x=0, color="orange", linestyle="--", lw=2)

        # Red Error Lines for outliers
        outlier_threshold = np.mean(residuals) + 2 * np.std(residuals)
        plt.axvline(x=outlier_threshold, color="red", linestyle="--", lw=2)
        plt.axvline(x=-outlier_threshold, color="red", linestyle="--", lw=2)

        plt.xlabel("Residuals (Errors)")
        plt.ylabel("Frequency")
        plt.title("Residual Analysis: Distribution of Errors with Outlier Markers")
        plt.grid()
        plt.show()

    # Function to plot Residuals vs. Fitted Values (with red error lines)
    def plot_residuals_vs_fitted(self, y_test, y_pred):
        """Plots residuals vs. predicted values with red error lines."""
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, edgecolor="k", color="blue")

        # ✅ Red Error Lines
        for i in range(len(y_test)):
            plt.plot([y_pred[i], y_pred[i]], [0, residuals[i]], color="red", alpha=0.6)

        plt.axhline(y=0, color="orange", linestyle="--", lw=2)
        plt.xlabel("Predicted Values (Fitted)")
        plt.ylabel("Residuals (Errors)")
        plt.title("Residuals vs. Predicted Values with Error Lines")
        plt.grid()
        plt.show()

    # ✅ Function to create a Q-Q Plot
    def plot_qq_plot(self, y_test, y_pred):
        """Creates a Q-Q plot to check normality of residuals."""
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot of Residuals")
        plt.grid()
        plt.show()

    # ✅ Function to plot Cook's Distance for Outlier Detection
    def plot_cooks_distance(self, X_train, y_train):
        """Plots Cook's Distance to detect influential data points."""
        X_train_const = sm.add_constant(X_train)
        model_ols = sm.OLS(y_train, X_train_const).fit()
        influence = model_ols.get_influence()
        cooks_d = influence.cooks_distance[0]

        plt.figure(figsize=(8, 6))
        plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",", linefmt="b-", basefmt="r-")
        plt.xlabel("Observation Index")
        plt.ylabel("Cook's Distance")
        plt.title("Cook's Distance for Outlier Detection")
        plt.grid()
        plt.show()

    # ✅ Function to plot Actual vs. Predicted Line Plot (with error lines)
    def plot_actual_vs_predicted_line(self, y_test, y_pred):
        """Plots actual vs. predicted values over time or sample index with error lines."""
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label="Actual", marker="o", color="orange")  # Actual in Orange
        plt.plot(y_pred, label="Predicted", marker="s", color="blue")  # Predicted in Blue

        # ✅ Red Error Lines
        for i in range(len(y_test)):
            plt.plot([i, i], [y_test[i], y_pred[i]], color="red", alpha=0.6)

        plt.xlabel("Sample Index")
        plt.ylabel("Target Variable")
        plt.title("Actual vs. Predicted Values Over Time with Error Lines")
        plt.legend(loc="upper right")  # Legend in Upper Right
        plt.grid()
        plt.show()
