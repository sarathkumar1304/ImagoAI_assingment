from sklearn.decomposition import PCA
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import joblib
import os
from sklearn.manifold import TSNE


class PCAImplementation:
    def apply_pca(self,X_scaled:pd.DataFrame, y:pd.Series,n_components=2,
                  whiten:bool=False,svd_solver:str="auto")->Tuple[
        np.ndarray,pd.Series
    ]:
        """
        Apply PCA to reduce dimensionality.

        :param X_scaled: Normalized features
        :param n_components: Number of principal components
        :return: Reduced feature set
        """
        pca = PCA(n_components=n_components,whiten =whiten,svd_solver=svd_solver)
        X_scaled = X_scaled.drop("vomitoxin_ppb",axis=1)
        X_pca = pca.fit_transform(X_scaled)
        logging.info(f"{X_pca.shape}")
        logging.info("PCA pickle file saved  successfully")
        
        explained_variance = pca.explained_variance_ratio_
        print(f"Variance explained by first {n_components} components: {explained_variance.sum() * 100:.2f}%")
        logging.info(f"Variance explained by first {n_components} components: {explained_variance.sum() * 100:.2f}%")
        

        # Scatter plot
        plt.figure(figsize=(8,6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.6)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Projection")
        plt.colorbar(label="DON Concentration")

        # Save the plot before showing it
        plt.savefig("pca_projection.png", dpi=300, bbox_inches='tight')  # Save as high-quality PNG
        # plt.savefig("pca_projection.pdf", dpi=300, bbox_inches='tight')  # Save as PDF (optional)

        plt.show()
        logging.info(f"{type(X_pca),type(y)}")
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "pca.pkl")
        joblib.dump(pca, model_path)
        logging.info(f"Model saved locally at {model_path}")
        return X_pca,y

    # Explained variance




class TSNEImplementation:
    def apply_tsne(self, X_scaled: pd.DataFrame, y: pd.Series, n_components=2, perplexity=30, learning_rate=200, random_state=42) -> np.ndarray:
        """
        Apply t-SNE for dimensionality reduction and visualization.

        :param X_scaled: Normalized feature set (without the target variable)
        :param y: Target variable
        :param n_components: Number of components (default = 2 for visualization)
        :param perplexity: Controls the balance between local/global aspects (default = 30)
        :param learning_rate: Step size during optimization (default = 200)
        :param random_state: Ensures reproducibility
        :return: Reduced feature set (X_tsne)
        """

        logging.info(f"Applying t-SNE with {n_components} components, perplexity={perplexity}, learning_rate={learning_rate}")

        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
        X_tsne = tsne.fit_transform(X_scaled)

        logging.info(f"t-SNE shape: {X_tsne.shape}")
        
        # Scatter Plot for t-SNE Visualization
        plt.figure(figsize=(8, 6))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis", alpha=0.6)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.title("t-SNE Projection of Data")
        plt.colorbar(label="Target Value")

        # Save the plot
        plt.savefig("tsne_projection.png", dpi=300, bbox_inches="tight")  # High-quality PNG

        plt.show()
        
        # Save the t-SNE model
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "tsne.pkl")
        joblib.dump(tsne, model_path)

        logging.info(f"t-SNE model saved at {model_path}")

        return X_tsne
