from src.pca_implementation import PCAImplementation
from src.pca_implementation import TSNEImplementation
import pandas as pd
import logging
from typing import Tuple
from zenml import step
import numpy as np


@step(enable_cache=False)
def pca_implementation_step(
    X_scaled: pd.DataFrame, 
    y: pd.Series, 
    method: str = "pca",  # User chooses "pca" or "tsne"
    components: int = 2, 
    whiten: bool = False, 
    svd_solver: str = "auto", 
    perplexity: int = 30, 
    learning_rate: int = 200
) -> Tuple[np.ndarray, pd.Series]:
    """
    Apply PCA or t-SNE based on user selection.

    :param X_scaled: Normalized features
    :param y: Target variable
    :param method: "pca" for PCA, "tsne" for t-SNE
    :param components: Number of components for PCA/t-SNE
    :param whiten: PCA whitening (ignored for t-SNE)
    :param svd_solver: PCA solver method (ignored for t-SNE)
    :param perplexity: t-SNE perplexity (ignored for PCA)
    :param learning_rate: t-SNE learning rate (ignored for PCA)
    :return: Reduced feature set and target variable
    """

    logging.info(f"Applying {method.upper()} to the data")

    if method.lower() == "pca":
        pca = PCAImplementation()
        X_transformed, y = pca.apply_pca(
            X_scaled=X_scaled, y=y, n_components=components,
            whiten=whiten, svd_solver=svd_solver
        )
        logging.info(f"Applied PCA with parameters: n_components={components}, whiten={whiten}, svd_solver={svd_solver}")

    elif method.lower() == "tsne":
        tsne = TSNEImplementation()
        X_transformed = tsne.apply_tsne(
            X_scaled=X_scaled, y=y, n_components=components,
            perplexity=perplexity, learning_rate=learning_rate
        )
        logging.info(f"Applied t-SNE with parameters: n_components={components}, perplexity={perplexity}, learning_rate={learning_rate}")

    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    logging.info(f"Applied {method.upper()} successfully.")
    return X_transformed, y
