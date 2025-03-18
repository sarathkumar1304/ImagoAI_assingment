import pandas as pd
import joblib  # Load PCA model
from zenml import step

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Dynamically imports data, applies PCA, and returns JSON for inference."""

    # Load dataset
    df = pd.read_csv("data/TASK-ML-INTERN.csv")

    # Drop non-feature columns
    df.drop(["hsi_id", "vomitoxin_ppb"], axis=1, inplace=True)

    # Select only 2 samples
    sample_data = df.sample(n=2, random_state=42)  # Randomly select 2 samples
    

    # Load trained PCA model
    pca = joblib.load("models/pca.pkl")

    # #Ensure the same scaler (if used during PCA training)
    # scaler = joblib.load("models/min_max_scaler.pkl")  # Load the same scaler
    # sample_data_scaled = scaler.transform(sample_data)  # Scale before PCA

    # Apply PCA transformation (reduce to 2 components)
    X_pca = pca.transform(sample_data)

    # Convert transformed data to structured JSON
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca.insert(0, 'hsi_id', [f"imagoai_corn_{i}" for i in range(len(df_pca))])  # Unique ID
    
    # Convert DataFrame to JSON (structured format)
    json_data = df_pca.to_json(orient="records")

    return json_data
