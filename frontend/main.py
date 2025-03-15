
import streamlit as st
import pandas as pd
import requests
import joblib
import json
import os
import numpy as np
from sklearn.decomposition import PCA

def project():
    # Load local model components
    pca_path = "models/pca.pkl"
    local_model_path = "models/model.pkl"

    # Load models if they exist

    pca = joblib.load(pca_path) if os.path.exists(pca_path) else None
    local_model = joblib.load(local_model_path) if os.path.exists(local_model_path) else None

    # MLflow Server Endpoint
    MLFLOW_SERVER_URL = "http://127.0.0.1:8000/invocations"

    # Streamlit UI
    st.title("Imago AI Vomitoxin Prediction System")
    st.write("Upload a CSV file to predict vomitoxin levels in corn samples.")

    # Upload CSV File
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    # Prediction Function
    def predict(data):
        """Predict using MLflow server if available, else use local model."""
        
        # Try MLflow first
        try:
            st.write("Checking MLflow Server...")
            
            # Convert data to JSON format for MLflow
            json_data = json.dumps({"inputs": df.to_numpy().tolist()})  #  Convert to NumPy first

            
            # Send request to MLflow
            response = requests.post(MLFLOW_SERVER_URL, json={"inputs": json_data})
            
            # If successful, return MLflow prediction
            if response.status_code == 200:
                predictions = response.json()
                st.success("Prediction done using MLflow Server!")
                return np.array(predictions)
            else:
                st.warning("MLflow Server not responding. Falling back to local model.")
        
        except requests.exceptions.RequestException:
            st.warning("MLflow Server is not running. Using local model.")

        # If MLflow fails, use local model
        if local_model and pca:
            # Apply Scaling & PCA
            # scaled_data = scaler.transform(data)
            pca_data = pca.transform(data)
            
            # Predict with local model
            predictions = local_model.predict(pca_data)
            st.success("Prediction done using Local Model!")
            return predictions
        else:
            st.error("No model available for prediction.")
            return None

    # Process File & Predict
    if uploaded_file:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Ensure required columns exist
            if "hsi_id" not in df.columns:
                st.error("CSV file must contain `hsi_id` column.")
            else:
                # Extract Features (Drop hsi_id)
                X = df.drop(columns=["hsi_id"])

                # Make Predictions
                predictions = predict(X)

                if predictions is not None:
                    # Create output DataFrame
                    df["vomitoxin_ppb"] = predictions
                    
                    # Mark Harmful Cases (vomitoxin > 1000)
                    df["Harmful"] = df["vomitoxin_ppb"].apply(lambda x: "⚠️ Yes" if x > 1000 else "✅ No")
                    
                    # Display DataFrame
                    st.subheader("Prediction Results")
                    st.write(df[["hsi_id", "vomitoxin_ppb", "Harmful"]])

                    # Highlight Harmful Cases
                    st.subheader("Harmful Cases")
                    harmful_df = df[df["vomitoxin_ppb"] > 1000]
                    st.write(harmful_df[["hsi_id", "vomitoxin_ppb"]])

        except Exception as e:
            st.error(f"Error processing file: {e}")
