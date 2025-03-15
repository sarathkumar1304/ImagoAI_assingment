import streamlit as st
from PIL import Image


st.set_page_config(page_title="ImagoAI Assignment", layout="wide")
# Function to load images
def load_image(image_path):
    return Image.open(image_path)

# Function to display the home UI
def home_ui():


    # Title & Project Overview
    st.title("🚀 ImagoAI Assignment")
    st.write("### 📌 Hyperspectral Data Analysis & Mycotoxin Prediction")

    # Project Architecture
    st.image("assets/architecture.svg", caption="🏗️ Project Architecture", use_container_width =True)

    # Objective Section
    st.header("🎯 Objective")
    st.write("This project involves processing hyperspectral imaging data, performing dimensionality reduction, and developing ML/DL models to predict mycotoxin levels in corn samples.")

    # Tools and Technologies
    st.header("🛠️ Tools & Technologies Used")
    st.markdown("""
    - **Programming & ML/DL**: Python, Machine Learning, Deep Learning (CNN, MLP, LSTM)
    - **MLOps**: ZenML (Pipeline Management), MLflow (Experiment Tracking)
    - **UI Framework**: Streamlit (For interactive visualization)
    """)

    # Model Performance
    st.header("📊 Model Performance with PCA")
    st.dataframe({
        "Model": ["Linear Regression", "Gradient Boosting (No Optuna)", "XGBoost (No Optuna)", "SVM"],
        "MAE": [4381.5, 1690.3, 2042.3, 3643.8],
        "RMSE": [12199.1, 3174.3, 3855.3, 13533.4],
        "R² Score": [0.1339, 0.9414, 0.9135, -0.0660]
    })

    # MLflow Experiment Tracking
    st.header("📈 Experiment Tracking")
    st.image("assets/mlflow_3.png", caption="MLflow Experiment Tracking", use_container_width =True)

    # Future Improvements
    st.header("🔮 Future Improvements")
    st.markdown("""
    - **Optimize Gradient Boosting using Bayesian Optimization**
    - **Implement Stacking (XGBoost + Gradient Boosting)**
    - **Use Physics-Informed Machine Learning (PIML)**
    - **Deploy models using Kubernetes for scalable inference**
    """)

# Run the UI
if __name__ == "__main__":
    home_ui()
