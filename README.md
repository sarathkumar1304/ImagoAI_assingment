# 🚀 ImagoAI Assignment

## 🏗️ Project Architecture:

![Project Architecture](assets/architecture.svg)

## 📌 Problem Statement
You are provided with a compact hyperspectral dataset containing spectral reflectance data from corn samples across multiple wavelength bands.

## 🎯 Objective
This assignment assesses your ability to process hyperspectral imaging data, perform dimensionality reduction, and develop a machine learning model to predict mycotoxin levels (e.g., DON concentration) in corn samples.


## 📜 Project Overview
This project implements various machine learning and deep learning models to analyze and predict outcomes based on structured data. The workflow includes data preprocessing, dimensionality reduction, model training, evaluation, and deployment using MLOps tools.

## 🛠️ Tools and Technologies Used
- **Programming & ML/DL**: Python, Machine Learning, Deep Learning (CNN, MLP, LSTM)
- **MLOps**:
  - **ZenML**: Data pipeline, artifact management, and orchestration
  - **MLflow**: Experiment tracking, model registry, and deployment
- **UI Framework**:
  - **Streamlit**: For building interactive user interfaces

## 📂 Repository Structure
```
ImagoAI_Assignment/
|│──.zen/
      │──config.yaml
|│──assets/
       │──images of the project
│── data/TASK-ML-INTERN.csv                 # Raw and processed data
|── models
     |── pickle file                        # Trained model pickle file
|──myenv             
│── notebooks/
      |-- analyze_plots                     # Source code for all residual plots
      |-- EDA.ipynb                         # Jupyter Notebooks for EDA & model training
      |-- ml_reseach.ipynb                  # All Ml algo implementation
      |-- mlp_research.ipynb                # MLP  implementation
      |-- tsne_implementation.ipynb         # TSNE implementation
│── pipelines/                              # Python scripts for modular implementation
        |── deployment_pipeline.py          # Contininous pipeline and inference pipeline
        |── training_pipeline.py            # training machine learning pipeline
|── src/
      |── data_ingestion.py                 # data ingestions
      |── data_preprocessing.py             # data cleaning
      |── data_splitting.py                 # data split into train and test 
      |── model_building.py                 # model building for model training
      |── model_evaluation.py               # model evaluation of trained model
      |── outlier_detection.py              # remove outliers and apply min max scaling
      |── pca_implementation.py             # pca implementation or tsne implementation
|── steps/
      |── data_ingestion_step.py           # data ingestions step contain zenml step to track flow of data 
      |── data_preprocessing_step.py       # data preprocessing  step like cleaning fill null values
      |── data_splitting_step.py           # data spliited into train and test step
      |── model_building_step.py           # model building step
      |── model_evaluation_step.py         # model evaluation step 
      |── outlier_detection_step.py        # outlier detection , outlier removal and apply minmax scaling step
      |── pca_implementation_step.py       # pca implemention and selection step
      |── dynamic_importer.py              # import sample data for testing
      |──prediction_service_loader.py      # mlflow prediction service loader
      |──predictor.py                      # prediction 
│── run_pipeline.py                        # run whole pipeline at one place 
│── run_deployment.py                      # Model deployment process
│── README.md                              # Project documentation
│── requirements.txt                       # List of dependencies
│── DockerFile
│── docker-compose.yml
```

## ⚙️ Installation
To set up the environment and install dependencies, follow these steps:
```bash
mkdir ImagoAI_Assignment
cd ImagoAI_Assignment

python3 -m venv myenv
source myenv/bin/activate

pip install zenml["server"]
pip install -r requirements.txt

zenml init
zenml integration install mlflow -y
zenml register experiment-tracker ImagoAI_experiment_tracker -flavor=mlflow
zenml register model-deployer ImagoAI_model_deployer -flavor=mlflow
zenml stack register -a d -o d -e ImagoAI_experiment_tracker -d ImagoAI_model_deployer --set

streamlit run app.py
```

```
python3 run_pipeline.py
```

we visuliaze the dash board like below and you can track the experiment using mlflow 📊

#### Data pipeline 🏗️

![zenml image](assets/zenml_1.png)



####  📈 Experiment Tracking 

![mlflow image](assets/mlflow_3.png)

#### 📊 Model Metric Tracking

![mlflow image](assets/mlflow_4.png)


#### 🔄 Continious Deployment Pipeline

![zenml_image_2](assets/zenml_2.png)


#### 🤖 Inference Pipeline 

![zenml image_3](assets/zenml_3.png)


## 🧹 Data Preprocessing
- Dropped unwanted columns (e.g., `hsi_id`)
- Removed outliers using IQR method
- Applied standard scaling or MinMax scaling
- Performed dimensionality reduction using PCA or t-SNE

## 🤖 Model Training
Implemented and evaluated the following machine learning models:
- **Linear Regression**
- **AdaBoost Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**
- **Decision Tree Regressor**
- **Random Forest Regressor**

**Neural Networks** 
- **MLP 🔗**


## 📊 Model Evaluation for PCA
Performance of models with PCA 
| Model                  | MAE  | RMSE  | R² Score | Hyperparameter tuning  using Optuna|
|------------------------|------|------|---------|------------------------|
| Linear Regression     | 4381.5374 | 12199.1782 | 0.1339 | False |
| Gradient Boosting     | 3115.4155  | 6486.3161  | 0.7551  | True |
| Gradient Boosting     | 1690.3613  | 3174.3115  | 0.9414  | Flase |
| AdaBoost              |1922.8089  | 3438.2346 | 0.9312   | True |
| AdaBoost              | 2006.6873  | 3413.3609 | 0.9322   | False |
| Random Forest         | 2775.4079, | 6836.0452 |0.7280 | False |
| Random Forest         | 2726.9571 | 6015.5543| 0.7894  | True |
| Decision Tree         | 2497.0567 | 6843.7254 | 0.7274  | True |
| Decision Tree         |  2430.3297 | 4751.6847 |0.8686 | False |
| xgboost | 2473.6890 | 4434.9331 | 0.8855 | True |
| xgboost | 2042.3208| 3855.3484, | 0.9135| False |
|SVM |  3579.0400, |13264.9131|-0.0241 |True|
|SVM |3643.8956 |13533.4645, | -0.0660 | False

### **📊 Model Evaluation Interpretation for PCA**
This table compares **various regression models** with **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² Score**. Let's analyze the results:

#### **1️⃣ Best Performing Model**  
 **Gradient Boosting (No Optuna, R² = 0.9414)**  
   - **MAE = 1690.36**, **RMSE = 3174.31**, **R² = 0.9414**  
   - This means **94.14% of the variance is explained**, making it the **best model**.  

 **XGBoost (No Optuna, R² = 0.9135)**  
   - **MAE = 2042.32**, **RMSE = 3855.35**, **R² = 0.9135**  
   - Strong performance, but slightly lower than Gradient Boosting.

#### **2️⃣ Models That Benefit from Hyperparameter Tuning (Optuna)**
 **Gradient Boosting (Optuna: R² = 0.7551)**  
   - **Optuna tuning significantly improved performance** compared to Linear Regression but **not as strong** as the untuned version.

 **Random Forest (Optuna: R² = 0.7894)**  
   - Tuned version performed **better than default (R² = 0.7280)**.

 **XGBoost (Optuna: R² = 0.8855)**  
   - Tuning **boosted performance but not as much** as untuned Gradient Boosting.

#### **3️⃣ Worst Performing Models**
 **Support Vector Machine (SVM)**
   - **Worst R² Score (-0.0660, -0.0241)**, meaning **it performs worse than a baseline predictor**.
   - **Very high RMSE and MAE**, indicating it struggles with the dataset.

 **Linear Regression (R² = 0.1339)**
   - **Very poor fit** compared to tree-based models.

---

### **🏆 Final Conclusion**
| **Best Model**  | **Gradient Boosting (No Optuna)** 🏆  |
|----------------|--------------------------------|
| **R² Score**   | **0.9414** (Best variance explanation) |
| **MAE**        | **1690.36** (Lowest error) |
| **RMSE**       | **3174.31** (Good generalization) |

#### **🔹 Recommendations**
1. **Use Gradient Boosting (Untuned)** → It has the best **R² score (0.9414)** and the lowest error.
2. **If you want a backup model**, use **XGBoost (No Optuna, R² = 0.9135)**.
3. **Avoid SVM & Linear Regression** as they perform the worst.
4. **Optuna tuning helped some models** (Random Forest, XGBoost) but wasn't necessary for Gradient Boosting.


## **🔮 Future Improvements**
- **Optimize Gradient Boosting further using Bayesian Optimization.**
- **Experiment with Stacking XGBoost + Gradient Boosting for better ensemble learning.**
- **Try TabNet (Deep Learning for Structured Data) to improve performance.**
- **Use Physics-Informed Machine Learning (PIML) for spectral data.**
- **Apply Autoencoders or Contrastive Learning for Feature Learning.**
- **Implement Explainable AI (XAI) techniques to understand feature importance.**
- **Experiment with Hybrid Models (Combining ML + Deep Learning approaches).**
- **Deploy models using Kubernetes for scalable inference.**
- **Explore Federated Learning for privacy-preserving model training.**


