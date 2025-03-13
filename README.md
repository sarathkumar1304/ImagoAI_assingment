# ImagoAI Assignment

## Project Architecture:

![Project Architecture](assets/architecture.svg)

## Problem Statement
You are provided with a compact hyperspectral dataset containing spectral reflectance data from corn samples across multiple wavelength bands.

## Objective
This assignment assesses your ability to process hyperspectral imaging data, perform dimensionality reduction, and develop a machine learning model to predict mycotoxin levels (e.g., DON concentration) in corn samples.


## Project Overview
This project implements various machine learning and deep learning models to analyze and predict outcomes based on structured data. The workflow includes data preprocessing, dimensionality reduction, model training, evaluation, and deployment using MLOps tools.

## Tools and Technologies Used
- **Programming & ML/DL**: Python, Machine Learning, Deep Learning (CNN, MLP, LSTM)
- **MLOps**:
  - **ZenML**: Data pipeline, artifact management, and orchestration
  - **MLflow**: Experiment tracking, model registry, and deployment
- **UI Framework**:
  - **Streamlit**: For building interactive user interfaces

## Repository Structure
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
      |-- EDA.ipynb                         # Jupyter Notebooks for EDA & model training
      |-- cnn.ipynb                         # cnn implementation
      |-- lstm.ipynb                        # LSTM implementation
      |-- MLP.ipynb                         # MLP implementation
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
```

## Installation
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

we visuliaze the dash board like below and you can track the experiment using mlflow 

#### Data pipeline 

![zenml image](assets/zenml_1.png)



#### Experiment Tracking 

![mlflow image](assets/mlflow_3.png)

#### Model Metric Tracking

![mlflow image](assets/mlflow_4.png)


#### Continious Deployment Pipeline

![zenml_image_2](assets/zenml_2.png)


#### Inference Pipeline 

![zenml image_3](assets/zenml_3.png)


## Data Preprocessing
- Dropped unwanted columns (e.g., `hsi_id`)
- Removed outliers using IQR method
- Applied standard scaling or MinMax scaling
- Performed dimensionality reduction using PCA or t-SNE

## Model Training
Implemented and evaluated the following machine learning models:
- **Linear Regression**
- **AdaBoost Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**
- **Decision Tree Regressor**
- **Random Forest Regressor**

**Neural Networks** 
- **CNN**
- **LSTM**
- **MLP**


## Model Evaluation
Performance of models with PCA & t-SNE:
| Model                  | MAE  | RMSE  | R² Score |
|------------------------|------|------|---------|
| Gradient Boosting     | 529  | 687  | 96.4%   |
| AdaBoost              | 600  | 1544 | 94.97   |
| Random Forest         | 700  | 3067 | 93.7  |
| Decision Tree         | 529  | 687  | 92.7  |

## Future Improvements
- Optimize hyperparameters for better performance
- Explore deep learning models for enhanced prediction
- Improve data preprocessing techniques
- Deploy model using cloud services

---
**Contributors**: R. Sarath Kumar
