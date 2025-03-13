from zenml import pipeline,step
from steps.data_ingestion_step import data_ingestion_step
from steps.data_preprocessing_step import data_preprocessing_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitting_step import split_data_step
from steps.pca_implementation_step import pca_implementation_step
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step


@pipeline(name="ML Task")
def training_pipeline():
    """
    This pipeline is for performing a Machine Learning task of predicting the level of vomitoxin in a grain sample.
    The pipeline consists of the following steps:
    1. Data Ingestion: This step reads the data from a csv file and loads it into a pandas DataFrame.
    2. Data Preprocessing: This step drops the unneeded columns and replaces missing values with the mean of the column.
    3. Outlier Detection: This step detects outliers in the dataset and removes them.
    4. Feature Engineering: This step applies PCA to the dataset to reduce the number of features.
    5. Data Splitting: This step splits the dataset into training and testing sets.
    6. Model Building: This step builds a model using the training data.
    7. Model Evaluation: This step evaluates the model using the test data.
    The model is trained using a Gradient Boosting Regressor with hyperparameter tuning.
    The output of the pipeline is the trained model.
    """
    df = data_ingestion_step(path ="data/TASK-ML-INTERN.csv" )
    df = data_preprocessing_step(df= df,strategy="mean",columns_to_drop="hsi_id")
    X_scaled,y = outlier_detection_step(df,target_column="vomitoxin_ppb")
    X_pca,y = pca_implementation_step(X_scaled,y,components=2,method="pca",whiten=True,svd_solver="randomized")
    X_train, X_test, y_train, y_test = split_data_step(X_pca,y)
    model = model_building_step(model_name = "gradient",X_train= X_train,y_train=y_train,tune_hyperparameters=True)
    results= model_evaluation_step(model= model,X_test=X_test,y_test= y_test)

    return  model

if __name__ == "__main__":
    training_pipeline()
