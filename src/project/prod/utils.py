import pandas as pd
import numpy as np


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from pathlib import Path
import os
# import pickle
import dill
from src.project.data_transformation.transform import delta_date_feature, ravel_text_column

#------------------------------- Features pipeline -------------------------------------------

def delta_date_feature(dates):
    """
    Given a 2D array containing dates, returns the delta in days between each date 
    and the most recent date in its column.
    """
    dates = pd.DataFrame(dates, columns=["last_review"])
    dates['last_review'] = pd.to_datetime(dates["last_review"], format=f"%Y-%m-%d", errors="coerce")

    max_dates = dates['last_review'].max()
    return dates['last_review'].apply(lambda d : (max_dates - d)).dt.days.fillna(max_dates).to_numpy().reshape(-1, 1)

def ravel_text_column(x):
    return x.ravel()

def load_feature_transformation_pipeline():
    """
    loads the saved pipeline for feature transformation

    parameters:

    inputs:
    -----------


    output:
    -----------
    feature transformation pipeline, feature_names

    """

    # first download the saved pipeline artifact from the dvc tracked 

    artifact_path = Path(r"F:\machine learning\mlops\end to end machine learning pipeline\MLOPs_workflow\src\project\prod\prod_artifacts")
    # pipe_feats_path = Path("F:\machine learning\mlops\end to end machine learning pipeline\MLOPs_workflow\src\project\prod\prod_artifacts\feat_pipeline.pkl")

    with open(artifact_path.joinpath("feat_pipeline.pkl"), 'rb') as pipe_file:
        feat_pipeline = dill.load(pipe_file)

    with open(artifact_path.joinpath("feature_name"), 'rb') as feat_file:
        pipeline_features = dill.load(feat_file)

    # get the feature names for the pipeline
    
    return feat_pipeline, pipeline_features
   

#----------------------------- Models ------------------------------------------------

def load_models():
    """
    loads the saved pickle file from the mlflow registry
    


    Args
        inputs :
        -----------
        Nil

        Output:
        ----------
        machine learning model
    """
    
    # the model is downloaded by using the DVC config files and will be downloaded 
    # to artifact store in /app directory of docker container
    # which we need to access thats it  
    path = Path(r"F:\machine learning\mlops\end to end machine learning pipeline\MLOPs_workflow\src\project\prod\prod_artifacts\random_forest_model.pkl")


    with open(path, 'rb') as file:
        loaded_pickle_model =  dill.load(file)

    return loaded_pickle_model





load_feature_transformation_pipeline()

    