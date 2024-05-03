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
import pickle


#------------------------------- Features pipeline -------------------------------------------

def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


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

    pipe_artifact_path = Path("prod_artifacts//feat_pipeline.pkl")
    pipe_feats_path = Path("prod_artifacts//feature_name.pkl")

    with open(pipe_artifact_path, 'rb') as file:
        feat_pipeline = pickle.load(file)

    with open(pipe_feats_path, 'rb') as file:
        pipeline_features = pickle.load(file)

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
    path = Path("prod_artifacts//model")


    with open(path, 'rb') as file:
        loaded_pickle_model =  pickle.load(file)

    return loaded_pickle_model





# print(load_feature_transformation_pipeline())

    