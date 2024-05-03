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

#--------------------------------------------------------------------------------------------------------------#




def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()

def get_feature_transformation_pipeline():
    """
    constructs the pipeline for feature transformation

    parameters:

    inputs:
    -----------
    
    max_tdidf_features : 

    output:
    -----------
    feature transformation pipeline

    """
    

    # Let's handle the categorical features first
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    # NOTE: we do not need to impute room_type because the type of the room
    # is mandatory on the websites, so missing values are not possible in production
    # (nor during training). That is not true for neighbourhood_group
    ordinal_categorical_preproc = OrdinalEncoder()

    
    # Build a pipeline with two steps:
    # 1 - A SimpleImputer(strategy="most_frequent") to impute missing values
    # 2 - A OneHotEncoder() step to encode the variable
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )

    # Let's impute the numerical columns to make sure we can handle missing values
    # (note that we do not scale because the RF algorithm does not need that)
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # A MINIMAL FEATURE ENGINEERING step:
    # we create a feature that represents the number of days passed since the last review
    # First we impute the missing review date with an old date (because there hasn't been
    # a review for a long time), and then we create a new feature from it,
    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    # Some minimal NLP for the "name" column
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})

    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features = 5,
            stop_words='english'
        ),
    )

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )


    # list of features
    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    # list of transformed features
    new_features = ordinal_categorical + ['neighbourhood_group_Bronx','neighbourhood_group_Brooklyn','neighbourhood_group_Manhattan','neighbourhood_group_Queens', \
                                                  'neighbourhood_group_Staten Island'] + zero_imputed + ['last_review'] + \
                            ['apartment', 'bedroom', 'cozy', 'private', 'room']

    ######################################
    # Create the inference pipeline. 
    # The pipeline must have 2 steps: a step called "preprocessor" applying the
    # ColumnTransformer instance that we saved in the `preprocessor` variable, 
    # HINT: Use the explicit Pipeline constructor so you can assign the names to the steps, do not use make_pipeline
    
    feature_transformation_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),             
        ]
    )

    return feature_transformation_pipeline, processed_features, new_features

def create_feast_dataframe(data):

    # get the Id and target columns separately
    Ids = data["id"]
    target_df = data["price"]

    columns = list(data.columns)
    columns.remove("price")
    columns.remove('id')


    # create 4 different dataframes for 4 sources of features
    # first_df_columns = columns[0:5]
    # second_df_columns = columns[5:9]
    # third_df_columns = columns[9:13]
    # forth_df_columns = columns[13:19]


    # create 4 separate dataframe
    # data_1 = data[first_df_columns]
    # data_2 = data[second_df_columns]
    # data_3 = data[third_df_columns]
    # data_4 = data[forth_df_columns]


    # add Ids to each dataframe
    # data_1_w_id = pd.concat([data_1, Ids], axis = 1)
    # data_2_w_id = pd.concat([data_2, Ids], axis = 1)
    # data_3_w_id = pd.concat([data_3, Ids], axis = 1)
    # data_4_w_id = pd.concat([data_4, Ids], axis = 1)

    # add iD and time stamo to target also
    target_df_w_id = pd.concat([target_df, Ids], axis = 1)


    # add timestamo to each dataframe
    timestamps = pd.date_range(
        end=pd.Timestamp.now(), 
        periods=len(data), 
        freq='D').to_frame(name="event_timestamp", index=False)


    # data_1_id_ts = pd.concat([data_1_w_id, timestamps], axis = 1)
    # data_2_id_ts = pd.concat([data_2_w_id, timestamps], axis = 1)
    # data_3_id_ts = pd.concat([data_3_w_id, timestamps], axis = 1)
    # data_4_id_ts = pd.concat([data_4_w_id, timestamps], axis = 1)

    # add iD and time stamo to target also
    target_df_id_ts = pd.concat([target_df_w_id, timestamps], axis = 1)



    # save the new dataframes to 
    file_direc = Path("F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//processed")


    # feast expects data to be in parquet format
    # data_1_id_ts.to_parquet(Path(f"{file_direc}//data_1.parquet"), index=False)
    # data_2_id_ts.to_parquet(Path(f"{file_direc}//data_2.parquet"), index=False)
    # data_3_id_ts.to_parquet(Path(f"{file_direc}//data_3.parquet"), index=False)
    # data_4_id_ts.to_parquet(Path(f"{file_direc}//data_4.parquet"), index=False)

    target_df_id_ts.to_parquet(Path(f"{file_direc}//target.parquet"), index=False)


def data_process(path):
    """
    
    """
    # get the data
    data = pd.read_csv(path)

    # we transform on the whole dataset
    # later while training we will get the data and then split it into train and test


    # first : transform the features  
    feature_transformation_pipeline, features_transformed, transformed_features = get_feature_transformation_pipeline()
    feature_transformed_data = feature_transformation_pipeline.fit_transform(data)

    # save the pipeline and the feature names in mlflow registry or aws

    #? we track the saved pipeline using DVC
    artifact_path = Path("MLOPs_workflow/src//project//prod//prod_artifacts")
    with open(Path(artifact_path, "feat_pipeline.pkl"), 'wb') as file:
        pickle.dump(feature_transformation_pipeline, file)
    

    # we will also save the feature names
    features_path = Path("MLOPs_workflow/src//project//prod//prod_artifacts")
    with open(Path(features_path, "feature_name.pkl"), 'wb') as file:
        pickle.dump(transformed_features, file)

    #? lateer we will track it using the feat pipeline

    # feature_transformed_data is a sparse matrix so we create a new dataframe
    features_df = pd.DataFrame(feature_transformed_data, columns= transformed_features)
    # add the ID and Target column also

    features_df['id'] = data['id']
    features_df['price'] = data['price']


    # second : we will store the data into 4 separate dataframe so that we can use feast in more advanced way
    #          and better understand the working of it

    create_feast_dataframe(features_df)

    
    # second: create a feature store, and upload the features  
    # third: version control the data on google drive but include the code to do the same for AWS S3  


data_path = Path("F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//raw//AB_NYC_2019.csv")

data_process(data_path)


