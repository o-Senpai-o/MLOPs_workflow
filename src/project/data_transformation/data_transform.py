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
# import dill  # Instead of pickle
import joblib
import cloudpickle
import src.project.data_transformation.data_transformer_utils.data_transform_utils  as data_transform_utils
from src.project.data_transformation.data_transformer_utils.data_transform_utils import delta_date_feature, ravel_text_column
    #--------------------------------------------------------------------------------------------------------------#


# def delta_date_feature(dates):
#     """
#     Given a 2D array containing dates, returns the delta in days between each date 
#     and the most recent date in its column.
#     """
#     dates = pd.DataFrame(dates, columns=["last_review"])
#     dates['last_review'] = pd.to_datetime(dates["last_review"], format=f"%Y-%m-%d", errors="coerce")

#     max_dates = dates['last_review'].max()
#     return dates['last_review'].apply(lambda d : (max_dates - d)).dt.days.fillna(max_dates).to_numpy().reshape(-1, 1)

# def ravel_text_column(x):
#     return x.ravel()

def get_feature_transformation_pipeline():
    """
    Constructs a feature transformation pipeline.

    Returns:
    --------
    feature_transformation_pipeline : Pipeline
        Scikit-learn pipeline for feature preprocessing.
    processed_features : list
        List of input features before transformation.
    new_features : list
        List of transformed feature names.
    """

    # Categorical Features
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]

    ordinal_categorical_preproc = OrdinalEncoder()

    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    # Numerical Features with Zero Imputation
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

    cloudpickle.register_pickle_by_value(data_transform_utils)
    cloudpickle.dumps(delta_date_feature)
    cloudpickle.dumps(ravel_text_column)

    # Date Transformation
    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(data_transform_utils.delta_date_feature, validate=False)
    )
   

    # Text Feature Engineering for 'name' column
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        FunctionTransformer(data_transform_utils.ravel_text_column, validate=False),  # Ensures 1D input for TF-IDF
        TfidfVectorizer(binary=False, max_features=5, stop_words='english')
    )

    
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop"  # Drops unused columns
    )

    # Feature Lists
    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    new_features = ordinal_categorical + \
                ['neighbourhood_group_Bronx', 'neighbourhood_group_Brooklyn', 
                    'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens', 
                    'neighbourhood_group_Staten Island'] + \
                zero_imputed + ['last_review'] + \
                ['apartment', 'bedroom', 'cozy', 'private', 'room']

    # Final Pipeline
    feature_transformation_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor)]
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
    first_df_columns = columns[0:5]
    second_df_columns = columns[5:9]
    third_df_columns = columns[9:13]
    forth_df_columns = columns[13:19]


    # create 4 separate dataframe
    data_1 = data[first_df_columns]
    data_2 = data[second_df_columns]
    data_3 = data[third_df_columns]
    data_4 = data[forth_df_columns]


    # add Ids to each dataframe
    data_1_w_id = pd.concat([data_1, Ids], axis = 1)
    data_2_w_id = pd.concat([data_2, Ids], axis = 1)
    data_3_w_id = pd.concat([data_3, Ids], axis = 1)
    data_4_w_id = pd.concat([data_4, Ids], axis = 1)

    # add iD and time stamo to target also
    target_df_w_id = pd.concat([target_df, Ids], axis = 1)


    # add timestamo to each dataframe
    timestamps = pd.date_range(
        end=pd.Timestamp.now(), 
        periods=len(data), 
        freq='D').to_frame(name="event_timestamp", index=False)


    data_1_id_ts = pd.concat([data_1_w_id, timestamps], axis = 1)
    data_2_id_ts = pd.concat([data_2_w_id, timestamps], axis = 1)
    data_3_id_ts = pd.concat([data_3_w_id, timestamps], axis = 1)
    data_4_id_ts = pd.concat([data_4_w_id, timestamps], axis = 1)

    # add iD and time stamo to target also
    target_df_id_ts = pd.concat([target_df_w_id, timestamps], axis = 1)



    # save the new dataframes to 
    file_direc = Path("F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//processed")


    # feast expects data to be in parquet format
    data_1_id_ts.to_parquet(file_direc.joinpath("data_1.parquet"), index=False)
    data_2_id_ts.to_parquet(file_direc.joinpath("data_2.parquet"), index=False)
    data_3_id_ts.to_parquet(file_direc.joinpath("data_3.parquet"), index=False)
    data_4_id_ts.to_parquet(file_direc.joinpath("data_4.parquet"), index=False)
    target_df_id_ts.to_parquet(file_direc.joinpath("target.parquet"), index=False)

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
    artifact_path = Path("F:\machine learning\mlops\end to end machine learning pipeline\MLOPs_workflow\src\project\prod\prod_artifacts")
    
    
    
    #------------------------------------------pipeline --------------------------------------------------
    cloudpickle.register_pickle_by_value(data_transform_utils)
    
    if os.path.exists(artifact_path):
        # print(artifact_path)
        # with open(artifact_path.joinpath("feat_pipeline.pkl"), "wb") as file:
        #     pickle.dump(feature_transformation_pipeline, file)
        with open(artifact_path.joinpath("feat_pipeline.pkl"), "wb") as file:
            cloudpickle.dump(feature_transformation_pipeline, file)
        print("pipeline saved as pickle file")

    else:

        # make a directory first then open a file 
        os.makedirs(artifact_path)
        with open(artifact_path.joinpath("feat_pipeline.pkl"), 'wb') as file:
            cloudpickle.dump(feature_transformation_pipeline, file)

        print("artifact directory created and pipeline saved as pickle file ")
    


    
    #------------------------------------------pipeline features--------------------------------------------------
    # we will also save the feature names
    features_path = artifact_path
    if not os.path.isfile(Path(features_path, "feature_name.pkl")):
        with open(Path(features_path, "feature_name.pkl"), 'wb') as file:
            joblib.dump(transformed_features, file)
    else:

        with open(Path(features_path, "feature_name.pkl"), 'wb') as file:
            joblib.dump(transformed_features, file)

    # #? later we will track it using the feat pipeline





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




if __name__ == "__main__":
    data_path = Path("F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//raw//AB_NYC_2019.csv")

    data_process(data_path)


