import pandas as pd
from sklearn.impute import SimpleImputer
from src.project.data_transformation.data_transform_utils import *
from pathlib import Path
from src.project.prod.utils import load_models
import joblib


def CombineDataFrames(dataframes):
    """ 
    combine list of dataframes
    """
    result = pd.concat(dataframes, axis=1)
    return result


def MeanImputer(data, columns):
    """ 
    impute null values with mean values , doest word column wise but will work on overall data 
    
    """
    # Numerical Features with Zero Imputation
    mean_imputer = SimpleImputer(strategy="mean", fill_value=0)
    mean_imputed = mean_imputer.fit_transform(data[columns])

    mean_imputed_df = pd.DataFrame(mean_imputed, columns=columns)
    return mean_imputed_df


def pipeline(df, path, training=True):
    """

    will take a dataframe as input and process the required features and return the transformed dataframe,
    if the training args is true that means we need to train the sklearn artifacts used in each classes and save them in respectve 
    folders

    Parameters:
    ---------------------
    df : pd.DataFrame
        dataframe to preprocess
    
    path : pathlib.Path
        path to store or retrieve artifacts
    
    training : boolean
        whether to fit and save the artifacts or retrieve the fitter artifacts


    """
    # get the columns that need processing
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
        
    # Numerical Features with Zero Imputation
    mean_imputed_columns = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
        ]
    
    
    
    if training:
        # we will only save the artifacts in the respective folder

        # first for the given dataframe, process the ordinal categorial features 
        oheobj = OrdinalEncoderTransformer()
        oheobj.fit(df, column_name=ordinal_categorical[0])
        oheobj.save(path / "OrdinalCat")

        nonordinalprocess = NonOrdinalCategoricalTransformer()
        nonordinalprocess.fit(df, column_name=non_ordinal_categorical[0])
        nonordinalprocess.save(path / "NonOrdinalCat")

        datefeatureprocess = DeltaDatetimeFeature()
        datefeatureprocess.fit(df, column_name="last_review")
        datefeatureprocess.save(path / "DateFeature")

        # training doesnt need mean imputation
        # meanimputerprocessed = MeanImputer(mean_imputed_columns)

        nameprocess = tfidfVectorizerCompute()
        nameprocess.fit(df, column_name="name")
        nameprocess.save(path / "tfidf")

        return "done"

    # during the inference
    else:

        # load the required classes 
        oheobj = OrdinalEncoderTransformer.load(path / "OrdinalCat")
        onehotencoded = oheobj.transform(df, column_name=ordinal_categorical[0])
        print(onehotencoded.shape)
        print(onehotencoded.columns)
        print("\n\n")


        nonordinalprocess = NonOrdinalCategoricalTransformer.load(path / "NonOrdinalCat")
        nonordinalencoded = nonordinalprocess.transform(df, "neighbourhood_group")
        print(nonordinalencoded.shape)
        print(nonordinalencoded.columns)
        print("\n\n")


        datefeatureprocess = DeltaDatetimeFeature.load(path / "DateFeature")
        date_transformed = datefeatureprocess.transform(df, 'last_review')
        print(date_transformed.shape)
        print(date_transformed.columns)
        print("\n\n")


        nameprocess = tfidfVectorizerCompute.load(path / "tfidf")
        name_transfored = nameprocess.transform(df, column_name='name')
        print(name_transfored.shape)
        print(name_transfored.columns)
        print("\n\n")

        # we need to change this also because , when we dont have to calculate mean during inference right ??
        meanimputerprocessed = MeanImputer(df, mean_imputed_columns)
        print(meanimputerprocessed.shape)
        print(meanimputerprocessed.columns)
        print("\n\n")

        datalist = [onehotencoded, nonordinalencoded, date_transformed, name_transfored, meanimputerprocessed]

        final_df = CombineDataFrames(datalist)
        return final_df
    
 



if __name__ == "__main__":
    
    data_path = Path(r"data\raw\AB_NYC_2019.csv")
    artifact_path = Path("src\project\prod\prod_artifacts")
    
    data = pd.read_csv(data_path)

    # df = pipeline(data, artifact_path , training=True)
    # print(df)

    df = pipeline(data, artifact_path , training=False)
    # print(df.head())
    # print(df.columns)
    # df['id'] = data['id']

    # df.rename(columns={"days_from_max_date": "last_review"}, inplace=True)

    print(df.head())
    artifact_path = Path(r"F:\machine learning\mlops\end to end machine learning pipeline\MLOPs_workflow\src\project\prod\prod_artifacts\random_forest_model")
    feature_name = Path(r"F:\machine learning\mlops\end to end machine learning pipeline\MLOPs_workflow\src\project\prod\prod_artifacts\feature_name.pkl")
    
    with open(feature_name, "rb") as file:
        features = joblib.load(file)
        print(features)
    
    
    with open(artifact_path, "rb") as file:
        model = joblib.load(file)

    # model = load_models()
    print(model.predict(df))






       
# 'room_type',
# ('neighbourhood_group_Bronx',),
# ('neighbourhood_group_Brooklyn',),
# ('neighbourhood_group_Manhattan',),
# ('neighbourhood_group_Queens',),
# ('neighbourhood_group_Staten Island',),
# 'days_from_max_date',
# 'apartment',
# 'bedroom',
# 'cozy',
# 'private',
# 'room', 
# 'minimum_nights',
# 'number_of_reviews',
# 'reviews_per_month', --> last_review
# 'calculated_host_listings_count',
# 'availability_365',
# 'longitude',
# 'latitude'],