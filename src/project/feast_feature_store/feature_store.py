# create a feature store
#? instead of using terminal we will try to do the feast __init__ -m ssssssss
#? and feast apply using python code

# upload the features to the feature store


# get the features from feature store
from feast import FeatureStore
import pandas as pd
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage


def getTrainDataFromFeatureStore():
    """
    retrieve the Training data from the feature store
    # save the data into new repository and return the data

    Args:
        input:
        ------------

        output:
        ------------
        training data : .csv
    
    """


    feature_store = FeatureStore(repo_path = 'F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//src//project//feast_feature_store//feature_repo')

    entity_df = pd.read_parquet("F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//processed//target.parquet")


    training_data = feature_store.get_historical_features(
        entity_df = entity_df,
        
        features =[
            "df_feature_set_1:room_type",
            "df_feature_set_1:neighbourhood_group_Bronx",
            "df_feature_set_1:neighbourhood_group_Brooklyn",
            "df_feature_set_1:neighbourhood_group_Manhattan",
            "df_feature_set_1:neighbourhood_group_Queens",

            "df_feature_set_2:neighbourhood_group_Staten Island",
            "df_feature_set_2:minimum_nights",
            "df_feature_set_2:number_of_reviews",
            "df_feature_set_2:reviews_per_month",

            "df_feature_set_3:calculated_host_listings_count",    
            "df_feature_set_3:availability_365",
            "df_feature_set_3:longitude",
            "df_feature_set_3:latitude",

            "df_feature_set_4:last_review",
            "df_feature_set_4:apartment",
            "df_feature_set_4:bedroom",
            "df_feature_set_4:cozy",
            "df_feature_set_4:private",
            "df_feature_set_4:room",
        ]
    )

    print(training_data)
    # dataset = feature_store.create_saved_dataset(
    #     from_=training_data,
    #     name="NYC_dataset",

    #     storage = SavedDatasetFileStorage("F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//processed//new")
    # )

    # save te new training data into new folder as csv file 
    # training_data.to_df().to_csv("F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//processed//feature_store_data//training_data.csv", index=False)

    # return the train data dataframe
    return training_data.to_df()



print(getTrainDataFromFeatureStore().head())