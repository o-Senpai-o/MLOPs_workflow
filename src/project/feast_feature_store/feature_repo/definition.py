"""
create the definition of our feature store

definition : 
    1) Entity
    
    2.1) File Source
    2.2) Feature View
    
    3.1) File Source
    3.2) Feature View

prerequisites:
    # each datafile from data source should have the following
        1) timestamp
        2) same entity 
    
           data1_df                      data2_df                     data3_df                       target
    ID feat1 feat2 TIMESTAMP     ID feat1 feat2 TIMESTAMP      ID feat1 feat2 TIMESTAMP          ID  target TIMESTAMP
    1   -      -      -            1   -      -      -          1   -      -      -              1      -       -
    2   -      -      -            2   -      -      -          2   -      -      -              2      -       -
    3   -      -      -            3   -      -      -          3   -      -      -              3      -       -
    4   -      -      -            4   -      -      -          4   -      -      -              4      -       -
    5   -      -      -            5   -      -      -          5   -      -      -              5      -       -


    # target should be a different data file    
    
    """


import feast
from feast import FileSource, FeatureView, Entity, Field
from feast.types import Float64, Int64

import time
from google.protobuf.duration_pb2 import Duration

from datetime import timedelta




# -------------- defining feature views ------------------



# Declaring an entity for the dataset
ID = Entity(
    name="id", 
    join_keys=["id"])


#? ----------------- Feature View 1 -----------------------

file_source_1 = FileSource(path = "F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow/data//processed//data_1.parquet",
                           name = "features_source_1",
                           event_timestamp_column = "event_timestamp",
                           )

df_1_feature_view = FeatureView(
    name = "df_feature_set_1",
    entities = [ID],
    ttl=timedelta(seconds=86400 * 3),      # 24hrs * 60min * 60sec * 3days --> total seconds in 3 days
    schema= [
        Field(name = "room_type", dtype = Float64),     # change it later when you get data
        Field(name = "neighbourhood_group_Bronx", dtype = Float64),
        Field(name = "neighbourhood_group_Brooklyn", dtype = Float64),
        Field(name = "neighbourhood_group_Manhattan", dtype = Float64),
        Field(name = "neighbourhood_group_Queens", dtype = Float64)
    ],
    source = file_source_1
)

#? ----------------- Feature View 2 -----------------------

file_source_2 = FileSource(path = "F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//processed//data_2.parquet",
                           name = "features_source_2",
                           event_timestamp_column = "event_timestamp",
                           )

df_2_feature_view = FeatureView(
    name = "df_feature_set_2",
    entities = [ID],
    ttl=timedelta(seconds=86400 * 3),     
    schema= [
        Field(name = "neighbourhood_group_Staten Island", dtype = Float64),     # change it later when you get data
        Field(name = "minimum_nights", dtype = Float64),
        Field(name = "number_of_reviews", dtype = Float64),
        Field(name = "reviews_per_month", dtype = Float64)
        
    ],
    source = file_source_2
)

#? ----------------- Feature View 3 -----------------------

file_source_3 = FileSource(path = "F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//processed//data_3.parquet",
                           name = "features_source_3",
                           event_timestamp_column = "event_timestamp",
                           )

df_3_feature_view = FeatureView(
    name = "df_feature_set_3",
    entities = [ID],
    ttl=timedelta(seconds=86400 * 3),      
    schema=  [
        Field(name = "calculated_host_listings_count", dtype = Float64),     # change it later when you get data
        Field(name = "availability_365", dtype = Float64),
        Field(name = "longitude", dtype = Float64),
        Field(name = "latitude", dtype = Float64)
    ],
    source = file_source_3
)

#? ----------------- Feature View 4 -----------------------

file_source_4 = FileSource(path = "F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//processed//data_4.parquet",
                           name = "features_source_4",
                           event_timestamp_column = "event_timestamp",
                           )

df_4_feature_view = FeatureView(
    name = "df_feature_set_4",
    entities = [ID],
    ttl=timedelta(seconds=86400 * 3),      
    schema=  [
        Field(name = "last_review", dtype = Float64),     
        Field(name = "apartment", dtype = Float64),
        Field(name = "bedroom", dtype = Float64),
        Field(name = "cozy", dtype = Float64),
        Field(name = "private", dtype = Float64),
        Field(name = "room", dtype = Float64)
    ],
    source = file_source_4
)


#? ----------------------- target --------------------------

file_source_target = FileSource(
                            path = "F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//processed//target.parquet",
                            event_timestamp_column = "event_timestamp",
                            name = "target" 
                            )

df_target_feature_view = FeatureView(
        name = "target_feature_view",
        entities = [ID],
        ttl=timedelta(seconds=86400 * 3),
        schema=  [
            Field(name = "price", dtype = Int64)
        ],
        source = file_source_target 
)


