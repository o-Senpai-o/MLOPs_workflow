import argparse
import logging
import pandas as pd
from src.project import logger

from pathlib import Path




def process_data(args):
    """
    process the raw csv file and cleans it and makes features

    Input : raw data csv file

    Output : processed data csv file
    """

    # log the start of workflow
    logger.info("starting the data processing workflow")

    # reading the dataset
    df = pd.read_csv(args.artifact_path)

    # Drop outliers
    idx = df['price'].between(10, 350)      #in dollars
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Drop rows in the dataset that are not in the proper geolocation
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the cleaned dataset 
    logger.info("Saving the output artifact")
    clean_file_path = "MLOPs_workflow\data\processed"
    file_name = "clean_sample.csv"
    
    df.to_csv(Path(clean_file_path,file_name), index=False)

    




if __name__ == "__main__":
    # we will get our configurations or arguments from config file 
    parser = argparse.ArgumentParser(description = "basic cleaning of raw dataset")

    parser.add_argument("artifact_path", type = str, help = "location of raw dataset")

    args = parser.parse_args()
    # call the data clearning function
    process_data(args)



