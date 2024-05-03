# fetches the data from the source files
# from src import logger
import hydra

import pandas as pd
import os
import kaggle
from dotenv import load_dotenv

import argparse

from src.project import logger


# loading the environment variable files for kaggle access
load_dotenv("MLOPs_workflow\src\project\configs")



def run(args):
    """
    download the data from the source file

    Args: 
        url : url from which the data is fetched

    """

    # logger.info("download the data from source url")

    # Assign the Kaggle data set URL into variable
    dataset = args.uri
    # print(dataset)
    # Using opendatasets let's download the data sets
    raw_data_path = args.data.data.artifact_loc
   

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files('New York City Airbnb Open Data',
                                  path=raw_data_path,
                                  unzip=True)

    # you can write all your data ingestio code here in this file
    # data can be fetched from multiple locations  
    # as we already have our data present locally we can skip this step for next level of development
    # download_data(args["url"])



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data ingestion from source")
    
    parser.add_argument("uri", type=str, help = "source url of the raw data")
    

    args = parser.parse_args()
    run()