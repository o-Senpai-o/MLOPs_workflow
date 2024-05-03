from src import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.project.modelling.fetch_model import getModel


from feast_feature_store import getTrainDataFromFeatureStore
from fetch_model import getModel

import os
import shutil

from sklearn.metrics import mean_absolute_error 
import mlflow


#---------------------------------------------------------------------------------------------------------------



def train(args):
    """
    Trains model on training data

    Args:
        inputs:
        ------------
        argument_parser : 
            configs : dict = configuration for models
        
            
        output:
        ------------
        None
    
    """

    # Get the Random Forest configuration


    # Fix the random seed for the Random Forest, so we get reproducible results

    # get the training data from the feature store first
    logger.info("Downloading training set artifact from feature store")

    # retrieve the training data from feature store
    data = getTrainDataFromFeatureStore()

    # read the training data
    X = data
    y = X.pop("price")  # this removes the column "price" from X and puts it into y

    # there are some features which are not required for model training 
    # like event_timestamp, 
    X = X.drop(["event_timestamp"], axis = 1)
    
    # split the data into train and test data
    X_train, X_val, y_train, y_val = train_test_split(
                            X, y, test_size = args.val_size,
                            stratify = X[args.stratify_by],
                            random_state = args.random_seed
                        )


    # get te model for training 
    # get the configs for random forest model
    model = getModel(model_name= "random_forest", configs = args.configs)        

    # Then fit it to the X_train, y_train data
    # Fit the pipeline full_pipeline (will transform and train the data )
    #! per yaar data toh humara feature store se aaya hai toh pipeline se fit kyu kare
    #! khali model se fit karna chaheye
    model.fit(X_train, y_train)

    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = model.score(X_val, y_val)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    # Save the sk_pipe pipeline as a mlflow.sklearn model in the directory "random_forest_dir"

    export_path = "random_forest_dir"
    signature = infer_signature(X_val, y_pred)

    mlflow.sklearn.save_model(
        model,
        export_path,
        signature=signature,
        serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=X_val.iloc[:5]
    )

    # Upload the model we just exported to W&B
    # HINT: use wandb.Artifact to create an artifact. Use args.output_artifact as artifact name, "model_export" as
    # type, provide a description and add rf_config as metadata. Then, use the .add_dir method of the artifact instance
    # you just created to add the "random_forest_dir" directory to the artifact, and finally use
    # run.log_artifact to log the artifact to the run

    
    # Plot feature importance
    fig_feat_imp = plot_feature_importance(model, processed_features)

