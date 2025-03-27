import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

from src.project.feast_feature_store.feature_store import getTrainDataFromFeatureStore  
from src.project.data_transformation.pipeline import pipeline

#---------------------------------------------------------------------------------------------------------------




def train():
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
    # logger.info("Downloading training set artifact from feature store")

    # retrieve the training data from feature store
    # data = getTrainDataFromFeatureStore()

    # print(data.shape)

    data_path = Path(r"data\raw\AB_NYC_2019.csv")
    artifact_path = Path("src\project\prod\prod_artifacts")
    
    data = pd.read_csv(data_path)

    # df = pipeline(data, artifact_path , training=True)
    # print(df)

    df = pipeline(data, artifact_path , training=False)
    

    # read the training data
    X = df
    y = data["price"]  # this removes the column "price" from X and puts it into y

    # there are some features which are not required for model training 
    # like event_timestamp, 
    # X = X.drop(["event_timestamp"], axis = 1)
    
    # split the data into train and test data
    X_train, X_val, y_train, y_val = train_test_split(
                            X, y, test_size = 0.2,
                            random_state = 445
                        )


    # get te model for training 
    # get the configs for random forest model
    model = RandomForestRegressor(
                                n_estimators = 150 ,
                                max_depth = 15,
                                min_samples_split = 4,
                                min_samples_leaf = 3,
                                n_jobs = -1,
                                criterion = 'absolute_error',
                                max_features = 0.5,
                                oob_score =True
                                )


    model.fit(X_train, y_train)
    # # Then fit it to the X_train, y_train data
    # # Fit the pipeline full_pipeline (will transform and train the data )
    # #! per yaar data toh humara feature store se aaya hai toh pipeline se fit kyu kare
    # #! khali model se fit karna chaheye

    # # Compute r2 and MAE
    # logger.info("Scoring")
    # r_squared = model.score(X_val, y_val)

    # y_pred = model.predict(X_val)
    # mae = mean_absolute_error(y_val, y_pred)

    # logger.info(f"Score: {r_squared}")
    # logger.info(f"MAE: {mae}")

    # logger.info("Exporting model")

    # # Save model package in the MLFlow sklearn format
    # if os.path.exists("random_forest_dir"):
    #     shutil.rmtree("random_forest_dir")

    # # Save the sk_pipe pipeline as a mlflow.sklearn model in the directory "random_forest_dir"

    # export_path = "random_forest_dir"
    # # signature = infer_signature(X_val, y_pred)

    # # mlflow.sklearn.save_model(
    # #     model,
    # #     export_path,
    # #     signature=signature,
    # #     serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    # #     input_example=X_val.iloc[:5]
    # # )

    filename = 'F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//src//project//prod//prod_artifacts//random_forest_model'
    
    with open(filename, "wb") as file:
        joblib.dump(model, file)
    


    
    # # Plot feature importance
    # # fig_feat_imp = plot_feature_importance(model, processed_features)




if __name__ == "__main__":
    train()