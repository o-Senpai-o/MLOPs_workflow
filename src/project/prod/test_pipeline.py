import pandas as pd
from utils import pipeline
import pandas as pd
import joblib
from pathlib import Path

# artifact_path = Path("src\project\prod\prod_artifacts")
BASE_DIR = Path(__file__).resolve().parent
artifact_path = BASE_DIR / "prod_artifacts"
print("artifact path", artifact_path)

model_path = artifact_path / "random_forest_model"


def testing_pipeline():
    
    # pipeline, _, _ = get_feature_transformation_pipeline()
    # test_data = pd.read_csv("src\project\prod\AB_NYC_2019.csv")           ----> local 
    test_data = pd.read_csv("AB_NYC_2019.csv")                             #----> docker 

    df = pipeline(test_data, path = artifact_path, training = False)

    print(f"Pipeline loaded successfully:")
    print(df.head)


    # loading the model 
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    
    print(model)

    print(model.predict(df))
    return ""













if __name__ == "__main__":
    testing_pipeline()




