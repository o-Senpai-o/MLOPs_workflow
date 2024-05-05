import uvicorn
import pickle
import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Annotated
import pandas as pd


from tempfile import NamedTemporaryFile
from starlette.responses import FileResponse
from starlette.requests import Request

from utils import load_feature_transformation_pipeline, load_models
from utils import delta_date_feature

from fastapi.responses import HTMLResponse


# from BankNote import nyc


app = FastAPI()



# HTML form for file upload
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>File Upload Form</title>
</head>
<body>
    <h2>Upload CSV File</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
</body>
</html>
"""




#-------------------- takes input csv file and return output--------------

# Annotated is used for type hints and it give additional metadata about the args 
# example suppose  the input expects a integer then we can say tye hint as int
# but what if it should be in the given specific range 
# this is where Annotated will be used
  
@app.post("/predict")
def predict(file: UploadFile):
    # recieve the file as input from the user
    data = pd.read_csv(file.file)
    # return {"columns" : data.columns.tolist()}
    
    feat_id = data['id']

    # tranform the data into features for our model
    feature_transform_pipeline , pipeline_features = load_feature_transformation_pipeline()

    transformed_feats = feature_transform_pipeline.transform(data)

    # get the machine learning model from mlflow registry
    random_model = load_models()

    new_df = pd.DataFrame(transformed_feats, columns=pipeline_features)
    df = pd.concat([feat_id, new_df], axis=1)

    # now make predictions on the data 
    predictions = random_model.predict(df)

    # we can return the predictions along with the data
    data["predictions"] = predictions

    # Create HTML table from DataFrame
    html_table = data.to_html(index=False)

    # Return HTML response with DataFrame table
    return HTMLResponse(content=html_table)


@app.get("/")
async def main():
    return HTMLResponse(content=html_form, status_code=200)



if __name__ == "__main__":
    # importing the user defined function here to overcome the 
    # AttributeError: Can't get attribute 'function' on <module '__main__' (built-in)>
    from utils import delta_date_feature

    # runs the app on localhost of container exposed at port 8000
    uvicorn.run(app = app, host = '0.0.0.0', port = 8000)








# @app.post('/predict')
# def hello():
#     return {"hello" : "whats iup"}

# @app.post("/predict")
# def predict(data:nyc):
#     # get all the features
#     data = data.dict()

#     # get all the features
#     test_data = get_all_feat_values(data)

#     # download the model from the mlflow registry
#     model = getModelFromRegistry()

#     # make predictions on the test data
#     prediction = model.predict(test_data)

#     return {"prediction" : prediction}
