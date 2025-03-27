
from fastapi import FastAPI, UploadFile
import pandas as pd
from fastapi.responses import HTMLResponse
from utils import * 








app = FastAPI()

# curl -X POST -F "file=@AB_NYC_2019.csv" http://localhost:80/predict
# wget --post-file=test.csv --output-document=- http://localhost:80/predict



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
BASE_DIR = Path(__file__).resolve().parent



@app.post("/predict")
def predict(file: UploadFile):
    print(f"Received file: {file.filename}")
    try:
        data = pd.read_csv(file.file)
        print("CSV read successfully!")
    except Exception as e:
        return {"error": f"Failed to read CSV: {str(e)}"}

   
    # ---------------- Pipeline-------------------------

    new_data = pipeline(df= data, path = BASE_DIR / "prod_artifacts", training=False)


    #----------------- model ---------------------------
    model = load_models()



    # -----------------Prediction---------------------- 
    predictions = model.predict(new_data)
    
    new_data["predictions"] = predictions

    #------------------Response ------------------------
   

    # Create HTML table from DataFrame
    html_table = new_data.to_html(index=False)
    print("converted to html table")


    # Return HTML response with DataFrame table
    # return HTMLResponse(content=html_table)
    return {"columns" : new_data['predictions'].values.tolist()}
    


@app.get("/")
async def main():
    return HTMLResponse(content=html_form, status_code=200)



# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)




# some debugging tricks :
# run commands inside docker 
# docker exec -it <container_id> /bin/bash
# docker run -it --entrypoint /bin/bash mlops-application 
# python -X faulthandler test_pipeline.py
