
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import pickle




data = pd.read_csv("F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//data//processed//feature_store_data//training_data.csv")
X = data.sample(10000)
y = X.pop("price")  # this removes the column "price" from X and puts it into y

# there are some features which are not required for model training 
# like event_timestamp, 
X = X.drop(["event_timestamp"], axis = 1)

# print(X.columns)

# split the data into train and test data
X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size = 0.2,
                        random_state = 555
                        )

# print(X_train.columns)

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


filename = 'F://machine learning//mlops//end to end machine learning pipeline//MLOPs_workflow//src//project//prod//prod_artifacts//random_forest_model.pkl'
pickle.dump(model, open(filename, 'wb'))
 
 
# some time later...
 
# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# prediction = loaded_model.predict(X_test, Y_test)