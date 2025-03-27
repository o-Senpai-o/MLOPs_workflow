import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import TransformerMixin, BaseEstimator
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer





class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    """Applies Ordinal Encoding to a specified column and persists the fitted encoder."""

    def __init__(self):
        """
        Initializes the transformer with a specific column to encode.

        Parameters:
        -----------
        column_name : str
            The name of the column to apply ordinal encoding.
        """
        self.encoder = OrdinalEncoder()

    def fit(self, X, column_name):
        """
        Fits the ordinal encoder on the specified column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe containing the column to be encoded.
        """
        if column_name not in X:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        self.encoder.fit(X[[column_name]])
        return self

    def transform(self, X, column_name):
        """
        Transforms the specified column using the fitted encoder.

        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe containing the column to be transformed.

        Returns:
        --------
        np.ndarray
            Transformed column as a NumPy array.
        """
        if column_name not in X:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        # no extra column will be created here as we are transforming only one column
        x = self.encoder.transform(X[[column_name]])
        return pd.DataFrame(x, columns = ['room_type'])

    def fit_transform(self, X, column_name):
        """Fits and transforms the data in one step."""
        return self.fit(X, column_name).transform(X, column_name)

    def save(self, path):
        """
        Saves the fitted encoder to a file.

        Parameters:
        -----------
        path : str
            The file path to save the encoder.
        """
        artifact_path = path / "ordinal_encoder.pkl"
        path.mkdir(parents=True, exist_ok=True)
        
      
        with open(artifact_path, "wb") as f:
            pickle.dump(self.encoder, f)
            print("file saved")


    @staticmethod
    def load(path):
        """
        Loads a previously saved encoder.

        Parameters:
        -----------
        path : str
            The file path from where to load the encoder.
        column_name : str
            The column name for which the encoder was originally created.

        Returns:
        --------
        OrdinalEncoderTransformer
            A new instance of OrdinalEncoderTransformer with the loaded encoder.



        usage : 
            # Load the trained encoder
            loaded_transformer = OrdinalEncoderTransformer.load("ordinal_encoder.pkl", column_name="room_type")
        """

        artifact_path = path.joinpath("ordinal_encoder.pkl")
        with open(artifact_path, "rb") as f:
            loaded_encoder = pickle.load(f)
        
        # create a new instance with the column name and loaded encoder
        # return the new instance to the caller
        transformer = OrdinalEncoderTransformer()
        transformer.encoder = loaded_encoder
        return transformer


class NonOrdinalCategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for handling non-ordinal categorical features.
    1. Imputes missing values using the most frequent category.
    2. Applies One-Hot Encoding with `handle_unknown="ignore"` to avoid errors during inference.

    Attributes:
    ------------
    imputer : SimpleImputer
        Imputer for handling missing categorical values.
    encoder : OneHotEncoder
        One-hot encoder for transforming categorical values.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X, column_name):
        """
        Fits the imputer and the one-hot encoder.
        """
        out = self.imputer.fit_transform(X[column_name].values.reshape(-1, 1))  # Fit imputer
        # print(X[column_name])
        self.encoder.fit(out)  # Fit encoder on imputed data
        return self  # Return self for method chaining

    def transform(self, X, column_name):
        """
        Applies imputation and one-hot encoding to the data.
        """
        out = self.imputer.transform(X[column_name].values.reshape(-1, 1))# Impute missing values
        X_encoded = self.encoder.transform(out)  # One-hot encode
        # print(X_encoded.shape)
        features = self.get_feature_names(column_name)
        print(features)
        # column_list = [col[0] for col in features]
        return pd.DataFrame(X_encoded, columns=features)
        

    def fit_transform(self, X, column_name):
        """
        Combines fit and transform for efficiency.
        """
        return self.fit(X, column_name).transform(X, column_name)

    def get_feature_names(self, column_name):
        """
        Returns the feature names for the encoded categories.
        """
        return self.encoder.get_feature_names_out([column_name])

    def save(self, path):
        """
        Saves the transformer (including imputer and encoder) as a pickle file.
        """

        # save the imputer first and then the encode

        imputer_path = path / "imputer.pkl"
        encoder_path = path / "encoder.pkl"


        path.mkdir(parents=True, exist_ok=True)

        with open(imputer_path, "wb") as f:
            pickle.dump(self.imputer, f)

        with open(encoder_path, "wb") as f:
            pickle.dump(self.encoder, f)

    @staticmethod
    def load(path):
        """
        Loads a saved transformer from a pickle file.
        """

        # from the current class's artifact folder get the artifacts 
        for artifact in path.iterdir():
            if "imputer" in str(artifact):
                with open(artifact, "rb") as f:
                    imputer_pkl = pickle.load(f)
            else:
                with open(artifact, "rb") as f:
                    encoder_pkl = pickle.load(f)
            

        
        # create a new instance with the loaded pickle file 
        non_ordinal_categorical_transformer = NonOrdinalCategoricalTransformer()
        non_ordinal_categorical_transformer.imputer = imputer_pkl
        non_ordinal_categorical_transformer.encoder = encoder_pkl

        # return the instance of this class with arguments already loaded
        # this can be directly used for transformation
        # example provided below 
        return non_ordinal_categorical_transformer


class DeltaDatetimeFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        """ we need a imputer to fill null values and a function transformer to transform the date"""
        self.imputer = SimpleImputer(strategy='constant', fill_value='2010-01-01')
        self.max_date = None
        
        

    def fit(self, X, column_name):
        X_copy = X.copy()
        
        # Convert column to string before applying SimpleImputer
        X_copy[column_name] = X_copy[column_name].astype(str)
        
        # Impute missing values with "2010-01-01"
        X_copy[column_name] = self.imputer.fit_transform(X_copy[[column_name]]).ravel()

        # Convert back to datetime
        X_copy[column_name] = pd.to_datetime(X_copy[column_name], format="%Y-%m-%d", errors="coerce")

        # Compute max date while handling NaT values
        self.max_date = np.max(X_copy[column_name])
        return self
    
    def transform(self, X, column_name):
        # now caluclate the delta date
        data_copy = X.copy()
        # use the imputer to fill null values
        data_copy['last_review_date'] = self.imputer.fit_transform(data_copy['last_review'].values.reshape(-1, 1)).reshape(-1,)

        # Convert to datetime, handling errors gracefully
        data_copy['last_review_date'] = pd.to_datetime(data_copy['last_review_date'], format="%Y-%m-%d", errors="coerce")

        if pd.isna(self.max_date):  # If all values are NaT, return a default fill value
            return np.full((len(data_copy), 1), fill_value=-1)  # -1 can indicate missing values

        # Vectorized computation of delta in days
        delta_days = (self.max_date - data_copy['last_review_date']).dt.days.fillna(0).values
        return pd.DataFrame(delta_days,columns=["days_from_max_date"])    
    
    def fit_transform(self, X, column_name):
        return self.fit(X, column_name).transform(X, column_name)
    
    def save(self, path):
        """ save the artifacts of the model"""

        imputer_path = path / "imputer.pkl"
        max_date_val = path / "max_value.pkl"

        path.mkdir(parents=True, exist_ok=True)

        with open(imputer_path, "wb") as f:
            pickle.dump(self.imputer, f)
        
        with open(max_date_val, "wb") as f:
            pickle.dump(self.max_date, f)
    
    
    @staticmethod
    def load(path):
        
        for artifact in path.iterdir():
            print(artifact)
            if "imputer" in str(artifact):
                with open(artifact, "rb") as f:
                    imputer = pickle.load(f)
            else:
                with open(artifact, "rb") as f:
                    max_value = pickle.load(f)
        
        # create a new instance of the class
        delta_date_feature = DeltaDatetimeFeature()
        delta_date_feature.imputer = imputer   
        delta_date_feature.max_date = max_value

        return delta_date_feature


class tfidfVectorizerCompute(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy="constant", fill_value="")
        self.vectorizer = TfidfVectorizer(binary=False, max_features=5, stop_words='english')

    def fit(self, X, column_name):
        new = self.imputer.fit_transform(X[[column_name]])        # or [[]] --> .reshape(-1, 1)
        self.vectorizer.fit(new.ravel())
        return self
    
    def transform(self, X, column_name):
        data_copy = X.copy()
        new = self.imputer.transform(data_copy[[column_name]])
        print(new.shape)
        vectorized = self.vectorizer.transform(new.flatten())
        
        # vectorized are csr_matrix, we need to convert it into a dataframe 
        return pd.DataFrame(vectorized.toarray(), columns=self.get_feature_names())
    
    def get_feature_names(self,):
        return self.vectorizer.get_feature_names_out().tolist()
    
    def fit_transform(self, X, column_name):
        return self.fit(X, column_name).transform(X, column_name)

    def save(self, path):
        imputer_path = path / "imputer.pkl"
        vectorizer_path = path / "vectorizer.pkl"

        path.mkdir(parents=True, exist_ok=True)
        
        with open(imputer_path, "wb") as f:
            pickle.dump(self.imputer, f)
        
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @staticmethod
    def load(path):
        for artifact in path.iterdir():
            print(artifact)
            if "imputer" in str(artifact):
                with open(artifact, "rb") as f:
                    imputer = pickle.load(f)
            else:
                with open(artifact, "rb") as f:
                    vectorizer = pickle.load(f)
        
        tfidfVectorizerobj = tfidfVectorizerCompute()
        tfidfVectorizerobj.imputer = imputer
        tfidfVectorizerobj.vectorizer = vectorizer

        return tfidfVectorizerobj