import pandas as pd
import numpy as np


def delta_date_feature(dates):
    """
    Computes the number of days between each date in `dates` and the most recent date.
    
    Parameters:
    -----------
    dates : array-like
        A 2D array (or Pandas Series) containing dates.

    Returns:
    --------
    np.ndarray
        A reshaped array of delta days.
    """
    # Ensure input is a DataFrame with the correct column name
    df = pd.DataFrame(dates, columns=["last_review"])

    # Convert to datetime, handling errors gracefully
    df["last_review"] = pd.to_datetime(df["last_review"], format="%Y-%m-%d", errors="coerce")

    # Compute max date while avoiding NaT issues
    max_date = df["last_review"].max()

    if pd.isna(max_date):  # If all values are NaT, return a default fill value
        return np.full((len(df), 1), fill_value=-1)  # -1 can indicate missing values

    # Vectorized computation of delta in days
    delta_days = (max_date - df["last_review"]).dt.days.fillna(0).to_numpy().reshape(-1, 1)

    return delta_days

def ravel_text_column(x):
    return x.ravel()