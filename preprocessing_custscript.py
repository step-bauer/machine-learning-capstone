
import argparse
import os
import warnings
import boto3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def scale_and_impute(X_train):
    """
    Description
    -----------
        this method scales data to [0,1] and imputes missing values


     Return
    ------
        pd.DataFrame : result scale and impute method
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    scaler  = StandardScaler()
    
    col_names = X_train.columns
    index = X_train.index

    # scale
    print('Scaling dataset')
    df = scaler.fit_transform(X_train)
    # impute
    print('Imputing missing values')
    df = pd.DataFrame(imputer.fit_transform(df))

    df.set_index(index, inplace=True)
    df.columns = col_names
    
    print('Scale and Impute transformation completed.')

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #parser.add_argument("--n_pca_components", type=int, default=10)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    customers_input_data_path = os.path.join("/opt/ml/processing/input", "customers_cleaned.csv")
    #customers_input_data_path = os.path.join("/opt/ml/processing/input2", "customers_cleaned.csv")

    print("Reading input data from {}".format(customers_input_data_path))    
    df_customers = pd.read_csv(customers_input_data_path)    
    print("Running preprocessing transformations for customers data")
    df_customers = scale_and_impute(df_customers)
    
    #print("Reading input data from {}".format(customers_input_data_path))
    #df_customers = pd.read_csv(customers_input_data_path)    
    #print("Running preprocessing transformations for customers data")
    #df_customers = scale_and_impute(df_customers)


    print("customers data shape after preprocessing: {}".format(df_customers.shape))
    #print("Customers data shape after preprocessing: {}".format(df_customers.shape))

    train_customers_output_path = os.path.join("/opt/ml/processing/train", "train_customers.csv")
    #train_customers_output_path = os.path.join("/opt/ml/processing/train", "train_customers.csv")

    print("Saving training features to {}".format(train_customers_output_path))
    pd.DataFrame(df_customers).to_csv(train_customers_output_path, header=False, index=False)
    
    #print("Saving test features to {}".format(train_customers_output_path))
    #pd.DataFrame(df_customers).to_csv(train_customers_output_path, header=False, index=False)
