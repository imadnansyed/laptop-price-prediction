import sys
import os
import pandas as pd
import numpy as np
from src.utils.logger import logging
from src.utils.exception import CustomException
from dataclasses import dataclass


# config class
@dataclass
class OutlierTreatmentConfig:
    outlier_treatment_path : str = os.path.join('artifacts', 'post_outlier_treatment.csv')

# working class

class OutlierTreatment:
    def __init__(self):
        self.outlier_treatment_config = OutlierTreatmentConfig()
        
    def treat_outliers(self, df):
        
        """
        Here no outliers were found in the numerical features except price feature. So, we will treat the outliers in price feature using IQR method.
        """
        
        try:
            # column names to lower case
            df.columns = df.columns.str.lower()
            
            # Handling price feature
            
            ## price type conversion
            df["price"] = df["price"].astype(int)
            
            ## IQR method
            Q1 = df["price"].quantile(0.25)
            Q3 = df["price"].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers from price column
            df = df[~((df["price"] < lower_bound) | (df["price"] > upper_bound))]
            
            # Log transformation on price column
            df["price"] = np.log(df["price"])
            
            return (
                df
            )     
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_outlier_treatment(self, df_path):
        try:
            # load dataset
            df = pd.read_csv(df_path)
            logging.info("Dataset loaded successfully for outlier treatment")
            
            # pass to treat_outliers function
            outlier_treated_df = self.treat_outliers(df)
            logging.info("Outlier treatment completed successfully")
            
            # save the treated dataset to artifacts folder
            os.makedirs(os.path.dirname(self.outlier_treatment_config.outlier_treatment_path), exist_ok=True)
            outlier_treated_df.to_csv(self.outlier_treatment_config.outlier_treatment_path, index=False)
            logging.info("Outlier treated dataset saved successfully")
            
            # return the outlier_treated_dataset of the treated dataset
            return (
                self.outlier_treatment_config.outlier_treatment_path
                    )
        
        except Exception as e:
            raise CustomException(e, sys)