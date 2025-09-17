import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging


@dataclass
class DataCleaningConfig:
    cleaned_data_path:str = os.path.join('artifacts', 'cleaned.csv')

class DataCleaning:
    def __init__(self):
        self.cleaning_config = DataCleaningConfig()
        
    def data_cleaning(self, df:pd.DataFrame)->pd.DataFrame:
        
        """
        This function will clean the data and return the cleaned dataframe.
        """
        
        try:
            # removing unnamed column if exists
            df = df.drop(columns=['Unnamed: 0'])
            
            # cleaning Ram feature
            df["Ram"] = df["Ram"].str.replace("GB", "")
            
            # cleaning Weight feature
            df["Weight"] = df["Weight"].str.replace("kg", "")
            
            # type Conversion of Ram and Weight feature
            df["Ram"] = df["Ram"].astype(int)
            df["Weight"] = df["Weight"].astype(float)
            
            return(df)
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_cleaning(self, data_path):
        try:
            # read the data from the data_path
            df = pd.read_csv(data_path)
            logging.info(f"Read the data from the {data_path}")
            
            # pass to data_cleaning function
            cleaned_df = self.data_cleaning(df)
            logging.info(f"Data cleaning completed. Cleaned data saved at {self.cleaning_config.cleaned_data_path}")
            
            # saving cleaned df
            os.makedirs(os.path.dirname(self.cleaning_config.cleaned_data_path), exist_ok=True)
            cleaned_df.to_csv(self.cleaning_config.cleaned_data_path, index=False)
            
            return (
                self.cleaning_config.cleaned_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)