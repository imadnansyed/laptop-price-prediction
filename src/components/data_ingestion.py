import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str= os.path.join('articfacts', 'raw.csv') 
    

class DataIngestion:
    def __init__(self):
        self.ingestion_cofig = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            df = pd.read_csv("notebook\data\laptop_data.csv")
            logging.info("Read the dataset as dataframe")
            
            # make dirs for the raw data path
            os.makedirs(os.path.dirname(self.ingestion_cofig.raw_data_path), exist_ok=True)
            logging.info("Created the directory for the raw data")
            
            # export the df to raw data path as csv
            df.to_csv(self.ingestion_cofig.raw_data_path, index=False)
            logging.info("Exported the data to raw data path")
            
            return(
                self.ingestion_cofig.raw_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
