import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.components.hyper_parameter_tuning import HyperParaTuning
from src.components.data_cleaning import DataCleaning
from src.components.feature_engg import FeatureEngineering
from src.components.feature_selection import FeatureSelection
from src.components.model_selection import ModelSelection
from src.components.outlier_treatment import OutlierTreatment
from src.utils.exception import CustomException
from src.utils.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str= os.path.join('artifacts', 'raw.csv') 
    

class DataIngestion:
    def __init__(self):
        self.ingestion_cofig = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            df = pd.read_csv("notebook\\data\\laptop_data.csv")
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


if __name__ == "__main__":
    DI_obj = DataIngestion()
    raw_path = DI_obj.initiate_data_ingestion()
    
    DC_obj = DataCleaning()
    cleaned_path = DC_obj.initiate_data_cleaning(raw_path)
    
    FE_obj = FeatureEngineering()
    post_fe_path = FE_obj.initiate_feature_engineering(cleaned_path)
    
    FS_obj = FeatureSelection()
    post_fs_path = FS_obj.initiate_feature_selection(post_fe_path)
    
    OT_obj = OutlierTreatment()
    post_ot_path = OT_obj.initiate_outlier_treatment(post_fs_path)
    
    MS_obj = ModelSelection()
    top_3_models_df, ordinal_preprocessor_path, target_preprocessor_path = MS_obj.initiate_model_selection(post_ot_path)
    
    HPT_obj = HyperParaTuning()
    preprocessors_paths = [ordinal_preprocessor_path, target_preprocessor_path]
    best_model, best_score = HPT_obj.tune_and_select_best(top_3_models_df, preprocessors_paths, post_ot_path)
    
    print(best_model)
    print(best_score)