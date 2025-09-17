import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging

@dataclass
class FeatureEngineeringConfig:
    post_feature_engg_data_path: str = os.path.join('artifacts', 'post_feature_engg_data.csv')

class FeatureEngineering:
    def __init__(self):
        self.feature_engg_config = FeatureEngineeringConfig()

    def feature_engineering(self, df:pd.DataFrame)-> pd.DataFrame:
        
    
        """
        This  function will perform feature engineering on the given dataframe.
        """
        try:
            # Extracting Touch Screen feature
            df["ips"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0)
            
            # Extracting IPS feature
            df["touch_screen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)
            
            # Extacting PPI feature
            df[["x_res", "y_res"]] = df["ScreenResolution"].str.split("x", n=1, expand =True)
            df['x_res'] = df['x_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])
            df['x_res'] = df['x_res'].astype('int')
            df['y_res'] = df['y_res'].astype('int')
            df['ppi'] = (((df['x_res']**2) + (df['y_res']**2))**0.5/df['Inches']).astype('float')
            

            
            # cleaning CPU feature
            df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
            
            # function to fetch processor type
            def fetch_processor(text):
                if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
                    return text
                else:
                    if text.split()[0] == 'Intel':
                        return 'Other Intel Processor'
                    else:
                        return 'AMD Processor'

            df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
            
            ## working with Memory Feature
            
            # 1. SSD
            ssd_match = df["Memory"].str.extract(r'(\d+)(GB|TB)\sSSD')
            df["ssd"] = np.where(
                ssd_match[1] == "TB",
                ssd_match[0].astype(float) * 1024,  # TB → GB
                ssd_match[0].astype(float))          # GB stays as number

            # 2. HDD
            hdd_match = df["Memory"].str.extract(r'(\d+(?:\.\d+)?)(GB|TB)\s(?:HDD|Hybrid)')
            df["hdd"] = np.where(
                hdd_match[1] == "TB",
                hdd_match[0].astype(float) * 1024,
                hdd_match[0].astype(float)
            )
            
            # 3. Flash 
            flash_match = df["Memory"].str.extract(r'(\d+(?:\.\d+)?)(GB|TB)\sFlash')
            df["flash"] = np.where(
                flash_match[1] == "TB",
                flash_match[0].astype(float) * 1024,  # TB → GB
                flash_match[0].astype(float)
            )
            
            # filling NaN values with 0
            df.fillna(0, inplace=True)
            
            # converting to int
            df[['ssd', 'hdd', 'flash']]=df[['ssd', 'hdd', 'flash']].astype('int')
            
            # working with GPU feature
            df["gpu_name"] = df["Gpu"].str.split(" ", n=1, expand =True)[0]
            df = df[df['gpu_name'] != 'ARM'] # ARM have only one value
            
            ## working with OS feature
            def cat_os(inp):
                if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
                    return 'Windows'
                elif inp == 'macOS' or inp == 'Mac OS X':
                    return 'Mac'
                else:
                    return 'Others/No OS/Linux'
                
            df['os'] = df['OpSys'].apply(cat_os)
            
            
            # dropping unnecessary cols
            df = df.drop(columns= ["Gpu", "x_res", "y_res","OpSys" ,"Inches", "ScreenResolution", 'Cpu','Cpu Name', "Memory"])
            
            return(
                df
            )  
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_feature_engineering(self, df_path):
        
        try:
            # loading dataset
            df = pd.read_csv(df_path)
            logging.info(f"df Loaded from {df_path}")
            
            # passing df to feature_engineering function
            post_feature_engineering_df = self.feature_engineering(df)
            
            # creating dir and saving df
            os.makedirs(os.path.dirname(self.feature_engg_config.post_feature_engg_data_path), exist_ok=True)
            post_feature_engineering_df.to_csv(self.feature_engg_config.post_feature_engg_data_path, index=False)
            
            return (
                self.feature_engg_config.post_feature_engg_data_path
            )   
                 
        except Exception as e:
            raise CustomException(e, sys)