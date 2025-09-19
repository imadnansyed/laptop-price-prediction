from dataclasses import dataclass
import os, sys, pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from src.utils.hyper_para_tuning_utils import hyperparameter_tuning
from src.utils.exception import CustomException
from src.utils.logger import logging
import pandas as pd

from src.utils.utils import save_obj

@dataclass
class HyperParatuningConfig:
    prediction_pipeline_path: str = os.path.join("artifacts", "pipelines", "prediction_pipeline.pkl")
    best_preprocessor_pipeline_path: str = os.path.join("artifacts", "pipelines", "best_preprocessor_pipeline.pkl")

class HyperParaTuning:
    def __init__(self):
        self.config = HyperParatuningConfig()

    def tune_and_select_best(self, top_3_models_df, preprocessors_paths, df_path: str):
        """
        Loops through top 3 models, loads dataset and preprocessors,
        performs hyperparameter tuning, and saves the best pipeline.
        """
        try:
            # 1. Load dataset
            df = pd.read_csv(df_path)
            X = df.drop(columns=['price', "weight"])
            y = df['price']

            # 2. Load preprocessors
            ordinal_preprocessor = pickle.load(open(preprocessors_paths[0], 'rb'))
            target_preprocessor = pickle.load(open(preprocessors_paths[1], 'rb'))
            
            preprocessors = {
                "ordinal": ordinal_preprocessor,
                "target" : target_preprocessor
            }

            # 3. Choose param grids
            param_grids = {
                "linear_regression": {},
                "random_forest": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [None, 10, 20]
                },
                "gradient_boosting": {
                    "model__n_estimators": [100, 200],
                    "model__learning_rate": [0.05, 0.1]
                },
                "svr": {
                    "model__C": [1.0, 10.0],
                    "model__kernel": ["rbf", "linear"]
                },
                "xgboost": {
                    "model__n_estimators": [100, 200],
                    "model__learning_rate": [0.05, 0.1]
                }
            }

            model_map = {
                "linear_regression": LinearRegression(),
                "random_forest": RandomForestRegressor(random_state=42),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
                "svr": SVR(),
                "xgboost": XGBRegressor(random_state=42)
            }

            best_score = -float("inf")
            best_pipeline = None

            for i in range(len(top_3_models_df)):
                logging.info(f"Hyperparameter tuning for {top_3_models_df.loc[0, "Model"]}...")
                print(f"Hyperparameter tuning for {top_3_models_df.loc[0, "Model"]}...")
                model_name = top_3_models_df.loc[i, "Model"]
                encoding = top_3_models_df.loc[i, "Encoding"]
                preprocessor = preprocessors[encoding]
                
                model = model_map[model_name]
                param_grid = param_grids[model_name]

                best_params, best_estimator, score = hyperparameter_tuning(
                    preprocessor, model, param_grid, X, y
                )
                
                logging.info(f"{model_name} with {preprocessor} best params: {best_params}, R2: {score}")

                if score > best_score:
                    best_score = score
                    best_pipeline = best_estimator
                    
            os.makedirs(os.path.dirname(self.config.prediction_pipeline_path), exist_ok=True)
            save_obj(self.config.prediction_pipeline_path, best_pipeline)
            
            best_preprocessor = best_estimator.named_steps["preprocessor"]
            save_obj(self.config.best_preprocessor_pipeline_path, best_preprocessor)

            logging.info(f"Prediction pipeline saved at {self.config.prediction_pipeline_path}")

            return best_pipeline, best_score

        except Exception as e:
            logging.error("Error occurred during hyperparameter tuning.", exc_info=True)
            raise CustomException(e, sys)
        

