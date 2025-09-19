import sys
import os
import pandas as pd
from src.utils.model_selection_utils import evaluate_pipeline
from src.utils.utils import save_obj
from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from category_encoders import TargetEncoder

@dataclass
class ModelSelectionConfig:
    ordinal_transformer_path: str = os.path.join('artifacts', 'ordinal_transformer.pkl')
    target_transformer_path: str = os.path.join('artifacts', 'target_transformer.pkl')

class ModelSelection:
    def __init__(self):
        self.config = ModelSelectionConfig()

    def initiate_model_selection(self, df_path: str):
        try:
            logging.info("Starting model selection process.")
            logging.info(f"Reading dataset from: {df_path}")

            df = pd.read_csv(df_path)
            logging.info(f"Dataset loaded successfully with shape: {df.shape}")

            X = df.drop(columns=['price', "weight"])
            y = df['price']
            logging.info("Split dataset into X (features) and y (target).")

            cat_columns = [x for x in X.columns if X[x].dtype == 'object']
            num_columns = [x for x in X.columns if X[x].dtype != 'object']
            logging.info(f"Categorical columns: {cat_columns}")
            logging.info(f"Numerical columns: {num_columns}")

            ordinal_preprocessor = ColumnTransformer(
                transformers=[
                    ("cat_encode", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_columns),
                    ("num_scale", StandardScaler(), num_columns)
                ],
                remainder="passthrough"
            )
            logging.info("Ordinal preprocessor created.")

            target_preprocessor = ColumnTransformer(
                transformers=[
                    ("cat_encode", TargetEncoder(), cat_columns),
                    ("num_scale", StandardScaler(), num_columns)
                ],
                remainder="passthrough"
            )
            logging.info("Target preprocessor created.")

            logging.info("Evaluating pipelines with ordinal preprocessor.")
            ordinal_results = evaluate_pipeline(ordinal_preprocessor, X, y)
            ordinal_results['Encoding'] = 'ordinal'

            logging.info("Evaluating pipelines with target preprocessor.")
            target_results = evaluate_pipeline(target_preprocessor, X, y)
            target_results['Encoding'] = 'target'

            # Combine both results
            all_results = pd.concat([ordinal_results, target_results], ignore_index=True)

            # Sort by R2 and pick top 4
            top_3_models = all_results.sort_values(by="R2", ascending=False).head(3)[["Model","Encoding"]]

            os.makedirs(os.path.dirname(self.config.ordinal_transformer_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.target_transformer_path), exist_ok=True)

            save_obj(self.config.ordinal_transformer_path, ordinal_preprocessor)
            save_obj(self.config.target_transformer_path, target_preprocessor)
            logging.info("Preprocessors saved successfully.")
            
            top_3_models = top_3_models.reset_index(drop=True)
            return top_3_models, self.config.ordinal_transformer_path, self.config.target_transformer_path

        except Exception as e:
            logging.error("Error occurred during model selection.", exc_info=True)
            raise CustomException(e, sys)
