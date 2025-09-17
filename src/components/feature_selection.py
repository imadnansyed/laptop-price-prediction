import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.feature_selection_utils import correlation_feature_selection, random_forest_feature_selection, gradient_boosting_feature_selection, permutation_feature_selection, rfe_feature_selection, linear_regression_weights_feature_selection, lasso_feature_selection, shap_feature_selection
from sklearn.preprocessing import OrdinalEncoder



# config class for feature selection
@dataclass
class FeatureSelectionConfig:
    post_feature_selection_path: str = os.path.join('artifacts', 'post_feature_selection.csv')

# working class for missing values imputation
class FeatureSelection:
    def __init__(self):
        self.config = FeatureSelectionConfig()

    def feature_selection(self, df: pd.DataFrame):
        
        try:    
            data_label_encoded = df.copy()
            
            cat_columns=[x for x in data_label_encoded.columns if data_label_encoded[x].dtype == 'object']
            for col in cat_columns:
                oe = OrdinalEncoder()
                data_label_encoded[col] = oe.fit_transform(data_label_encoded[[col]])

            # Splitting the dataset into training and testing sets
            y_label = data_label_encoded['Price']
            X_label = data_label_encoded.drop('Price', axis=1)


            selectors = [
                correlation_feature_selection,
                random_forest_feature_selection,
                gradient_boosting_feature_selection,
                permutation_feature_selection,
                lasso_feature_selection,
                rfe_feature_selection,
                linear_regression_weights_feature_selection,
                shap_feature_selection,
            ]
            
            merged = selectors[0](data_label_encoded)
            for selector in selectors[1:]:
                merged = merged.merge(selector(X=X_label, y=y_label), on='feature')

            merged.set_index('feature', inplace=True)
            mean_scores = merged.mean(axis=1)
            low_features = mean_scores.nsmallest(3).index.tolist()

            logging.info(f"Dropping low-importance features: {low_features}")
            return df.drop(columns=low_features), low_features, merged

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_feature_selection(self, df_path):
        try:
            df = pd.read_csv(df_path)
            logging.info("dataset loaded for feature selection")

            filtered_df, low_features, merged_scores = self.feature_selection(df)

            os.makedirs(os.path.dirname(self.config.post_feature_selection_path), exist_ok=True)
            filtered_df.to_csv(self.config.post_feature_selection_path, index=False)
            logging.info(f"post feature selection dataset saved in path {self.config.post_feature_selection_path}")

            return (
                self.config.post_feature_selection_path
            )
        except Exception as e:
            raise CustomException(e, sys)