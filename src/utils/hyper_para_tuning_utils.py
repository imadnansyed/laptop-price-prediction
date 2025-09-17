from src.utils.exception import CustomException
from src.utils.logger import logging
import pandas as pd
from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
import sys
import os

def hyperparameter_tuning(preprocessor, model, param_grid, X, y):
    
    try:
        """
        Run hyperparameter tuning for a single model + preprocessor.
        Returns (best_params, best_estimator, best_score).
        """
        pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                ("model", model)])
        cv = KFold(n_splits=5, random_state=42, shuffle=True)

        scoring = {
            'r2': 'r2',
            'mae': make_scorer(mean_absolute_error, greater_is_better=False)
        }

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            refit='r2',
            n_jobs=-1
        )

        grid.fit(X, y)

        return grid.best_params_, grid.best_estimator_, grid.best_score_
    
    except Exception as e:
        raise CustomException(e, sys)