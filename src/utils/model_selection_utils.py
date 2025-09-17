from src.utils.exception import CustomException
import pandas as pd
import sys
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline



def evaluate_pipeline(preprocessor, X, y):
    """
    This will return a Dataframe for the preprocessor pipepline with Columns :
    "Model", "R2", "MAE"
    """
    try :
        models = {
            "linear_regression" : LinearRegression(),
            "random_forest" : RandomForestRegressor(),
            "gradient_boosting" : GradientBoostingRegressor(),
            "svr" : SVR(),
            "xgboost" : XGBRegressor()
            }

        results = []

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

            y_pred = cross_val_predict(pipeline, X, y, cv=kf, n_jobs=-1)

            r2 = r2_score(y,y_pred)

            mae = mean_absolute_error(y,y_pred)

            results.append((name, r2, mae))

        return pd.DataFrame(results, columns=["Model", "R2", "MAE"])
    
    except Exception as e:
        raise CustomException(e, sys)