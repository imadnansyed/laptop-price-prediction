import sys
import os
import pandas as pd
import numpy as np
from src.utils.logger import logging
from src.utils.exception import CustomException
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import shap



# Technique 1 : correlation
def correlation_feature_selection(df:pd.DataFrame)->pd.DataFrame:
    try:
        print("Calculating correlation feature selection")
        corr_df = df.corr()['Price'].to_frame().reset_index().rename(columns={'index':'feature','Price':'corr_coeff'})

        return(
            corr_df
        )
    except Exception as e:
        raise CustomException(e, sys)
    
# Technique 2 : Random Forest Regressor
def random_forest_feature_selection(X: pd.DataFrame, y: pd.Series)->pd.DataFrame:
    try:
        print("Calculating random forest feature selection")
        rf = RandomForestRegressor()
        rf.fit(X, y)
        
        random_forest_feature_selection_df = pd.DataFrame({
            "feature": X.columns,
            "rf_importance": rf.feature_importances_}).sort_values(by="rf_importance", ascending=False).reset_index(drop=True)

        return(
            random_forest_feature_selection_df
        )

        
    except Exception as e:
        raise CustomException(e, sys)
    
# Technique 3 - Gradient Boosting Feature importances
def gradient_boosting_feature_selection(X: pd.DataFrame, y: pd.Series)->pd.DataFrame:
    try:
        print("Calculating gboosting feature selection")
        gb = GradientBoostingRegressor()
        gb.fit(X, y)

        gradient_boosting_feature_selection_df = pd.DataFrame({
            "feature": X.columns,
            "gb_importance": gb.feature_importances_
        }).sort_values(by="gb_importance", ascending=False).reset_index(drop=True)
        
        
        return(
            gradient_boosting_feature_selection_df
        )
        
    except Exception as e:
        raise CustomException(e, sys)
    
    
# Technique 4 - Permutation Importance
def permutation_feature_selection(X: pd.DataFrame, y: pd.Series)->pd.DataFrame:
    try:
        print("Calculating permutation feature selection")
        X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest regressor on label encoded data
        rf_label = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_label.fit(X_train_label, y_train_label)

        # Calculate Permutation Importance
        perm_importance = permutation_importance(rf_label, X_test_label, y_test_label, n_repeats=30, random_state=42)

        # Organize results into a DataFrame
        perm_importance_df = pd.DataFrame({
            'feature': X.columns,
            'permutation_importance': perm_importance.importances_mean
        }).sort_values(by='permutation_importance', ascending=False)
    
        return (
            perm_importance_df
                )
    except Exception as e:
        raise CustomException(e, sys)
    

# Technique 5 - LASSO
def lasso_feature_selection(X: pd.DataFrame, y: pd.Series)->pd.DataFrame:
    try:
        print("Calculating lasso feature selection")
        X_scaled = StandardScaler().fit_transform(X)
        lasso = Lasso(alpha = 0.01)
        lasso.fit(X_scaled, y)
        
        # Extract coefficients
        lasso_feature_selection_df  = pd.DataFrame({
            'feature': X.columns,
            'lasso_coeff': lasso.coef_
        }).sort_values(by='lasso_coeff', ascending=False)
        
        return (
            lasso_feature_selection_df
        )
    
    except Exception as e:
        raise CustomException(e, sys)
    

# Technique 6 - RFE => Recursive Feature Elimination
def rfe_feature_selection(X: pd.DataFrame, y: pd.Series)->pd.DataFrame:
    try: 
        print("Calculating rfe feature selection")
        estimator = RandomForestRegressor()

        selector = RFE(estimator, n_features_to_select=X.shape[1], step=1)
        selector = selector.fit(X, y)

        selected_features = X.columns[selector.support_]

        selected_coeffs = selector.estimator_.feature_importances_



        rfe_feature_selection_df = pd.DataFrame({
            'feature': selected_features,
            'rfe_score': selected_coeffs
        }).sort_values(by='rfe_score', ascending=False)

        return(
            rfe_feature_selection_df
        )

    except Exception as e:
        raise CustomException(e, sys)
    
    
# Technique 7 - Linear Regression Weights
def linear_regression_weights_feature_selection(X: pd.DataFrame, y: pd.Series)->pd.DataFrame:
    try:
        print("Calculating linear reg feature selection")
        X_scaled = StandardScaler().fit_transform(X)
        lin_reg = LinearRegression()
        lin_reg.fit(X_scaled, y)

        # Extract coefficients
        linear_regression_weights_feature_selection_df = pd.DataFrame({
            'feature': X.columns,
            'reg_coeffs': lin_reg.coef_
        }).sort_values(by='reg_coeffs', ascending=False)

        return (
            linear_regression_weights_feature_selection_df
        )
        
    except Exception as e:
        raise CustomException(e, sys)
    

# Technique 8 - SHAP
def shap_feature_selection(X: pd.DataFrame, y: pd.Series)->pd.DataFrame:
    try:
        print("Calculating shap feature selection")
        rf = RandomForestRegressor(n_estimators = 100, random_state=42)
        rf.fit(X, y)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        
        shap_feature_selection_df = pd.DataFrame({
            'feature': X.columns,
            'SHAP_score': np.abs(shap_values).mean(axis=0)
        }).sort_values(by='SHAP_score', ascending=False)
        
        return (
            shap_feature_selection_df
        )
        
    
    except Exception as e:
        raise CustomException(e, sys)