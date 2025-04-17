#!/usr/bin/env python
# coding: utf-8

# Example file demonstrating how to reuse common methods
# This file shows how to use the extracted common methods in other projects

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
import utils

def run_house_price_custom_model(data_path, model_type="ridge", alpha=1.0):
    """Run house price prediction pipeline with different models"""
    # Set up environment
    utils.setup_environment(seed=123)  # Custom random seed
    
    print(f"Running house price prediction with {model_type} model, alpha={alpha}")
    
    # Load data
    df = utils.load_and_overview_data(data_path)
    
    # Perform data preprocessing (reusing utility functions)
    df = utils.remove_outliers(df, [
        "((df['GrLivArea'] < 4000) | (df['SalePrice'] > 300000))",
        "(df['SalePrice'] < 500000)"
    ])
    df = utils.fill_missing_values(df)
    df = utils.create_features(df)
    
    # Transform target
    df, target_column = utils.transform_target(df, 'SalePrice')
    
    # Prepare features
    numeric_features = [c for c in df.select_dtypes(include=['int64', 'float64']).columns 
                       if c != target_column and c != 'SalePrice']
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    df, numeric_features, categorical_features = utils.prepare_features(
        df, numeric_features, categorical_features
    )
    
    # Set X and y
    y = df[target_column]
    X = df[numeric_features + categorical_features]
    
    # Create preprocessing pipeline
    preprocessor = utils.create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Choose model based on type
    if model_type.lower() == "ridge":
        from sklearn.linear_model import Ridge
        model_class = Ridge
        model_params = {"alpha": alpha}
    elif model_type.lower() == "lasso":
        from sklearn.linear_model import Lasso
        model_class = Lasso
        model_params = {"alpha": alpha}
    else:
        from sklearn.linear_model import LinearRegression
        model_class = LinearRegression
        model_params = {}
    
    # Create model pipeline
    pipeline = utils.create_model_pipeline(preprocessor, model_class, model_params)
    
    # Split data
    X_train, X_test, y_train, y_test = utils.split_data(X, y)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    results = utils.evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    
    # Save model
    model_filename = f"pkl/house_price_{model_type}_pipeline.pkl"
    utils.save_model(pipeline, model_filename)
    
    print(f"\n✅ Custom model pipeline completed successfully with {model_type}")
    return pipeline, results

# Example call to the function  
if __name__ == "__main__":
    # Example with Ridge regression
    run_house_price_custom_model("dataset.csv", "ridge", alpha=0.5)
    
    # Example with Lasso regression
    # run_house_price_custom_model("dataset.csv", "lasso", alpha=0.01)