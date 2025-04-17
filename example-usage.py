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
    if model_type.lower