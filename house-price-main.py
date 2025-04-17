#!/usr/bin/env python
# coding: utf-8

# 🏠 House Price Prediction Pipeline
# Author: Randy Jin

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Import utility functions
import utils

def run_house_price_prediction_pipeline(data_path=f"data/dataset.csv"):
    """Run the complete house price prediction pipeline"""
    # Set up environment
    utils.setup_environment()
    
    # Load and overview data
    df = utils.load_and_overview_data(data_path)
    
    # Exploratory Data Analysis
    utils.plot_distribution(df, 'SalePrice')
    utils.plot_distribution(df, 'SalePrice', log_transform=True)
    utils.plot_correlation_heatmap(df)
    sale_price_corr = utils.plot_top_correlations(df, 'SalePrice')
    
    # Select top 4 features most correlated with SalePrice (excluding SalePrice itself) for visualization
    top_features = sale_price_corr.index[:5][1:]  # Exclude SalePrice itself
    utils.plot_scatter_features_vs_target(df, top_features, 'SalePrice', "top_features_scatter.png")
    
    # Analyze categorical features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical features: {categorical_features}")
    utils.plot_categorical_features(df, categorical_features[:4], 'SalePrice', "categorical_features_boxplot.png")
    
    # Feature Engineering
    # Remove outliers
    df = utils.remove_outliers(df, [
        "((df['GrLivArea'] < 4000) | (df['SalePrice'] > 300000))",
        "(df['SalePrice'] < 500000)"
    ])
    
    # Fill missing values for important fields
    df = utils.fill_missing_values(
        df,
        numeric_cols=['LotArea', 'GrLivArea', 'TotalBsmtSF'],
        categorical_cols=['Alley', 'GarageType', 'MasVnrType', 'Fence', 'FireplaceQu', 'MiscFeature']
    )
    
    # Create new features
    df = utils.create_features(df)
    
    # Transform target variable
    df, target_column = utils.transform_target(df, 'SalePrice', 'log')
    
    # Define features for modeling
    original_numeric_features = [
        'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'FullBath', 'HalfBath', 
        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
        'LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'WoodDeckSF', 
        'OpenPorchSF', 'MasVnrArea', 'BsmtFinSF1', 'YearSold'
    ]
    
    engineered_features = [
        'LotArea_Log', 'GrLivArea_Log', 'TotalBsmtSF_Log', 'TotalSF_Log',
        'HouseAge', 'TotalBath', 'HasGarage', 'TotalSF'
    ]
    
    all_numeric_features = original_numeric_features + engineered_features
    
    # Prepare features
    df, numeric_features, categorical_features = utils.prepare_features(df, all_numeric_features, categorical_features)
    
    # Set X and y
    y = df[target_column]
    X = df[numeric_features + categorical_features]
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Verify columns
    print(f"Numeric columns being processed: {numeric_features[:5]}... (total: {len(numeric_features)})")
    print(f"Categorical columns being processed: {categorical_features[:5]}... (total: {len(categorical_features)})")
    
    # Create preprocessing pipeline
    preprocessor = utils.create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Debug pipeline transformation
    debug_pipeline = make_pipeline(preprocessor)
    transformed = debug_pipeline.fit_transform(X)
    print("Transformed shape:", transformed.shape)
    
    # Get all OneHot encoded fields
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    
    print("🚀 Transformed Categorical fields count: ", len(encoded_feature_names))
    print("🚀 Sample of column names after OneHot encoding:")
    print(encoded_feature_names[:10])  # Show first 10 to avoid cluttering
    
    final_feature_names = numeric_features + encoded_feature_names.tolist()
    print("🚀 Transformed fields in total: ", len(final_feature_names))
    
    # Create full pipeline
    pipeline = utils.create_model_pipeline(preprocessor, LinearRegression)
    
    # Split data
    X_train, X_test, y_train, y_test = utils.split_data(X, y)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    results = utils.evaluate_model(pipeline, X_train, X_test, y_train, y_test, is_log_scale=True)
    
    # Save model
    utils.save_model(pipeline, "pkl/house_price_pipeline.pkl")
    
    # Visualize results
    # Log scale prediction vs actual
    utils.plot_predictions_vs_actual(
        results['actual']['log']['test'],
        results['predictions']['log']['test'],
        "log",
        "prediction_vs_actual_log.png"
    )
    
    # Original scale prediction vs actual
    utils.plot_predictions_vs_actual(
        results['actual']['original']['test'],
        results['predictions']['original']['test'],
        "original",
        "prediction_vs_actual_original.png"
    )
    
    # Calculate residuals
    residuals = results['actual']['log']['test'] - results['predictions']['log']['test']
    utils.plot_residuals(results['predictions']['log']['test'], residuals)
    
    # Feature importance analysis
    sorted_coef, intercept = utils.analyze_feature_importance(pipeline, numeric_features, categorical_features)
    
    print("\n✅ House Price Prediction Pipeline Completed Successfully")
    return pipeline, df, results

# If running as a script, execute the main function
if __name__ == "__main__":
    run_house_price_prediction_pipeline()