#!/usr/bin/env python
# coding: utf-8

# 🏠 House Price Prediction Utilities
# Contains various reusable functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# =================== Environment Configuration Functions ===================

def setup_environment(seed=42):
    """Set up environment variables, random seed, and visualization parameters"""
    # Configure visualization settings
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    # Set random seed for reproducibility
    np.random.seed(seed)
    # Create directories for saving outputs
    os.makedirs("images", exist_ok=True)
    os.makedirs("pkl", exist_ok=True)

# =================== Data Loading and Overview Functions ===================

def load_and_overview_data(file_path):
    """Load data and print basic overview information"""
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print("\nData sample:")
    print(df.head())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    print("\nColumns with missing values:")
    print(missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False))
    
    return df

# =================== Visualization Functions ===================

def plot_and_save(plot_func, filename, *args, **kwargs):
    """General plotting and saving function"""
    plt.figure(figsize=(10, 6))
    plot_func(*args, **kwargs)
    filepath = os.path.join("images", filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_distribution(data, column, log_transform=False, filename=None):
    """Plot distribution with optional log transformation"""
    if log_transform:
        data_to_plot = np.log1p(data[column])
        title = f'Distribution of Log-Transformed {column}'
        x_label = f'Log({column} + 1)'
        if filename is None:
            filename = f"log_{column.lower()}_distribution.png"
    else:
        data_to_plot = data[column]
        title = f'Distribution of {column}'
        x_label = f'{column}'
        if filename is None:
            filename = f"{column.lower()}_distribution.png"
    
    def _plot():
        sns.histplot(data_to_plot, kde=True)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('Frequency')
    
    return plot_and_save(_plot, filename)

def plot_correlation_heatmap(data, filename="correlation_heatmap.png"):
    """Plot correlation heatmap"""
    numerical_data = data.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numerical_data.corr()
    
    def _plot():
        plt.figure(figsize=(14, 12))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features')
    
    return plot_and_save(_plot, filename)

def plot_top_correlations(data, target_column, top_n=10):
    """Output features most correlated with the target variable"""
    numerical_data = data.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numerical_data.corr()
    
    target_corr = correlation_matrix[target_column].sort_values(ascending=False)
    target_corr = pd.DataFrame(target_corr)
    target_corr.columns = [f'Correlation with {target_column}']
    print(f"\nTop {top_n} Features Most Correlated with {target_column}:")
    print(target_corr.head(top_n))
    
    return target_corr

def plot_scatter_features_vs_target(data, features, target, filename="feature_scatter.png"):
    """Plot scatter plots of features versus target variable"""
    def _plot():
        for i, feature in enumerate(features):
            plt.subplot(2, 2, i+1)
            plt.scatter(data[feature], data[target], alpha=0.5)
            plt.title(f'{feature} vs {target}')
            plt.xlabel(feature)
            plt.ylabel(target)
        plt.tight_layout()
    
    return plot_and_save(_plot, filename)

def plot_categorical_features(data, features, target, filename="categorical_features.png"):
    """Plot boxplots of categorical features versus target variable"""
    def _plot():
        for i, feature in enumerate(features):
            plt.subplot(2, 2, i+1)
            sns.boxplot(x=feature, y=target, data=data)
            plt.title(f'{feature} vs {target}')
            plt.xticks(rotation=45)
        plt.tight_layout()
    
    return plot_and_save(_plot, filename)

def plot_predictions_vs_actual(y_actual, y_pred, scale_type="log", filename=None):
    """Plot comparison of predicted vs actual values"""
    if filename is None:
        filename = f"prediction_vs_actual_{scale_type}.png"
    
    def _plot():
        plt.scatter(y_actual, y_pred, alpha=0.5)
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], '--r')
        y_label = "Predicted Price"
        x_label = "Actual Price"
        if scale_type == "log":
            y_label += " (Log Scale)"
            x_label += " (Log Scale)"
        else:
            y_label += " ($)"
            x_label += " ($)"
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"Prediction vs Actual ({scale_type.capitalize()} Scale)")
        plt.tight_layout()
    
    return plot_and_save(_plot, filename)

def plot_residuals(y_pred, residuals, filename="residual_plot.png"):
    """Plot residuals"""
    def _plot():
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='r', linestyle='--')
        plt.xlabel('Predicted Log Price')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True)
    
    return plot_and_save(_plot, filename)

def plot_top_feature_importance(sorted_coefficients, intercept, top_n=15, filename="feature_importance.png"):
    """Plot feature importance chart"""
    top_features = sorted_coefficients.head(top_n).index
    top_coefficients = sorted_coefficients.loc[top_features, 'Coefficient'].values
    
    def _plot():
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_features, top_coefficients)
        plt.title(f'Top {top_n} Features by Coefficient Magnitude')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Color negative and positive coefficients differently
        for i, bar in enumerate(bars):
            if top_coefficients[i] < 0:
                bar.set_color('salmon')
            else:
                bar.set_color('skyblue')
    
    plot_and_save(_plot, filename)
    
    # Print linear regression equation (top 10 terms)
    top_10_features = sorted_coefficients.head(10).index
    top_10_coeffs = sorted_coefficients.loc[top_10_features, 'Coefficient'].values
    
    equation = f"log(SalePrice) = {intercept:.4f}"
    for feat, coef in zip(top_10_features, top_10_coeffs):
        sign = "+" if coef > 0 else ""
        equation += f" {sign} {coef:.4f} × {feat}"
    equation += " + ..."
    
    print("\nLinear Regression Equation (top 10 terms):")
    print(equation)

# =================== Feature Engineering Functions ===================

def remove_outliers(df, filters):
    """Remove outliers based on filter conditions"""
    original_rows = df.shape[0]
    for filter_condition in filters:
        df = df[eval(filter_condition, {"df": df})]
    print(f"Rows before removing outliers: {original_rows}")
    print(f"Rows after removing outliers: {df.shape[0]}")
    return df

def fill_missing_values(df, numeric_cols=None, categorical_cols=None):
    """Fill missing values"""
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                print(f"Filling {df[col].isnull().sum()} missing values in {col}")
                df[col] = df[col].fillna(df[col].median())
    
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')
    
    return df

def create_features(df):
    """Create derived features"""
    # 1. House age at time of sale
    if 'YearSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YearSold'] - df['YearBuilt']
    
    # 2. Total bathrooms
    if 'FullBath' in df.columns and 'HalfBath' in df.columns:
        df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']
    
    # 3. Log transform skewed numerical features
    skewed_features = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
    for feature in skewed_features:
        if feature in df.columns:
            df[feature + '_Log'] = np.log1p(df[feature])
    
    # 4. Total square footage
    if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
        df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
        df['TotalSF_Log'] = np.log1p(df['TotalSF'])
    
    # 5. Has garage binary feature
    if 'GarageArea' in df.columns:
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    
    return df

def transform_target(df, target_column, transform_type='log'):
    """Transform target variable"""
    if transform_type == 'log':
        df[f'Log{target_column}'] = np.log1p(df[target_column])
        return df, f'Log{target_column}'
    return df, target_column

def prepare_features(df, numeric_features, categorical_features):
    """Prepare features, ensuring they exist in the dataset"""
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    all_features = numeric_features + categorical_features
    
    # Handle missing values in selected features
    for col in all_features:
        if col in df.columns and df[col].dtype.name != 'object':
            # For numeric columns, fill NaN with median
            df[col] = df[col].fillna(df[col].median())
        elif col in df.columns:
            # For categorical columns, fill NaN with most frequent value
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'None')
    
    print(f"Features used: {len(all_features)}")
    print(f"Numeric fields count: {len(numeric_features)}")
    print(f"Categorical fields count: {len(categorical_features)}")
    
    return df, numeric_features, categorical_features

# =================== Model Building and Evaluation Functions ===================

def create_preprocessing_pipeline(numeric_cols, categorical_cols):
    """Create preprocessing pipeline"""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder='drop')
    
    return preprocessor

def create_model_pipeline(preprocessor, model_class, model_params=None):
    """Create complete model pipeline"""
    if model_params is None:
        model_params = {}
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_class(**model_params))
    ])
    
    return pipeline

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def evaluate_model(pipeline, X_train, X_test, y_train, y_test, is_log_scale=True):
    """Evaluate model performance"""
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Make predictions on training and test sets
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Evaluation metrics on log scale
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    print(f"\n📊 Evaluation Metrics (Log Scale):")
    print(f"Training Data:")
    print(f"MAE (log scale): {train_metrics['mae']:.4f}")
    print(f"RMSE (log scale): {train_metrics['rmse']:.4f}")
    print(f"R^2 (log scale): {train_metrics['r2']:.4f}")
    
    print(f"\nTest Data:")
    print(f"MAE (log scale): {test_metrics['mae']:.4f}")
    print(f"RMSE (log scale): {test_metrics['rmse']:.4f}")
    print(f"R^2 (log scale): {test_metrics['r2']:.4f}")
    
    # If log scale, transform back to original scale and calculate metrics
    if is_log_scale:
        y_train_actual = np.expm1(y_train)
        y_pred_train_actual = np.expm1(y_pred_train)
        y_test_actual = np.expm1(y_test)
        y_pred_test_actual = np.expm1(y_pred_test)
        
        actual_metrics = calculate_metrics(y_test_actual, y_pred_test_actual)
        
        print(f"\n📊 Evaluation Metrics (Original Scale):")
        print(f"MAE : ${actual_metrics['mae']:.2f}")
        print(f"RMSE: ${actual_metrics['rmse']:.2f}")
        print(f"R^2 : {actual_metrics['r2']:.4f}")
        
        return {
            'log_scale': {'train': train_metrics, 'test': test_metrics},
            'original_scale': actual_metrics,
            'predictions': {
                'log': {'train': y_pred_train, 'test': y_pred_test},
                'original': {'train': y_pred_train_actual, 'test': y_pred_test_actual}
            },
            'actual': {
                'log': {'train': y_train, 'test': y_test},
                'original': {'train': y_train_actual, 'test': y_test_actual}
            }
        }
    
    return {
        'metrics': {'train': train_metrics, 'test': test_metrics},
        'predictions': {'train': y_pred_train, 'test': y_pred_test},
        'actual': {'train': y_train, 'test': y_test}
    }

def save_model(pipeline, filename):
    """Save model to file"""
    joblib.dump(pipeline, filename)
    print(f"Model saved to '{filename}'")

def analyze_feature_importance(pipeline, df_numeric, df_categorical):
    """Analyze feature importance"""
    try:
        # Get the linear regression model
        linear_model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get all feature names after preprocessing
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(df_categorical).tolist()
        all_feature_names = df_numeric + cat_feature_names
        
        # Check if feature names length matches coefficients length
        if len(all_feature_names) != len(linear_model.coef_):
            print(f"Warning: Feature names length ({len(all_feature_names)}) doesn't match coefficients length ({len(linear_model.coef_)})")
            # Create generic feature names if mismatch
            all_feature_names = [f"Feature_{i}" for i in range(len(linear_model.coef_))]
        
        # Get coefficients
        coefficients = pd.DataFrame(linear_model.coef_, index=all_feature_names, columns=['Coefficient'])
        coefficients['Absolute Value'] = coefficients['Coefficient'].abs()
        
        # Sort by absolute value of coefficients
        sorted_coefficients = coefficients.sort_values('Absolute Value', ascending=False)
        
        # Display top 15 most important features
        print("\nTop 15 Most Important Features:")
        print(sorted_coefficients.head(15))
        
        # Intercept value
        print(f"\nIntercept: {linear_model.intercept_:.4f}")
        
        # Visualize top 15 most important features
        plot_top_feature_importance(sorted_coefficients, linear_model.intercept_)
        
        return sorted_coefficients, linear_model.intercept_
    
    except Exception as e:
        print(f"Error in feature importance analysis: {e}")
        print("Coefficient shape:", linear_model.coef_.shape)
        print("Showing raw coefficients instead:")
        
        # Plot raw coefficients
        def _plot():
            plt.bar(range(len(linear_model.coef_)), linear_model.coef_)
            plt.title('Raw Feature Coefficients')
            plt.xlabel('Feature Index')
            plt.ylabel('Coefficient Value')
            plt.tight_layout()
        
        plot_and_save(_plot, "raw_coefficients.png")
        return None, None