#!/usr/bin/env python
# coding: utf-8

# 🏠 House Price Prediction Pipeline
# Author: Randy Jin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

import joblib

# Configure visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Set random seed for reproducibility
np.random.seed(42)

# Create a directory for saving visualizations
os.makedirs("images", exist_ok=True)

# Load Data
df = pd.read_csv("dataset.csv")
print(f"Dataset shape: {df.shape}")

# Data Overview
print("Dataset dimensions:", df.shape)
print("\nData sample:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percent
})

# Display columns with missing values
print("\nColumns with missing values:")
print(missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False))

# Exploratory Data Analysis (EDA)
# Distribution of the target variable (SalePrice)
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price ($)')
plt.ylabel('Frequency')
filename = os.path.join("images", "sale_price_distribution.png")
plt.savefig(filename)
plt.close()

# Check log transformation of SalePrice
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(df['SalePrice']), kde=True)
plt.title('Distribution of Log-Transformed Sale Price')
plt.xlabel('Log(Sale Price + 1)')
plt.ylabel('Frequency')
filename = os.path.join("images", "log_sale_price_distribution.png")
plt.savefig(filename)
plt.close()

# Correlation analysis for numerical features
numerical_data = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
filename = os.path.join("images", "correlation_heatmap.png")
plt.savefig(filename)
plt.close()

# Top correlations with SalePrice
sale_price_corr = correlation_matrix['SalePrice'].sort_values(ascending=False)
sale_price_corr = pd.DataFrame(sale_price_corr)
sale_price_corr.columns = ['Correlation with SalePrice']
print("\nTop 10 Features Most Correlated with SalePrice:")
print(sale_price_corr.head(10))

# Scatter plots of top numerical features vs SalePrice
top_features = sale_price_corr.index[:5][1:]  # Exclude SalePrice itself

plt.figure(figsize=(18, 12))
for i, feature in enumerate(top_features):
    plt.subplot(2, 2, i+1)
    plt.scatter(df[feature], df['SalePrice'], alpha=0.5)
    plt.title(f'{feature} vs SalePrice')
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
plt.tight_layout()
filename = os.path.join("images", "top_features_scatter.png")
plt.savefig(filename)
plt.close()

# Analyze categorical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical features: {categorical_features}")

# Plot boxplots for top categorical features
plt.figure(figsize=(18, 15))
for i, feature in enumerate(categorical_features[:4]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x=feature, y='SalePrice', data=df)
    plt.title(f'{feature} vs SalePrice')
    plt.xticks(rotation=45)
plt.tight_layout()
filename = os.path.join("images", "categorical_features_boxplot.png")
plt.savefig(filename)
plt.close()

# Feature Engineering
print(f"Rows before removing outliers: {df.shape[0]}")
df = df[(df['GrLivArea'] < 4000) | (df['SalePrice'] > 300000)]  # Remove houses with large area but low price
df = df[df['SalePrice'] < 500000]  # Remove very expensive houses
print(f"Rows after removing outliers: {df.shape[0]}")

# Handle missing values for critical fields
# Fill missing values for important numeric fields
for col in ['LotArea', 'GrLivArea', 'TotalBsmtSF']:
    if df[col].isnull().sum() > 0:
        print(f"Filling {df[col].isnull().sum()} missing values in {col}")
        df[col] = df[col].fillna(df[col].median())

# Feature creation
# 1. House age at time of sale
df['HouseAge'] = df['YearSold'] - df['YearBuilt']

# 2. Total bathrooms
df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']

# 3. Log transform skewed numerical features
skewed_features = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
for feature in skewed_features:
    df[feature + '_Log'] = np.log1p(df[feature])

# 4. Total square footage
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
df['TotalSF_Log'] = np.log1p(df['TotalSF'])

# 5. Has garage binary feature
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)

# Handling specific categorical missing values
for col in ['Alley', 'GarageType', 'MasVnrType', 'Fence', 'FireplaceQu', 'MiscFeature']:
    if col in df.columns:
        df[col] = df[col].fillna('None')

# Log transform the target variable
df['LogSalePrice'] = np.log1p(df['SalePrice'])

# Define features to use (original plus engineered)
features = [
    # Original numerical features
    'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'FullBath', 'HalfBath', 
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
    'LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'WoodDeckSF', 
    'OpenPorchSF', 'MasVnrArea', 'BsmtFinSF1', 'YearSold',
    
    # Log-transformed features
    'LotArea_Log', 'GrLivArea_Log', 'TotalBsmtSF_Log', 'TotalSF_Log',
    
    # Engineered features
    'HouseAge', 'TotalBath', 'HasGarage', 'TotalSF'
]

# Filter to keep only columns that exist in the dataframe
features = [f for f in features if f in df.columns]

# Add all categorical features that exist in the dataframe
categorical_cols = [col for col in categorical_features if col in df.columns]
features = features + categorical_cols

# Handle missing values in the input features before splitting
for col in features:
    if col in df.columns and df[col].dtype.name != 'object':
        # For numeric columns, fill NaN with median
        df[col] = df[col].fillna(df[col].median())
    elif col in df.columns:
        # For categorical columns, fill NaN with most frequent value
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'None')

y = df['LogSalePrice']  # Using log-transformed target
X = df[features]  # Use our selected features

print(f"Features used: {len(features)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Separate Features
df_numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
print("Numeric fields count: ", len(df_numeric))
df_categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical fields count: ", len(df_categorical))

# Print columns to verify
print(f"Numeric columns being processed: {df_numeric[:5]}... (total: {len(df_numeric)})")
print(f"Categorical columns being processed: {df_categorical[:5]}... (total: {len(df_categorical)})")

# Ensure all columns exist in the dataframe
df_numeric = [col for col in df_numeric if col in X.columns]
df_categorical = [col for col in df_categorical if col in X.columns]

# Preprocessing Pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, df_numeric),
    ("cat", categorical_pipeline, df_categorical)
], remainder='drop')  # Explicitly drop any columns not specified

# Debug Pipeline Transformation
debug_pipeline = make_pipeline(preprocessor)
transformed = debug_pipeline.fit_transform(X)
print("Transformed shape:", transformed.shape)

# Get all of the OneHot encoded fields
encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
encoded_feature_names = encoder.get_feature_names_out(df_categorical)

print("🚀 Transformed Categorical fields count: ", len(encoded_feature_names))

print("🚀 Sample of column names after OneHot encoding:")
print(encoded_feature_names[:10])  # Show first 10 to avoid cluttering

final_feature_names = df_numeric + encoded_feature_names.tolist()
print("🚀 Transformed fields in total: ", len(final_feature_names))

# Build Full Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Train Model
pipeline.fit(X_train, y_train)

# Apply cross-validation to assess model stability
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Make predictions on train and test sets
y_pred_train = pipeline.predict(X_train)
y_pred_log = pipeline.predict(X_test)

# Evaluate on log scale - Training data
train_mae_log = mean_absolute_error(y_train, y_pred_train)
train_mse_log = mean_squared_error(y_train, y_pred_train)
train_rmse_log = np.sqrt(train_mse_log)
train_r2_log = r2_score(y_train, y_pred_train)

# Evaluate on log scale - Test data
mae_log = mean_absolute_error(y_test, y_pred_log)
mse_log = mean_squared_error(y_test, y_pred_log)
rmse_log = np.sqrt(mse_log)
r2_log = r2_score(y_test, y_pred_log)

print(f"\n📊 Evaluation Metrics (Log Scale):")
print(f"Training Data:")
print(f"MAE (log scale): {train_mae_log:.4f}")
print(f"RMSE (log scale): {train_rmse_log:.4f}")
print(f"R^2 (log scale): {train_r2_log:.4f}")

print(f"\nTest Data:")
print(f"MAE (log scale): {mae_log:.4f}")
print(f"RMSE (log scale): {rmse_log:.4f}")
print(f"R^2 (log scale): {r2_log:.4f}")

# Inverse Transform Back to Original Scale
y_train_actual = np.expm1(y_train)
y_pred_train_actual = np.expm1(y_pred_train)
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_log)

# Evaluate on original scale - Test data
mae_actual = mean_absolute_error(y_test_actual, y_pred_actual)
mse_actual = mean_squared_error(y_test_actual, y_pred_actual)
rmse_actual = np.sqrt(mse_actual)
r2_actual = r2_score(y_test_actual, y_pred_actual)

print(f"\n📊 Evaluation Metrics (Original SalePrice Scale):")
print(f"MAE : ${mae_actual:.2f}")
print(f"RMSE: ${rmse_actual:.2f}")
print(f"R^2 : {r2_actual:.4f}")

# Save Model
joblib.dump(pipeline, f"pkl/house_price_pipeline.pkl")
print("Model saved to 'pkl/house_price_pipeline.pkl'")

# Visualize Results
# Prediction vs Actual (Log Scale)
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_log, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual Log Price")
plt.ylabel("Predicted Log Price")
plt.title("Prediction vs Actual (Log Scale)")
plt.tight_layout()
filename = os.path.join("images", "prediction_vs_actual_log.png")
plt.savefig(filename)
plt.close()

# Prediction vs Actual (Original Scale)
plt.figure(figsize=(10, 8))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], '--r')
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Prediction vs Actual (Original Scale)")
plt.tight_layout()
filename = os.path.join("images", "prediction_vs_actual_original.png")
plt.savefig(filename)
plt.close()

# Calculate residuals
residuals = y_test - y_pred_log

plt.figure(figsize=(10, 8))
plt.scatter(y_pred_log, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred_log.min(), xmax=y_pred_log.max(), color='r', linestyle='--')
plt.xlabel('Predicted Log Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
filename = os.path.join("images", "residual_plot.png")
plt.savefig(filename)
plt.close()

# Feature Importance Analysis
# Get the linear regression model
linear_model = pipeline.named_steps['model']

try:
    # Get all feature names after preprocessing
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(df_categorical).tolist()
    all_feature_names = df_numeric + cat_feature_names
    
    # Check if the length of feature names matches the coefficients
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
    
    # Visualize top 15 features by importance
    top_features = sorted_coefficients.head(15).index
    top_coefficients = sorted_coefficients.loc[top_features, 'Coefficient'].values
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_features, top_coefficients)
    plt.title('Top 15 Features by Coefficient Magnitude')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Color negative and positive coefficients differently
    for i, bar in enumerate(bars):
        if top_coefficients[i] < 0:
            bar.set_color('salmon')
        else:
            bar.set_color('skyblue')
    
    plt.tight_layout()
    filename = os.path.join("images", "feature_importance.png")
    plt.savefig(filename)
    plt.close()
    
    # Print linear regression equation with the most significant terms
    intercept = linear_model.intercept_
    top_10_features = sorted_coefficients.head(10).index
    top_10_coeffs = sorted_coefficients.loc[top_10_features, 'Coefficient'].values
    
    equation = f"log(SalePrice) = {intercept:.4f}"
    for feat, coef in zip(top_10_features, top_10_coeffs):
        sign = "+" if coef > 0 else ""
        equation += f" {sign} {coef:.4f} × {feat}"
    equation += " + ..."
    
    print("\nLinear Regression Equation (top 10 terms):")
    print(equation)

except Exception as e:
    print(f"Error in feature importance analysis: {e}")
    print("Coefficient shape:", linear_model.coef_.shape)
    print("Showing raw coefficients instead:")
    # Plot the raw coefficients
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(linear_model.coef_)), linear_model.coef_)
    plt.title('Raw Feature Coefficients')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.tight_layout()
    filename = os.path.join("images", "raw_coefficients.png")
    plt.savefig(filename)
    plt.close()

print("\n✅ House Price Prediction Pipeline Completed Successfully")