## 🏠 House Price Prediction Pipeline (Optimized)
# Author: Randy Jin
# Day 4: Optimized Linear Regression

# ✅ 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

# ✅ 2. Load Data
df = pd.read_csv("dataset.csv")

# ✅ 3. Remove Outliers (optional, based on EDA)
df = df[df['SalePrice'] < 500000]  # simple upper cap to reduce outlier impact

# ✅ 4. Log Transform Target Variable
df['LogSalePrice'] = np.log1p(df['SalePrice'])

# ✅ 5. Feature & Target Split
y = df['LogSalePrice']
X = df.drop(['SalePrice', 'LogSalePrice'], axis=1)

# ✅ 6. Identify Column Types
df_numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
df_categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()

# ✅ 7. Preprocessing Pipelines
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
])

# ✅ 8. Build Full Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# ✅ 9. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 10. Train Model
pipeline.fit(X_train, y_train)

# ✅ 11. Predict & Evaluate on log scale
y_pred_log = pipeline.predict(X_test)

# ✅ 12. Inverse Transform Back to Original Scale
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_log)

mae = mean_absolute_error(y_test_actual, y_pred_actual)
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"\n📊 Evaluation Metrics (Converted to Original SalePrice):")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 : {r2:.4f}")

# ✅ 13. Save Model
joblib.dump(pipeline, "optimized_house_price_pipeline.pkl")

# ✅ 14. Plot Prediction vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], '--r')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Prediction vs Actual (Linear Regression, Optimized)")
plt.tight_layout()
plt.savefig("optimized_prediction_vs_actual.png")
plt.show()