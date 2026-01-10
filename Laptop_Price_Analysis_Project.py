# Laptop Price Analysis & Prediction Project

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load Dataset


df = pd.read_csv("laptop_prices.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# 3. Basic Data Information


print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())


# 4. Check Missing Values


print("\nMissing Values:\n")
print(df.isnull().sum())



# 5. Exploratory Data Analysis (EDA)


# Company count
plt.figure(figsize=(10,5))
df['Company'].value_counts().plot(kind='bar')
plt.title("Laptop Count by Company")
plt.xlabel("Company")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# OS vs Price
plt.figure(figsize=(10,6))
sns.boxplot(x='OS', y='Price_euros', data=df)
plt.title("Operating System vs Price")
plt.xticks(rotation=45)
plt.show()

# Touchscreen vs Price
plt.figure(figsize=(6,6))
sns.barplot(x='Touchscreen', y='Price_euros', data=df)
plt.title("Touchscreen vs Price")
plt.show()

# RAM vs Price
plt.figure(figsize=(8,6))
sns.barplot(x='Ram', y='Price_euros', data=df)
plt.title("RAM vs Price")
plt.show()


# 6. Feature Selection

# Drop non-useful columns
df_model = df.drop(columns=['Product'])

# Convert categorical columns to numerical
df_model = pd.get_dummies(df_model, drop_first=True)

print("\nData After Encoding:")
print(df_model.head())


# 7. Split Features & Target


X = df_model.drop('Price_euros', axis=1)
y = df_model['Price_euros']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 8. Train Machine Learning Model


model = LinearRegression()
model.fit(X_train, y_train)

# 9. Model Prediction


y_pred = model.predict(X_test)

# --------------------------------------------
# 10. Model Evaluation
# --------------------------------------------

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-------------------")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.2f}")

# 11. Actual vs Predicted Visualization


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Laptop Prices")
plt.show()

print("\nProject Executed Successfully")
