import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("/Users/gordonli/Desktop/Waseda/spring semester/Intermediate Data Science /enhanced_anxiety_dataset.csv")

# Separate features and target
X = df.drop(columns=["Anxiety Level (1-10)"])
y = df["Anxiety Level (1-10)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# Get intercept and coefficients
intercept = model.intercept_
coefficients = model.coef_

# Match them to feature names
feature_names = X.columns

# Print the equation
equation = f"Anxiety_Level = {intercept:.3f}"
for name, coef in zip(feature_names, coefficients):
    equation += f" + ({coef:.3f} * {name})"

print(equation)

# Visualization (first for sleep hours)

feature = "Sleep Hours"
feature_idx = list(X.columns).index(feature)
X_mean = X_train_scaled.mean(axis=0)

# Generate a range for the chosen feature
x_vals = np.linspace(X_train_scaled[:, feature_idx].min(), X_train_scaled[:, feature_idx].max(), 100)
X_temp = np.tile(X_mean, (100, 1))
X_temp[:, feature_idx] = x_vals
y_vals = model.predict(X_temp)

plt.plot(x_vals, y_vals)
plt.xlabel(feature + " (scaled)")
plt.ylabel("Predicted Anxiety Level")
plt.title(f"Effect of {feature} on Anxiety Level (linear regression)")
plt.show()


# Bar Graph of Coefficients 
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient')

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title("Linear Regression Coefficients")
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()