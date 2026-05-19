import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Download stock data
df = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

# 2. Show data
print(df.head())

# 3. Visualize closing price
plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title("Apple Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# 4. Prepare data for prediction
df = df[['Close']]
df['Prediction'] = df['Close'].shift(-5)

# Remove null values
df = df.dropna()

# 5. Features and labels
X = df[['Close']]
y = df['Prediction']

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 7. Model training
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Prediction
predictions = model.predict(X_test)

# 9. Show sample predictions
print("Sample Predictions:")
print(predictions[:5])