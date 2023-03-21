import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

### === Dev: SammyFANG === ###

# Alpha Vantage API key
API_KEY = 'Apply your API' # USEING your API

# Define the URL for the Alpha Vantage API call
symbol = 'TSLA' # Change what you want
api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={API_KEY}&outputsize=compact&datatype=csv'

# Read data from Alpha Vantage API into a pandas DataFrame
df = pd.read_csv(api_url)

# Reverse the DataFrame so that the oldest data is first
df = df.iloc[::-1]
print(df)

# Use only the 'adjusted close' column and convert to numpy array
data = df['adjusted_close'].to_numpy()

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# Define the number of timesteps to use in each training sample
n_steps = 7

# Split the data into training and testing sets
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size-n_steps:]

# Generate training sequences
def generate_sequences(data, n_steps):
    X = []
    y = []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, 0])
        y.append(data[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

X_train, y_train = generate_sequences(train_data, n_steps)
X_test, y_test = generate_sequences(test_data, n_steps)

# Define the RNN model architecture
model = Sequential()
model.add(SimpleRNN(units=64, activation='relu', input_shape=(n_steps, 1)))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions on test data
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values to the original scale
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

print(y_pred)
# Plot the predicted and actual values
plt.title(">>> 61171025H_Prediction <<<")
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
# NTNU Sammy v0.0.1 2023.03