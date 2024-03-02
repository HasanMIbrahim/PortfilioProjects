import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

#Input the ticker of the crypto you want
crypto_currency = input("Enter the cryptocurrency ticker (e.g., BTC, ETH): ").upper()
against_currency = 'USD'

start= dt.datetime(2016,1,1)
end = dt.datetime.now()

data = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end)


#prepare Data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60
future_day = 30
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)- future_day):
 x_train.append(scaled_data[x-prediction_days:x, 0])
 y_train.append(scaled_data[x + future_day ,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


 #Creating neural network, the model for the prediction

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
# training the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


#testing the model

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = yf.download(f'{crypto_currency}-{against_currency}',test_start,test_end)
actual_prices = test_data['Close'].values
total_dataset=pd.concat((data['Close'], test_data['Close']), axis=0)

# Use the index as the datetime for total_dataset
total_dataset.index = pd.to_datetime(total_dataset.index)


model_inputs = total_dataset[len(total_dataset) - len(test_data)- prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

#make predictions using test model
x_test =[]
for x in range(prediction_days, len(model_inputs)):
 x_test.append(model_inputs[x-prediction_days:x, 0])


x_test= np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

# Convert the index of actual_prices and prediction_prices to datetime objects
actual_dates = total_dataset.index[-len(test_data):]
predicted_dates = total_dataset.index[-len(test_data) + prediction_days:]

prediction_prices = prediction_prices[:len(predicted_dates)]
plt.plot(actual_dates, actual_prices, color='black', label='Actual prices')
plt.plot(predicted_dates, prediction_prices, color='green', label='Predicted Prices')

# Format the x-axis as dates
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

# Predict next 30 days
real_data = [model_inputs[-prediction_days:, 0]]

for day in range(future_day):
    # Predict the next day
    prediction = model.predict(np.array(real_data[-prediction_days:]).reshape(1, prediction_days, 1))
    real_data = np.append(real_data, prediction[0, 0])

# Inverse transform the predicted values
predicted_prices_30_days = scaler.inverse_transform(np.array(real_data[prediction_days:]).reshape(-1, 1))

# Generate dates for the next 30 days
next_30_days_dates = pd.date_range(test_data.index[-1], periods=future_day + 1)[1:]

plt.plot(next_30_days_dates, predicted_prices_30_days, color='blue', label='Predicted Prices (Next 30 Days)')





plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend (loc='upper left')
plt.show()



#Predict next day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print()
