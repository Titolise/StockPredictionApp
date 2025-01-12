import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import datetime

#----- Title -----
st.title('Stock :orange[Trend Prediction] App')
st.divider()

#----- Sidebar -----
#Sidebar - Ticker Input
col1 = st.sidebar.header('Enter :red[Stock Ticker]: ')
user_input = st.sidebar.text_input('(e.g., AAPL)','AAPL').upper()

#Sidebar - Star day Input
st.sidebar.header('Enter :red[Start Date]: ')
start = st.sidebar.date_input('Start Date', datetime.date(2010, 1, 1))
end = datetime.date.today().strftime("%Y-%m-%d") #Set today as the final day

df = yf.download(user_input, start, end)

st.sidebar.write('Data Source: :violet[ Yahoo Finance]')
st.sidebar.write('Data range: ', start, ' to ', end)

#Describing Data
st.subheader(f'Data of :orange[{user_input}]')
st.write(df.describe().transpose())
st.divider()

#----- Visualizations -----
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)
st.divider()

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)
st.divider()

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df['Close'], 'b')
st.pyplot(fig)
st.divider()

# Splitting Data
data_training = pd.DataFrame(df['Close'][:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_scaled = scaler.fit_transform(data_training)

#Splitting Data in X/Y
x_train = []
y_train = []
for i in range(100, data_training_scaled.shape[0]):
    x_train.append(data_training_scaled[i-100:i, 0])
    y_train.append(data_training_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)


#Load model
model = tf.keras.models.load_model('my_model.h5')

#Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i, 0])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
scaler = scaler.scale_

scaler_factor = 1/scaler[0]
y_pred = y_pred*scaler_factor
y_test = y_test*scaler_factor


#Final Graph
st.subheader('Actual vs Predicted Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label = 'Actual Price')
plt.plot(y_pred,'r', label = 'Predicted Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend()
st.pyplot(fig2)
