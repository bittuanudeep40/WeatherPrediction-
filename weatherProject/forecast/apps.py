# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import requests
from datetime import datetime, timedelta
import pytz

# --- Machine Learning Functions (from your views.py) ---

def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna().drop_duplicates()
    return df

def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    return X, y, le

def train_rain_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    return X, y

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))[0]
        predictions.append(next_value)
    return predictions[1:]

# --- Streamlit App ---

st.title('Weather Forecast App 🌦️')

# Get user input for the city
city = st.text_input('Enter a city name:', 'London')

if st.button('Get Forecast'):
    if city:
        try:
            # --- Get Current Weather (from your views.py) ---
            API_KEY = '292b993e9753c48bdb5efc74d4ac54e5'  # It's better to use st.secrets for this
            BASE_URL = "https://api.openweathermap.org/data/2.5/"
            url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
            response = requests.get(url)
            data = response.json()

            current_weather = {
                'city': data['name'],
                'current_temp': round(data['main']['temp']),
                'feels_like': round(data['main']['feels_like']),
                'temp_min': round(data['main']['temp_min']),
                'temp_max': round(data['main']['temp_max']),
                'humidity': round(data['main']['humidity']),
                'description': data['weather'][0]['description'],
                'country': data['sys']['country'],
                'wind_deg': data['wind']['deg'],
                'pressure': data['main']['pressure'],
                'WindGustSpeed': data['wind']['speed'],
                'clouds': data['clouds']['all'],
                'Visibility': data['visibility'],
            }

            # --- Display Current Weather ---
            st.header(f"Current Weather in {current_weather['city']}, {current_weather['country']}")
            st.write(f"**Temperature:** {current_weather['current_temp']}°C")
            st.write(f"**Feels Like:** {current_weather['feels_like']}°C")
            st.write(f"**Description:** {current_weather['description'].title()}")
            st.write(f"**Humidity:** {current_weather['humidity']}%")
            st.write(f"**Wind Speed:** {current_weather['WindGustSpeed']} km/h")

            # --- Machine Learning Predictions ---
            historical_data = read_historical_data('weather.csv')
            X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
            X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
            temp_model = train_regression_model(X_temp, y_temp)
            hum_model = train_regression_model(X_hum, y_hum)
            future_temp = predict_future(temp_model, current_weather['temp_min'])
            future_humidity = predict_future(hum_model, current_weather['humidity'])

            # --- Display Future Forecast ---
            st.header('5-Hour Forecast')
            timezone = pytz.timezone('Asia/Karachi')
            now = datetime.now(timezone)
            next_hour = now + timedelta(hours=1)
            next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
            future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

            forecast_data = {
                'Time': future_times,
                'Temperature (°C)': [round(t, 1) for t in future_temp],
                'Humidity (%)': [round(h, 1) for h in future_humidity]
            }
            forecast_df = pd.DataFrame(forecast_data)
            st.table(forecast_df)

        except Exception as e:
            st.error(f"Could not retrieve weather data for '{city}'. Please check the city name and try again.")
            st.error(f"Error details: {e}")
    else:
        st.warning('Please enter a city name.')
