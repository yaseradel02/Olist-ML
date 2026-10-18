# Olist Delivery Time Predictor

## Project Overview
A machine learning application that predicts delivery times using historical Brazilian e-commerce data from Olist (2016–2018).

## Objective
Help customers estimate their order's delivery time and expected arrival date using machine learning.

## Features
- Predict delivery time using HistGradientBoostingRegressor.
- Calculate distance between customer and seller using ZIP codes.
- Estimate the delivery date.
- Alert customers when the estimated delivery date has passed.
- Interactive Streamlit web app.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Streamlit, Joblib.

## App link

https://olist-delivery-predictor.streamlit.app/

## Dataset
Brazilian E-Commerce Public Dataset by Olist (Kaggle).

## Limitations
- Uses historical data from 2016–2018.
- Distance is calculated as a straight line, not road distance.
- Does not consider traffic, weather, or unexpected delays.
- Only supports ZIP codes and sellers available in the dataset.
- Predictions are estimates, not guaranteed delivery dates.
