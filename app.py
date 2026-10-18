import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2
from datetime import date, timedelta
import math

st.set_page_config(page_title='Olist Delivery Predictor')
st.title('Olist Delivery Predictor')


@st.cache_resource
def load_model():
    return joblib.load('olist_delivery_model.joblib')

@st.cache_data
def load_lookups():
    sellers = pd.read_csv('olist_sellers_dataset.csv', dtype={'seller_id': str})
    geo = pd.read_csv('zip_locations.csv')
    return sellers.set_index('seller_id'), geo.set_index('geolocation_zip_code_prefix')

try:
    model = load_model()
    sellers, geo = load_lookups()
except Exception as e:
    st.error(f'Missing model or lookup data: {e}')
    st.stop()

# Straight-line distance between two latitude/longitude coordinates.
def distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2)**2
    return 6371 * 2 * atan2(sqrt(a), sqrt(max(0, 1 - a)))

with st.form('order_form'):
    customer_zip = st.text_input('Customer ZIP code (5 digits)', '28013')
    seller_id = st.text_input('Seller ID', '3442f8959a84dea7ee197c632cb2df15')
    approval_date = st.date_input('Order approval date', value=date.today())
    freight = st.number_input('Total freight cost (BRL)', min_value=0.0, value=10.00, step=1.0)
    count = st.number_input('Number of items', min_value=1, value=1, step=1)
    submit = st.form_submit_button('Predict delivery time', type='primary')

if submit:
    zip_text = customer_zip.strip()
    if not (zip_text.isdigit() and len(zip_text) == 5):
        st.error('Enter a 5-digit Brazilian ZIP prefix, for example 28013.')
    elif seller_id.strip() not in sellers.index:
        st.error('Seller ID not found in the sellers dataset.')
    else:
        customer_code = int(zip_text)
        seller_code = int(sellers.loc[seller_id.strip(), 'seller_zip_code_prefix'])
        if customer_code not in geo.index:
            st.error('Customer ZIP code not found in geolocation data.')
        elif seller_code not in geo.index:
            st.error('Seller ZIP code not found in geolocation data.')
        else:
            c_lat, c_lon = geo.loc[customer_code, ['geolocation_lat', 'geolocation_lng']]
            s_lat, s_lon = geo.loc[seller_code, ['geolocation_lat', 'geolocation_lng']]
            distance = distance_km(c_lat, c_lon, s_lat, s_lon)
            order = pd.DataFrame([{
                'order_total_fv': freight,
                'N_items': count,
                'distance': distance,
                'month': approval_date.month,
                'day': approval_date.day,
                'weekday': approval_date.weekday(),
                'is_weekend': int(approval_date.weekday() >= 5),
                'customer_zip_region': zip_text[:2],
                'seller_zip_region': str(seller_code).zfill(5)[:2]
            }])
            days = max(0.0, float(model.predict(order)[0]))
            arrival_date = approval_date + timedelta(days=round(days))
            st.metric('Estimated delivery time', f'{math.ceil(days)} days')
            st.write('Estimated arrival date:', ((arrival_date)).strftime('%d %B %Y'))

            if date.today() > arrival_date:
                st.error(
                    "⚠️ Your estimated delivery date has passed! "
                    "Please contact the carrier company for an update."
                )
            
