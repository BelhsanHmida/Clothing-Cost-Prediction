import streamlit as st 
import pandas as pd 
import numpy as np

st.title('House Price Prediction')
 
@st.cache_data
def load_data(nrows):
    data = pd.read_csv(r'C:\Users\hp\Desktop\HousepricePred\artifact\raw.csv',nrows=nrows)
    return data
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
 

# Load 10 rows of data into the dataframe.
data = load_data(10)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done!  ")
 

st.subheader('Raw data')
st.write(data)

st.subheader('Number of pickups by hour')
FEATURE_COLUMN='LotArea'

hist_values = np.histogram(data[FEATURE_COLUMN], bins=30, range=(0, data[FEATURE_COLUMN].max()))[0]
st.bar_chart(hist_values)

Romms = st.slider('rooms', 0, 2, 17)

