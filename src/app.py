import streamlit as st 
import pandas as pd 
import numpy as np
from src.piplines.predict_pipeline import PredictPipeline

import pickle  

st.title('Clothe Cost Prediction')
 
 
ID = st.number_input('ID', value=0)


st.write('Brand:')
selected_brand = st.radio("Select Brand", ['Bey&Bey', 'HA', 'Zen', 'Armani', 'Kontakt'])
st.write("Selected Brand:", selected_brand)

st.write('Style:')
selected_Style = st.radio("Select Style", ['mta3 a3res' ,'confy' ,'classy' ,'sport' ,'mta3 kolyoum' ,'posé' ,'formel'])
st.write("Selected Style:", selected_Style)

st.write("Type:")
selected_type = st.radio("Select Type", ['kabbout', 'sabbat', 'maryoul', 't-shirt', 'serwel', 'jacket', 'hoodie', 'socks'])

# Display the selected type
st.write("Selected Type:", selected_type)
  

Thickness = st.number_input('Enter Thickness', value=0.0)
Length = st.number_input('Enter Length', value=0) 
Width = st.number_input('Enter width', value=0)
Color_code_R = st.number_input('Enter color code Red', value=0)
Color_code_G = st.number_input('Enter color code Green', value=0)
Color_code_B = st.number_input('Enter color code Blue', value=0) 


def predict():
    with open(r'C:\Users\hp\Desktop\HousepricePred\artifact\model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open(r'C:\Users\hp\Desktop\HousepricePred\artifact\preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)

    row=np.array([ID,selected_brand,selected_Style,selected_type,Thickness,Length,Width,Color_code_R,Color_code_G,Color_code_B])
    
    X=pd.DataFrame([row],columns=['ID','marka','naw3','9at3a','khochn','toul','3ordh','R','G','B'])
    X=preprocessor.transform(X)
    prediction=model.predict(X)

    st.write(f'Price is : ',prediction)

    

st.button('Predict Product Cost ',on_click=predict)


 






 

