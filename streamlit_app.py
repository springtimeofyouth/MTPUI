import streamlit as st
import joblib
import numpy as np
# Load the pickled model

model1 = joblib.load('model_PB.joblib')

def predict(features):
    features = np.array(features).reshape(1, -1)
    prediction1 = model1.predict(features)
    return prediction1[0]

# Streamlit app
st.title('User Interface  for Prediction of Pb')


feature_1 = st.number_input('N2', value=0.0, format="%.3f", step=0.001)
feature_2 = st.number_input('CO2', value=0.0, format="%.3f", step=0.001)
feature_3 = st.number_input('H2S', value=0.0, format="%.3f", step=0.001)
feature_4 = st.number_input('C1', value=0.0, format="%.3f", step=0.001)
feature_5 = st.number_input('C2', value=0.0, format="%.3f", step=0.001)
feature_6 = st.number_input('C3', value=0.0, format="%.3f", step=0.001)
feature_7 = st.number_input('C4', value=0.0, format="%.3f", step=0.001)
feature_8 = st.number_input('C5', value=1.0, format="%.3f", step=0.001)
feature_9 = st.number_input('C6', value=1.0, format="%.3f", step=0.001)
feature_10 = st.number_input('C7+', value=1.0, format="%.4f", step=0.0001)
feature_11= st.number_input('SGC7+', value=0.0,format="%.4f", step=0.0001)
feature_12= st.number_input('MWC7+', value=0.0,format="%.3f", step=0.001)
feature_13= st.number_input('Temp (F)', value=50.0)

if st.button('Predict Pb'):
    features_bp = [feature_1, feature_2, feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_11,
                feature_12,feature_13]  # Adjust based on your feature set
    prediction_bp = predict(features_bp)
    st.write(f'The Bubble Point Pressure is : {prediction_bp} psia')
    
