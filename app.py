import pickle 
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model('model.h5')

with open('tokenizer.pkl','rb') as file:
    tokenizer= pickle.load(file)

st.title('Sentiment Analysis Of Twitter Tweets')
tweet= st.text_area('Enter the tweet :')

if st.button('predict sentiments') and tweet.strip():
    sequence = tokenizer.texts_to_sequences([tweet])
    sequence=pad_sequences(sequence,padding='post',maxlen=99)
    prediction = model.predict(sequence)
    predicted_class= np.argmax(prediction,axis=1)[0]
    sentiment_map = {0:'Negative',1:'Neutral',2:'Positive'}
    st.write('sentiment',sentiment_map[predicted_class])
