import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image
import pickle as pk
import random
from sklearn.linear_model import LinearRegression

# classifier
with open('E:/updatedModel.pkl', 'rb') as f:
    classifier = pk.load(f)
    
# vectorizer
with open('E:/vectorizer.pkl', 'rb') as f:
    vectorizer = pk.load(f)
 
# function
def predict_review(review):
    print("Function call")
    vec=vectorizer.transform([review])
#     print("Vectorized arrays for '"+review+ "' is:")
    print(vec)
    print("\n")
    prediction=classifier.predict(vec)
    #prediction=random.random()
#     print("The polarity of "+review+" suggested by the model is "+prediction)
   
    return prediction;

# result=predict_review("Amazon is a good company")
# print(result)

st.title("Major 1 Project (Opinion Mining and Sentiment Analysis)")
html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Opinion Mining and Sentiment Analysis</h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
text = st.text_input("Review","Type the review here")
result=""
if st.button("Predict"):
    result=predict_review(text)
st.success('The output is {}'.format(result))
           
