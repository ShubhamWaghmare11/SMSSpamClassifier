import streamlit as st
import requests
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

def text_transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    li=[]
    ps=PorterStemmer()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation and i.isalnum():
            li.append(ps.stem(i))
    return " ".join(li)

st.title("Email/SMS Spam Classifier")

sms=st.text_input("Enter the Message")


if st.button("predict"):
    # 1. preprocess
    transformed_sms= text_transform(sms)
    # 2. Vectorize
    vector_input=tfidf.transform([transformed_sms])

    # 3. predict
    result=model.predict(vector_input)[0]
    # 4. Display
    if result==1:
        st.header("Message is Spam")
    else:
        st.header("Message is not spam")

