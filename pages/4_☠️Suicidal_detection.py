import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

PAGE_TITLE = "Sucidal Tendency Detection | RNN"
PAGE_ICON = "☠️"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

model = load_model("dal_classification\Sucidal_classification.keras")


st.title("Titanic Survival Prediction")
st.image('Sucidal_classification\img.jpg',use_column_width=True)
st.write("Please enter the text you want to analyze in the text box and you will get the result as suicidal if the text has suicidal tendency or non suicidal if the text has non-suicidal tendency")

text = st.text_input("Enter the text you want to check")


voc_size = 5000
sent_length = 70


sen = re.sub('[^a-zA-Z]',' ', text)
sen = sen.lower()
sen = sen.split()
sen = [ps.stem(word) for word in sen if not word in stopwords.words('english')]
sen = ' '.join(sen)
encoded_sen = [one_hot(sen,voc_size)]
embeded_sen = pad_sequences(encoded_sen,padding='post',maxlen = sent_length)

pred = model.predict(embeded_sen)
posibility = pred.round(4)

if text :
    if posibility<0.5:
        st.info("This text does not have a sucidal tendency.")
        st.image('Sucidal_classification\non-sucidal.gif', use_column_width=True)

    else:
        st.warning("This text may have sucidal tendency.")
        st.image('Sucidal_classification\sucidal.gif', use_column_width=True)
        st.write(posibility)
