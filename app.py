import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
#Loading IMDB dataset
word_index=imdb.get_word_index()

# Loading model
model=load_model("simple_rnn_model.h5")


st.title("Movie Sentiment Analysis")
review=st.text_input("Enter your review")

def preprocess_review(review):
    review=review.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in review]
    pad_sequence=sequence.pad_sequences([encoded_review],maxlen=500)
    return pad_sequence

def predict(review):
    processed_review=preprocess_review(review)
    prediction=model.predict(processed_review)
    sentiment="It is a Positive review" if prediction[0][0]>0.5 else "It is a Negative review"
    return sentiment

# Predicting Sentiment
if review:
    sentiment=predict(review)
    st.write(sentiment)

