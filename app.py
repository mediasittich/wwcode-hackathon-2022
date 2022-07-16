import streamlit as st
import pandas as pd
from viz import *

def get_prediction(user_text):
    """
    Input: any string
    Returns: model predictions
    """
    import pickle
    # get model and vectorizer from pickle file
    with open('data/logreg.pkl', 'rb') as file:  
        logreg = pickle.load(file)
    with open('data/vec.pkl', 'rb') as file:
        vec = pickle.load(file)
    # get predictions
    text = vec.transform([user_text])
    preds = logreg.predict(text)
    proba = logreg.predict_proba(text)
    return preds[0], proba


st.title("Fake News Classifier")
st.write("""Recent studies have shown that credible and positive climate news reporting can change \
people’s viewpoints on climate change, particularly for people who oppose climate action. \
In order to make people aware of fake news, we use methods from natural language processing,  \
statistics and machine learning to inform users about the credibility of a given climate news article.""")

# get user to enter news
user_input = st.text_area("Enter news: ", "", 
help="Enter some news here to find out how accurate the information is.", height=60)

classify_button = st.button("Submit")

# once pressed, display prediction results based on user_input
if classify_button:
    results, proba = get_prediction(user_input)
    if results == 1:
        st.warning("The text is {:.2%} likely to be inaccurate!".format(proba[0][1]))
    if results == 0:
        st.success("The text is {:.2%} likely to be accurate!".format(proba[0][0]))

    st.subheader("Text Analysis:")
    # create and generate a word cloud image:
    create_wordcloud(user_input)
    # create bigram
    create_bigram(user_input)
    # sentiment analysis
    get_sentiment(user_input)


