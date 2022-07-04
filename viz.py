import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('vader_lexicon')

# set stopwords
stop_words = stopwords.words('english')

def create_wordcloud(text):
    # Create and generate a word cloud image:
    wordcloud = WordCloud(stopwords = stop_words, max_words=100, mode = "RGBA", background_color=None).generate(text)
    fig, ax = plt.subplots(figsize = (8,8))
    # Display the generated image:
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)


def create_bigram(text):
    vect = CountVectorizer(stop_words=stop_words, ngram_range=(2,2))
    bigrams = vect.fit_transform([text])
    bigram_df = pd.DataFrame(bigrams.toarray(), columns=vect.get_feature_names())
    bigram_frequency = pd.DataFrame(bigram_df.sum(axis=0)).reset_index()
    bigram_frequency.columns = ['bigram', 'frequency']
    bigram_frequency = bigram_frequency.sort_values(by='frequency', ascending=False).head(10)
    # plot bigram freq
    fig = px.bar(bigram_frequency, x="bigram", y="frequency",  title="Bigram Frequency")
    st.plotly_chart(fig, use_container_width=True)

def get_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(text)

    pos = round((sentiment_score['pos']), 2)*100
    neg = round((sentiment_score['neg']), 2)*100
    neutral = round((sentiment_score['neu']), 2)*100
    compound = round((sentiment_score['compound']), 2)*100

    vader_df = pd.DataFrame({"text":[text], "positive":[pos], "negative":[neg], "neutral":[neutral]})
    vader_df = vader_df[['positive', 'negative', 'neutral']].T.reset_index()
    vader_df.columns = ['sentiment', '%']

    fig = px.pie(vader_df, values='%', names='sentiment',title="Sentiment")
    fig.update_traces(texttemplate="%{percent:.0%}")
    st.plotly_chart(fig, use_container_width=True)
