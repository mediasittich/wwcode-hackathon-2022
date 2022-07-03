import streamlit as st
import pandas as pd

def get_prediction(user_text):
    # sample classifier
    # run model prediction
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    train_df = pd.read_csv('data/fake-news/train.csv')
    test_df = pd.read_csv('data/fake-news/train.csv')
    train_df = train_df.iloc[:1000]
    test_df = test_df.iloc[:1000]
    train_df = train_df.fillna(" ")
    x = train_df['text']
    y = train_df['label']
    x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)
    # Vectorize text reviews to numbers
    vec = CountVectorizer(stop_words='english')
    x = vec.fit_transform(x).toarray()
    x_test = vec.transform(x_test).toarray()
    model = MultinomialNB()
    model.fit(x, y)
    text = vec.transform([user_text])
    preds = model.predict(text)
    proba = model.predict_proba(text)
    return preds[0], proba


st.title("Fake News Classifier")

# get user to enter news
user_input = st.text_area("Enter news: ", "", 
help="Enter some news here to find out how accurate the information is.", 
height=10)

classify_button = st.button("Submit")

# once pressed, display prediction results based on user_input
if classify_button:
    results, proba = get_prediction(user_input)
    if results == 1:
        st.warning("The text is {:.2%} likely to be inaccurate!".format(proba[0][1]))
    if results == 0:
        st.success("The text is {:.2%} likely to be accurate!".format(proba[0][0]))

x = st.slider("Select an integer x", 0, 10, 1)
y = st.slider("Select an integer y", 0, 10, 1)

df = pd.DataFrame({"x": [x], "y": [y] , "x + y": [x + y]}, index = ["addition row"])
st.write(df)

