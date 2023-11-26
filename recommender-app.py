import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


orig_data=pd.read_csv('DatasetEnglish.csv')
sentiment_data=pd.read_csv('sentiment_data.csv')
data_extra = pd.merge(orig_data, sentiment_data, how='left', on='talk_id')

def get_similarity():
    tfidf = TfidfVectorizer(stop_words='english')
    data_extra['combined_features'] = data_extra['talk_slug'] + ' ' + data_extra['talk_description'] + ' ' + data_extra['speakers_name'] + ' ' +data_extra['topic_0_name'] + ' ' + data_extra['transcript']
    data_extra['combined_features'] = data_extra['combined_features'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data_extra['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

if "cosine_sim" not in st.session_state:
    st.session_state.cosine_sim = get_similarity()
cosine_sim = st.session_state.cosine_sim

st.title("TED Talks Recommender")

selected_topics = st.multiselect("Filter by topic", data_extra['topic_0_name'])

df = data_extra[data_extra["topic_0_name"].isin(selected_topics)]

selected_talk = st.selectbox("Select a talk", df['talk_title'])

if selected_talk is None:
    st.write("Please select a talk from the dropdown menu.")
else:
    st.write(f"You have selected: {selected_talk}")
    st.write(f"Speaker: {df[df['talk_title'] == selected_talk]['speakers_name'].values[0]}")
    st.write(f"Description: {df[df['talk_title'] == selected_talk]['talk_description'].values[0]}")

    @st.cache_data
    def get_top_5_sentiment(selected_talk):
        selected_score = df[df["talk_title"] == selected_talk]["text_polarity_score"].values[0]
        df["score_diff"] = abs(df["text_polarity_score"] - selected_score)
        filtered_df = df.sort_values(by="score_diff")
        top_sent = filtered_df[filtered_df["talk_title"] != selected_talk].head(5)
        return top_sent

    if st.button("Get top 5 related talks by sentiment"):
        top_5_sent = get_top_5_sentiment(selected_talk)
        st.write("Top 5 related talks based on sentiment score:")
        st.dataframe(top_5_sent[["talk_title", "speakers_name", "shortened_url"]])

    @st.cache_data
    def get_top_5_cosine(selected_talk):
        talk_index = df[df["talk_title"] == selected_talk].index.values[0]
        similar_talks = list(enumerate(cosine_sim[talk_index]))
        similar_talks = sorted(similar_talks, key=lambda x: x[1], reverse=True)
        cosine_5 = similar_talks[1:6]
        return cosine_5

    if st.button("Get top 5 related talks by cosine similarity"):
        top_5_cosine = get_top_5_cosine(selected_talk)
        st.write("Top 5 related talks based on cosine similarity:")
        for i in range(5):
            related_talk = data_extra[data_extra.index ==top_5_cosine[i][0]]['talk_slug'].values[0]
            related_name = data_extra[data_extra['talk_slug'] == related_talk]['talk_title'].values[0]
            related_speaker = data_extra[data_extra['talk_slug'] == related_talk]['speakers_name'].values[0]
            url = data_extra[data_extra['talk_slug'] == related_talk]['shortened_url'].values[0]
            related_score = top_5_cosine[i][1]
            st.write(f"{i+1}. {related_name} by {related_speaker} URL: {url} (Cosine similarity: {related_score:.2f})")

