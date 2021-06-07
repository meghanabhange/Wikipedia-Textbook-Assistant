import string
from pathlib import Path

import pandas as pd
import pke
import streamlit as st
import wikipedia
from nltk.corpus import stopwords
from transformers import pipeline


@st.cache(allow_output_mutation=True)
def load_nlp():
    nlp = pipeline("question-answering")
    return nlp


nlp = load_nlp()

extractor = pke.unsupervised.TopicRank()
commonwords = open("commonwords.txt").readlines()
commonwords = [words.replace("\n", "") for words in commonwords]


def extract_keyphrase(textbook_excerpt, extractor, n=5):
    open("textbook.txt", "w").write(textbook_excerpt)
    extractor.load_document(input="textbook.txt", language="en")
    pos = {"NOUN", "PROPN"}
    stoplist = list(string.punctuation)
    stoplist += ["-lrb-", "-rrb-", "-lcb-", "-rcb-", "-lsb-", "-rsb-"]
    # stoplist += commonwords[:100]
    stoplist += stopwords.words("english")
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    extractor.candidate_weighting(threshold=0.74, method="average")
    keyphrases = extractor.get_n_best(n=n + 5)
    output = []
    for key in keyphrases:
        if not (str(key[0]).strip() in commonwords):
            output.append(key[0])
    if len(output) < n:
        return output
    return output[:n]

def get_wikipedia(key):
  key = wikipedia.search(key)[0]
  summary = wikipedia.summary(key)
  url = wikipedia.page(key).url
  return summary, url


st.title("Wikipedia Recommender")

st.markdown("### Enter Your Excerpt Here")
textbook_excerpt = st.text_area("Enter Your Excerpt Here")

if not textbook_excerpt:
    st.warning("Please Enter an Excerpt")
    st.stop()
st.success("Processing Excerpt")

keyphrases = extract_keyphrase(textbook_excerpt, extractor, 5)
keyphrases = [key.title() for key in keyphrases]


if keyphrases:
    st.write(
        f"Hey, I found wikipedia information about {', '.join(keyphrases)} click on them to know more"
    )

    keyphrase = st.selectbox("Keyphrases", tuple(keyphrases))
    summary, url = get_wikipedia(keyphrase)
    context = summary
    answer = nlp(question=f"What is {keyphrase}?", context=context)
    st.markdown(f"### What is {keyphrase}?")
    st.write(answer["answer"])
    st.markdown("### Wikipedia Summary:")
    st.write(summary)
    st.write(f"Read more on wikipedia : {url}")
