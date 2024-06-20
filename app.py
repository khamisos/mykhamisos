import streamlit as st
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

# Download the NLTK punkt tokenizer
nltk.download('punkt')

# Function to tokenize Arabic text into sentences using heuristics
def tokenize_arabic_text(text):
    # Split text by punctuation marks commonly used in Arabic to end sentences
    sentences = re.split(r'(?<=[.؟!])\s+', text)
    return sentences

# Function to summarize Arabic text using TextRank
def summarize_arabic_text(text, sentence_count=3):
    # Tokenize the text into sentences
    sentences = tokenize_arabic_text(text)
    
    # Join sentences to form a single string suitable for sumy
    text = " ".join(sentences)
    
    # Initialize the parser and tokenizer
    parser = PlaintextParser.from_string(text, Tokenizer("arabic"))
    summarizer = TextRankSummarizer()
    
    # Generate the summary
    summary = summarizer(parser.document, sentence_count)
    
    # Join the summarized sentences
    summary_sentences = [str(sentence) for sentence in summary]
    summary_text = " ".join(summary_sentences)
    
    return summary_text

# Streamlit application
st.title("أداة تلخيص النصوص العربية")
st.write("أدخل النص العربي الذي ترغب في تلخيصه وعدد الجمل المطلوبة في الملخص.")

text = st.text_area("أدخل النص هنا", height=200)
sentence_count = st.number_input("عدد الجمل في الملخص", min_value=1, value=3)

if st.button("تلخيص"):
    if text.strip() != "":
        summary = summarize_arabic_text(text, sentence_count)
        st.subheader("الملخص:")
        st.markdown(f'<div style="direction: rtl; text-align: right;">{summary}</div>', unsafe_allow_html=True)
    else:
        st.warning("يرجى إدخال نص للتلخيص.")
