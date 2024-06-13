import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def summarize(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=50,
        min_length=25,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app
st.title("Arabic Text Summarization")

input_text = st.text_area("Enter Arabic text here:")
if st.button("Summarize"):
    if input_text.strip():
        tokenizer, model = load_model()
        summary = summarize(input_text, tokenizer, model)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")
