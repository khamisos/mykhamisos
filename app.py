import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

@st.cache_resource
def load_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer.src_lang = "ar_AR"
    tokenizer.tgt_lang = "ar_AR"
    return tokenizer, model

def summarize(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=50,
        min_length=25,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        forced_bos_token_id=tokenizer.lang_code_to_id["ar_AR"]
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
