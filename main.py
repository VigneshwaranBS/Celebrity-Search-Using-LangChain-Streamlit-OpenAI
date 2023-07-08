import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st


os.environ['OPENAI_API_KEY'] = openai_key

st.title("Langchain tutorial using OpenAI")
input_text = st.text_input("Search the topic")

# llms 
llm = OpenAI(temperature=0.6)

if input_text:
    st.write(llm(input_text))
    