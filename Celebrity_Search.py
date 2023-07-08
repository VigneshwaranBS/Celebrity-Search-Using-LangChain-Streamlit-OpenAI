import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain , SimpleSequentialChain , SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st
os.environ['OPENAI_API_KEY'] = openai_key


# headers = {
#     "authorization": st.secrets['openai_key'],
#     "content-type" : "applications/json"
# }

st.title("Celebrity Search")
input_text = st.text_input("Search the name of your fav celebrity")


# prompts templates

first_input_prompt =PromptTemplate(
    input_variables=['name'],
    template="Tell me about a person {name} "
    
)

# memory
person_mem = ConversationBufferMemory(input_key='name' , memory_key='chat_history')
dob_mem = ConversationBufferMemory(input_key='person', memory_key='chat_history')
desc_mem = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# llms 
llm = OpenAI(temperature=0.6)

chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True ,output_key="person",memory=person_mem)

# prompts templates
second_input_prompt =PromptTemplate(
    input_variables=['person'],
    template="When was {person} born on "
    
)

llm = OpenAI(temperature=0.6)

chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True ,output_key="dob",memory=dob_mem)


# prompts templates
third_input_prompt =PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in india "
    
)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True ,output_key="desc", memory=desc_mem)


parent_chain = SimpleSequentialChain(
    chains=[chain,chain2],verbose=True
)


parent_chain1 = SequentialChain(
    chains=[chain,chain2,chain3],
    input_variables=['name'],
    output_variables=['person','dob','desc'],
    verbose=True
)


if input_text:
    # st.write(parent_chain.run(input_text))
    st.write(parent_chain1({"name":input_text}))
    
    with st.expander('Person name :'):
        st.info(person_mem.buffer)
        
    with st.expander('Major events :'):
        st.info(desc_mem.buffer)    
        
        
    