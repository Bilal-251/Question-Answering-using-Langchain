from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

sec_key = os.getenv("HF_Token")

# LLM Part

repo_id = 'mistralai/Mistral-7B-Instruct-v0.2'

llm = HuggingFaceEndpoint(
    repo_id = repo_id,
    temperature= 0.7,
    model_kwargs= { 'max_length':128, 'token':sec_key}
)

# answer = llm.invoke("What is Machine Learning.")



# Streamlit App
st.title("Question Answering System")
st.write("Ask any question, and our model will provide an answer!")

# Input text box for the question
user_question = st.text_input("Enter your question:", placeholder="Type your question here")

if st.button("Generate Answer"):
    if user_question.strip():  # Ensure the question is not empty
        answer = llm.invoke(user_question)
        st.success("Answer:")
        st.write(answer)
    else:
        st.error("Please enter a valid question.")
