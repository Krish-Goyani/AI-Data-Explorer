import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

st.set_page_config(
    page_title="AI Data Explorer",
    page_icon="💻",
)
st.header("AI Data Explorer with Gemini API",divider="rainbow")


api_key = st.sidebar.text_input("Enter your Gemini API key", type="password")

# File uploader for CSV file
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

# Function to create and return an agent
def create_agent(api_key, df, llm):
    
    # Create the pandas agent with the DataFrame and LLM
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True, 
        allow_dangerous_code=True
    )
    return agent

# Application description
st.markdown("""
## About this Application 🤖📊

This application allows you to explore and analyze your dataset using an AI-powered agent. 
You can upload a CSV file and provide your Gemini API key to create an agent capable of answering questions about your data.

### How to Use 🛠️
1. 🔑 Enter your Gemini API key in the sidebar.
2. 📁 Upload a CSV file containing your dataset.
3. ❓ Enter your query about the dataset in the input field provided.
4. 🚀 The AI agent will process your query and display the results.

The AI agent leverages the power of a LangChain and large language model (LLM) to understand and analyze your data, providing insights and answers based on your questions.
""")

# Process the uploaded CSV file and create the agent
if uploaded_file is not None and api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=api_key)

    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV file:")
    st.dataframe(df)

    agent = create_agent(api_key, df, llm)

    # Input field for user query
    user_query = st.text_input("Enter your query about the dataset")

    # Process the user query and display the result
    if user_query:
        with st.spinner('Processing your query...'):
            try:
                result = agent.run(user_query)
                st.success("Query result:")
                result
            except Exception as e:
                st.error(f"Error processing query: {e}")
else:
    st.write("Please enter your Gemini API key and upload a CSV file")