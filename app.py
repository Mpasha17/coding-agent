import streamlit as st
import os
import sys
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- App Configuration ---
st.set_page_config(page_title="AI Coding Agent", layout="wide")
st.title("🤖 AI Python Coding Agent with LangChain & Streamlit")

# --- Load API Key and Initialize LangChain (Best Practice using .env) ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Function to Initialize LangChain Components ---
# We use st.cache_resource to avoid re-initializing on every interaction
@st.cache_resource
def initialize_langchain(api_key):
    """Initializes LangChain components and returns the chain."""
    if not api_key:
        st.error("Error: GOOGLE_API_KEY environment variable not set.")
        st.info("Please create a .env file and add your GOOGLE_API_KEY, then restart the app.")
        return None

    try:
        # Initialize the ChatGoogleGenerativeAI model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        # Define the prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Python coding assistant. Your task is to generate clean, "
                    "efficient, and correct Python code based on the user's request. "
                    "Only output the raw Python code itself, without any explanations, comments outside the code, "
                    "markdown formatting (like ```python), or introductory/concluding remarks. "
                    "Ensure the code directly addresses the user's requirement."
                ),
                (
                    "human",
                    "{user_request}"
                ),
            ]
        )

        # Define the output parser
        output_parser = StrOutputParser()

        # Create the LangChain Chain using LCEL
        coding_agent_chain = (
            {"user_request": RunnablePassthrough()}
            | prompt_template
            | llm
            | output_parser
        )
        return coding_agent_chain

    except Exception as e:
        st.error(f"Error initializing LangChain components: {e}")
        return None

# Initialize the chain
coding_agent_chain = initialize_langchain(API_KEY)

# --- Streamlit UI Elements ---

st.write("Enter your Python coding request below:")

# Use a text area for potentially longer coding requests
user_request = st.text_area("Your Request:", height=100, placeholder="e.g., create a Flask endpoint that returns 'Hello, World!'")

# Button to trigger code generation
generate_button = st.button("✨ Generate Code")

if generate_button and user_request:
    if coding_agent_chain:
        st.write("---")
        st.subheader("Generated Code:")
        # Show a spinner while generating
        with st.spinner("🤖 AI is thinking..."):
            try:
                # Invoke the chain
                generated_code = coding_agent_chain.invoke(user_request)

                # Display the generated code with syntax highlighting
                st.code(generated_code, language='python', line_numbers=True)

            except Exception as e:
                st.error(f"An error occurred during code generation: {e}")
    else:
         # This message appears if LangChain initialization failed
        st.warning("LangChain components not initialized. Please check API key and configuration.")

elif generate_button and not user_request:
    st.warning("Please enter your coding request first.")

# --- Optional: Sidebar for Info/Alternative API Key Input (Less Secure) ---
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses LangChain and Google's Gemini model to generate Python code based on your natural language requests.

    **Note:** Ensure your `GOOGLE_API_KEY` is set in a `.env` file for secure operation.
    """)
