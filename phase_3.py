import os
import warnings
import logging
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st

# Load .env for API key
load_dotenv()

# LangChain & PDF tools
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Disable warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Streamlit Page Config
st.set_page_config(page_title="RAG Chatbot", layout="centered")

# --- Viewport for Mobile Responsiveness ---
st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
""", unsafe_allow_html=True)

# --- Logo Container ---
try:
    st.image("./logo.png", use_column_width=False, width=200)
except FileNotFoundError:
    st.error("Logo image not found. Please ensure 'logo.png' is in the project directory.")

# --- Custom CSS for Dark UI and Animation ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700;800&family=Poppins:wght@300&display=swap');

    body {
        background-color: #2A2A2A;
        color: #E0E0E0 !important;
    }
    .main {
        background-color: #3A3A3A;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        margin-top: 20px; /* Space below logo */
    }

    /* Logo Container */
    .stImage {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px; /* Fixed height for vertical centering */
        width: 100%;
        margin-bottom: 10px;
    }

    /* Fade-In Header */
    .fadein-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }

    .fadein-text {
        opacity: 0;
        animation: fadeIn 2s ease-in-out forwards;
        font-size: 26px;
        font-family: 'Roboto', sans-serif;
        color: #E0E0E0 !important;
        font-weight: 800;
        letter-spacing: 0.01em;
        text-align: center;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        max-width: 90%;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Title Scale Animation */
    .stApp h1 {
        animation: scale 3s ease-in-out infinite;
        color: #E0E0E0 !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 300;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }

    @keyframes scale {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    /* Input Glow */
    input {
        color: #FFFFFF !important;
        background-color: #4A4A4A !important;
    }
    input::placeholder {
        color: #BBBBBB !important;
    }
    input:focus {
        border: 2px solid #34C759 !important;
        box-shadow: 0 0 8px rgba(52, 199, 89, 0.3);
        transition: all 0.3s ease-in-out;
    }

    /* Chat Bubbles */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #4A6FFA !important;
        border-radius: 12px;
        padding: 10px;
        margin: 5px 0;
        color: #FFFFFF !important;
    }
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: #4CAF50 !important;
        border-radius: 12px;
        padding: 10px;
        margin: 5px 0;
        color: #FFFFFF !important;
    }

    /* Download Button */
    .stDownloadButton button {
        background-color: #1A73E8;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 8px 16px;
    }

    /* Responsive tweaks for mobile screens */
    @media (max-width: 600px) {
        .fadein-text {
            font-size: 18px;
            letter-spacing: 0.005em;
        }
        .stApp h1 {
            font-size: 24px;
        }
        .main, .block-container {
            padding: 15px;
        }
        .stImage {
            height: 80px;
        }
        .stImage img {
            width: 150px !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Animated Header ---
st.markdown("""
    <div class="fadein-container">
        <div class="fadein-text">RAG Chatbot by Muhammad Aaqib Shaikh(58621) & Muhammad Saqib Shaikh(58620)</div>
    </div>
""", unsafe_allow_html=True)

st.title("RAG Chatbot")

# --- Chat History Setup ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# --- Load Vector Store with Progress Indicator ---
@st.cache_resource
def get_vectorstore():
    with st.spinner("Loading document..."):
        pdf_name = "./reflexion.pdf"
        loaders = [PyPDFLoader(pdf_name)]
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        ).from_loaders(loaders)
        st.info("Document loaded successfully!")
        return index.vectorstore

# --- Optional: Precomputed Vector Store (Commented for reference) ---
# from langchain.vectorstores import FAISS
# def load_or_create_vectorstore():
#     vectorstore_path = "./vectorstore"
#     if os.path.exists(vectorstore_path):
#         embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
#         return FAISS.load_local(vectorstore_path, embeddings)
#     else:
#         with st.spinner("Creating vector store..."):
#             pdf_name = "./reflexion.pdf"
#             loaders = [PyPDFLoader(pdf_name)]
#             index = VectorstoreIndexCreator(
#                 embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
#                 text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#             ).from_loaders(loaders)
#             index.vectorstore.save_local(vectorstore_path)
#             st.info("Vector store created and saved!")
#             return index.vectorstore

# --- Chat Input ---
prompt = st.chat_input("Enter your question here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template("""
        You are very smart at everything, you always give the best, 
        the most accurate and most precise answers. Answer the following Question: {user_prompt}.
        Start the answer directly. No small talk please.
    """)

    model = "llama3-8b-8192"

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load document")

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Export Chat History with Timestamps ---
if st.session_state.get("messages"):
    def format_chat():
        chat_lines = []
        for msg in st.session_state.messages:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            role = msg['role'].capitalize()
            content = msg['content']
            chat_lines.append(f"[{timestamp}] {role}: {content}")
        return "\n\n".join(chat_lines)

    st.download_button(
        label="📥 Save Chat History",
        data=format_chat(),
        file_name="chat_history.txt",
        mime="text/plain"
    )