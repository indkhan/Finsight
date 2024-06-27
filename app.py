import os
import streamlit as st
from typing import List, Dict
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set your API keys
from dotenv import load_dotenv
load_dotenv()

# Set up caching
set_llm_cache(InMemoryCache())

@st.cache_resource
def load_data(urls: List[str]):
    loader = WebBaseLoader(urls)
    data = loader.load()
    return data

@st.cache_resource
def split_text(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(_data)
    return all_splits

@st.cache_resource
def create_vectorstore(_splits):
    vectorstore = Chroma.from_documents(documents=_splits, embedding=OpenAIEmbeddings())
    return vectorstore

@st.cache_resource
def setup_qa_chain(_vectorstore):
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    Always include the source of your information.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

def check_confidence(question, answer, vectorstore):
    question_embedding = OpenAIEmbeddings().embed_query(question)
    answer_embedding = OpenAIEmbeddings().embed_query(answer)
    similarity = cosine_similarity([question_embedding], [answer_embedding])[0][0]
    return similarity

def ask_question(question: str, qa_chain, vectorstore, max_retries: int = 3) -> Dict:
    for attempt in range(max_retries):
        try:
            result = qa_chain({"query": question})
            answer = result["result"]
            sources = [doc.metadata.get('source', 'Unknown') for doc in result["source_documents"]]
            
            confidence = check_confidence(question, answer, vectorstore)
            
            if confidence < 0.5:  # Adjust this threshold as needed
                answer += " (Low confidence: This answer may not be reliable)"
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)}
            st.warning(f"An error occurred: {e}. Retrying...")

def main():
    st.title("RAG Q&A System")

    # Sidebar for URL input
    st.sidebar.header("Configuration")
    urls = st.sidebar.text_area("Enter URLs (one per line)", "https://www.example.com").split('\n')
    urls = [url.strip() for url in urls if url.strip()]

    if st.sidebar.button("Load Data"):
        with st.spinner("Loading and processing data..."):
            data = load_data(urls)
            splits = split_text(data)
            vectorstore = create_vectorstore(splits)
            qa_chain = setup_qa_chain(vectorstore)
            st.session_state['qa_chain'] = qa_chain
            st.session_state['vectorstore'] = vectorstore
        st.sidebar.success("Data loaded successfully!")

    # Main area for question input and answer display
    st.header("Ask a Question")
    question = st.text_input("Enter your question:")

    if st.button("Ask"):
        if 'qa_chain' not in st.session_state or 'vectorstore' not in st.session_state:
            st.error("Please load the data first!")
        elif not question:
            st.warning("Please enter a question!")
        else:
            with st.spinner("Thinking..."):
                result = ask_question(question, st.session_state['qa_chain'], st.session_state['vectorstore'])
            
            if "error" in result:
                st.error(f"An error occurred: {result['error']}")
            else:
                st.subheader("Answer:")
                st.write(result['answer'])
                
                st.subheader("Confidence:")
                st.progress(result['confidence'])

if __name__ == "__main__":
    main()