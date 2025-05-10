from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.tools import DuckDuckGoSearchRun
import os
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

required_vars = ['OPENAI_API_KEY', 'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def duckduckgo_search(query: str) -> str:
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        print(f"Error performing web search: {str(e)}")
        return "Sorry, I couldn't perform a web search at this time."

def clean_metadata(metadata: dict) -> dict:
    if not isinstance(metadata, dict):
        return {}
    cleaned = {}
    for key, value in metadata.items():
        if value is not None:
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, datetime):
                cleaned[key] = value.isoformat()
            else:
                cleaned[key] = str(value)
    return cleaned

def create_documents(texts, metadatas=None):
    documents = []
    for i, text in enumerate(texts):
        doc_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
        cleaned_metadata = clean_metadata(doc_metadata)
        documents.append(Document(page_content=text, metadata=cleaned_metadata))
    return documents

def create_vector_store(documents):
    if not documents:
        raise ValueError("No documents provided to create vector store")
    embeddings = OpenAIEmbeddings()
    filtered_documents = []
    for doc in documents:
        if not isinstance(doc, Document):
            continue
        cleaned_metadata = clean_metadata(doc.metadata)
        filtered_documents.append(Document(
            page_content=doc.page_content,
            metadata=cleaned_metadata
        ))
    if not filtered_documents:
        raise ValueError("No valid documents after filtering")
    return Chroma.from_documents(documents=filtered_documents, embedding=embeddings)

def get_similar_chunks(vector_store, query, k=5):
    return vector_store.similarity_search(query, k=k)

def create_conversation_chain(vector_store):
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4o"
    )
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template="""You are a helpful AI assistant analyzing social media trends. Use the following pieces of context to answer the question at the end. \
        If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n        Context: {context}\n\n        Chat History: {chat_history}\n        Human: {question}\n        Assistant:"""
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

def process_and_store_texts(texts, metadata=None):
    if texts and isinstance(texts[0], tuple):
        contents, metadatas = zip(*texts)
    else:
        contents = texts
        metadatas = [metadata] * len(texts)
    documents = create_documents(contents, metadatas)
    vector_store = create_vector_store(documents)
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
    insights_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Analyze the following text and provide 5 key insights about the topic. Each insight should be concise, meaningful, and highlight important trends or patterns:\n        {text}\n\n        Format the response as a list of bullet points, with each point on a new line starting with a dash (-). Focus on extracting meaningful insights rather than just summarizing the content."""
    )
    insights_chain = LLMChain(llm=llm, prompt=insights_prompt)
    try:
        combined_text = "\n".join([doc.page_content for doc in documents])
        insights = insights_chain.run(text=combined_text)
        insights_list = [point.strip('- ').strip() for point in insights.split('\n') if point.strip()]
        insight_sources = []
        for insight in insights_list:
            similar_doc = vector_store.similarity_search(insight, k=1)[0]
            url = similar_doc.metadata.get("url", "")
            insight_sources.append(url)
        conversation_chain = create_conversation_chain(vector_store)
        return vector_store, insights_list, conversation_chain, insight_sources
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        return vector_store, ["Error generating insights."], None, []

def display_chat_history(history):
    i = 0
    while i < len(history):
        if history[i]["role"] == "user":
            st.markdown(f"""
                <div style="
                    background-color: #1a2233;
                    color: #fff;
                    font-size: 1.1rem;
                    font-weight: 500;
                    margin-bottom: 0.5rem;
                    margin-left: 30%;
                    margin-right: 0;
                    border-radius: 1rem 1rem 0 1rem;
                    padding: 1rem 1.2rem;
                    max-width: 70%;
                    float: right;
                    clear: both;
                    text-align: right;
                ">
                    {history[i]['content']}
                </div>
            """, unsafe_allow_html=True)
            if i + 1 < len(history) and history[i+1]["role"] == "assistant":
                st.markdown(f"""
                    <div style="
                        background-color: #23272f;
                        color: #f7f7f7;
                        font-size: 1.1rem;
                        font-weight: 500;
                        margin-bottom: 1.2rem;
                        margin-right: 30%;
                        margin-left: 0;
                        border-radius: 1rem 1rem 1rem 0;
                        padding: 1rem 1.2rem;
                        max-width: 70%;
                        float: left;
                        clear: both;
                        text-align: left;
                    ">
                        {history[i+1]['content']}
                    </div>
                """, unsafe_allow_html=True)
                i += 2
            else:
                i += 1
        else:
            st.markdown(f"""
                <div style="
                    background-color: #23272f;
                    color: #f7f7f7;
                    font-size: 1.1rem;
                    font-weight: 500;
                    margin-bottom: 1.2rem;
                    margin-right: 30%;
                    margin-left: 0;
                    border-radius: 1rem 1rem 1rem 0;
                    padding: 1rem 1.2rem;
                    max-width: 70%;
                    float: left;
                    clear: both;
                    text-align: left;
                ">
                    {history[i]['content']}
                </div>
            """, unsafe_allow_html=True)
            i += 1 