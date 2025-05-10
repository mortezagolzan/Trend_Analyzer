import streamlit as st
from datetime import datetime, timedelta
from scrapers.reddit_scraper import search_reddit_posts
from core.text_processor import process_and_store_texts, get_similar_chunks, create_vector_store, create_conversation_chain, duckduckgo_search
from core.trend_core import get_cache_key, get_cached_results, cache_results, generate_wordcloud, format_key_insights

from pathlib import Path

st.set_page_config(
    page_title="Trend Analyzer",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .main {
        padding: 2rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        line-height: 1.6;
        color: #222;
        font-weight: 600;
        font-size: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .summary-box strong, .summary-box .highlight {
        color: #d97706;
        background: #fffbe6;
        padding: 0 0.2em;
        border-radius: 0.2em;
        font-weight: 700;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e3f2fd;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    .chat-message .content {
        margin-top: 0.5rem;
    }
    .source-citation {
        font-size: 0.9em;
        color: #666;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #ddd;
    }
    .source-link {
        color: #1a73e8;
        text-decoration: none;
        margin-left: 0.5rem;
        cursor: pointer;
    }
    .source-link:hover {
        text-decoration: underline;
    }
    .source-details {
        margin-top: 0.25rem;
        font-size: 0.9em;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
    }
    .source-content {
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 0.25rem;
        display: none;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_source" not in st.session_state:
    st.session_state.show_source = None
if "chat_expanded" not in st.session_state:
    st.session_state.chat_expanded = True
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

CACHE_DIR = Path("data/.cache")
CACHE_DIR.mkdir(exist_ok=True)

print("Streamlit script started")
print("Analysis done:", st.session_state.analysis_done)
print("Chat input value:", st.session_state.get("chat_input", None))

with st.sidebar:
    st.title("📊 Trend Analyzer")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app analyzes trends from Reddit using AI.
    
    Features:
    - Real-time trend analysis
    - AI-powered summarization
    - Interactive chat interface
    """)
    
    st.markdown("---")
    st.markdown("### Settings")
    cache_enabled = st.checkbox("Enable Caching", value=True)
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and LangChain")

st.title("Trend Analysis Dashboard")

with st.container():
    st.subheader("Search Parameters")
    topic = st.text_input(
        "Enter topic or keyword",
        placeholder="e.g., artificial intelligence, climate change"
    )
    source = "Reddit"
    st.markdown("**Source:** Reddit")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=7)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now()
        )

analyze_button = st.button("Analyze Trends", type="primary")

if analyze_button:
    if not topic:
        st.warning("Please enter a topic or keyword to analyze.")
    else:
        with st.spinner("Analyzing trends... This may take a few moments."):
            cache_key = get_cache_key(topic, source, start_date, end_date)
            cached_results = get_cached_results(cache_key) if cache_enabled else None
            
            if cached_results:
                st.info("Using cached results...")
                vector_store, key_insights, conversation_chain = cached_results
                st.session_state.conversation_chain = conversation_chain or create_conversation_chain(vector_store)
            else:
                content = search_reddit_posts(topic, start_date=start_date, end_date=end_date)
                source_name = "Reddit"
                if not content:
                    st.error("No content found or an error occurred while searching.")
                    st.session_state.analysis_done = False
                    st.session_state.conversation_chain = None
                else:
                    st.success(f"Found {len(content)} items from {source_name}!")
                    with st.spinner("Processing content and generating insights..."):
                        metadata = {
                            "source": source_name,
                            "topic": topic,
                            "date_range": f"{start_date} to {end_date}"
                        }
                        vector_store, key_insights, conversation_chain, insight_sources = process_and_store_texts(content, metadata)
                        if cache_enabled:
                            cache_results(cache_key, vector_store, key_insights, conversation_chain)
            st.session_state.key_insights = key_insights
            st.session_state.insight_sources = insight_sources
            st.session_state.similar_chunks = [chunk.page_content for chunk in get_similar_chunks(vector_store, topic)[:5]]
            st.session_state.vector_store = vector_store
            conversation_chain = create_conversation_chain(vector_store)
            st.session_state.conversation_chain = conversation_chain
            st.session_state.analysis_done = True
            st.rerun()

if st.session_state.analysis_done:
    st.subheader("Analysis Results")
    with st.expander("Summary", expanded=True):
        st.markdown("### Word Cloud Analysis")
        vector_store = st.session_state.get("vector_store", None)
        if vector_store is not None:
            try:
                docs = vector_store.similarity_search("", k=1000)
                if docs:
                    all_content = " ".join([doc.page_content for doc in docs])
                    wordcloud_img = generate_wordcloud(all_content)
                    st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{wordcloud_img}" alt="Word Cloud"></div>', unsafe_allow_html=True)
                else:
                    st.warning("No content available for word cloud generation.")
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")
        else:
            st.warning("No analysis results available. Please run an analysis first.")
        st.markdown("### Key Insights")
        key_insights = st.session_state.get("key_insights", [])
        insight_sources = st.session_state.get("insight_sources", [])
        if isinstance(key_insights, list):
            formatted_insights = format_key_insights(key_insights, insight_sources)
            st.markdown(formatted_insights, unsafe_allow_html=True)
        else:
            st.write(key_insights)
    with st.expander("Chat", expanded=st.session_state.chat_expanded):
        st.markdown("### Ask questions about the topic")
        vector_store = st.session_state.get("vector_store", None)
        history = st.session_state.chat_history
        i = 0
        while i < len(history):
            if history[i]["role"] == "user":
                st.markdown(f"""
                    <div style="
                        background: #1976d2;
                        color: #fff;
                        font-size: 1.1rem;
                        font-weight: 500;
                        margin-bottom: 0.5rem;
                        margin-left: 25%;
                        margin-right: 0;
                        border-radius: 1.2rem 1.2rem 0 1.2rem;
                        padding: 1rem 1.2rem;
                        max-width: 70%;
                        float: right;
                        clear: both;
                        text-align: right;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    ">
                        <span style="font-size:0.9em; opacity:0.7;">You</span><br>
                        {history[i]['content']}
                    </div>
                """, unsafe_allow_html=True)
                if i + 1 < len(history) and history[i+1]["role"] == "assistant":
                    st.markdown(f"""
                        <div style="
                            background: #f5f5f5;
                            color: #222;
                            font-size: 1.1rem;
                            font-weight: 500;
                            margin-bottom: 1.2rem;
                            margin-right: 25%;
                            margin-left: 0;
                            border-radius: 1.2rem 1.2rem 1.2rem 0;
                            padding: 1rem 1.2rem;
                            max-width: 70%;
                            float: left;
                            clear: both;
                            text-align: left;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                        ">
                            <span style="font-size:0.9em; opacity:0.7;">Assistant</span><br>
                            {history[i+1]['content']}
                        </div>
                    """, unsafe_allow_html=True)
                    i += 2
                else:
                    i += 1
            else:
                st.markdown(f"""
                    <div style="
                        background: #f5f5f5;
                        color: #222;
                        font-size: 1.1rem;
                        font-weight: 500;
                        margin-bottom: 1.2rem;
                        margin-right: 25%;
                        margin-left: 0;
                        border-radius: 1.2rem 1.2rem 1.2rem 0;
                        padding: 1rem 1.2rem;
                        max-width: 70%;
                        float: left;
                        clear: both;
                        text-align: left;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                    ">
                        <span style="font-size:0.9em; opacity:0.7;">Assistant</span><br>
                        {history[i]['content']}
                    </div>
                """, unsafe_allow_html=True)
                i += 1
        st.markdown('<div style="clear: both"></div>', unsafe_allow_html=True)
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input("Your question:", key="chat_input")
            submitted = st.form_submit_button("Send")
            if submitted and user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                if st.session_state.conversation_chain:
                    with st.spinner("Thinking..."):
                        response = st.session_state.conversation_chain({"question": user_question})
                        source_docs = response.get("source_documents", [])
                        sources_info = []
                        for doc in source_docs:
                            source_info = {
                                "source": doc.metadata.get("source", "Unknown"),
                                "username": doc.metadata.get("username", ""),
                                "timestamp": doc.metadata.get("timestamp", ""),
                                "content": doc.page_content,
                                "index": doc.metadata.get("source_index", 0)
                            }
                            sources_info.append(source_info)
                        assistant_answer = response["answer"].strip().lower()
                        uncertainty_phrases = [
                            "i don't know",
                            "not sure",
                            "don't have enough information",
                            "cannot answer",
                            "unable to find",
                            "no information available"
                        ]
                        if any(phrase in assistant_answer for phrase in uncertainty_phrases):
                            with st.spinner("Searching the web for more information..."):
                                web_answer = duckduckgo_search(user_question)
                                final_answer = f"Based on web search:\n\n{web_answer}"
                        else:
                            final_answer = response["answer"]
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": final_answer,
                            "sources": sources_info
                        })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Sorry, the conversation chain is not initialized. Please run analysis again.",
                        "sources": []
                    })
                st.session_state.chat_expanded = True
                st.rerun()