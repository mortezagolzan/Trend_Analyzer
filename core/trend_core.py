import hashlib
import json
from pathlib import Path
from datetime import datetime
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import re

CACHE_DIR = Path("data/.cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_key(topic: str, source: str, start_date: datetime, end_date: datetime) -> str:
    key_str = f"{topic}_{source}_{start_date}_{end_date}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cached_results(cache_key: str) -> tuple:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return None
            data = json.loads(content)
            if not isinstance(data, dict) or 'documents' not in data:
                return None
            documents = []
            for doc_data in data['documents']:
                if isinstance(doc_data, dict) and 'content' in doc_data:
                    documents.append(Document(
                        page_content=doc_data['content'],
                        metadata=doc_data.get('metadata', {})
                    ))
            if not documents:
                return None
            embeddings = OpenAIEmbeddings()
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=f"cached_{cache_key}"
            )
            return (
                vector_store,
                data.get('summary', ''),
                data.get('conversation_chain', None)
            )
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error reading cache file: {str(e)}")
        try:
            cache_file.unlink()
        except:
            pass
        return None

def cache_results(cache_key: str, vector_store, summary, conversation_chain):
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        documents = []
        store_data = vector_store.get()
        if not store_data or 'documents' not in store_data:
            return
        for doc in store_data['documents']:
            if hasattr(doc, 'page_content'):
                documents.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                })
        if not documents:
            return
        cache_data = {
            'documents': documents,
            'summary': summary,
            'conversation_chain': conversation_chain
        }
        temp_file = cache_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(cache_data, f)
        temp_file.rename(cache_file)
    except Exception as e:
        print(f"Error caching results: {str(e)}")
        try:
            temp_file.unlink()
        except:
            pass

def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=600,                
        height=300,               
        background_color='#18191a', 
        max_words=200,
        contour_width=2,
        contour_color='steelblue',
        colormap='tab20',         
        prefer_horizontal=0.7,
        min_font_size=10,
        max_font_size=90,
        random_state=42,
        scale=2,
        margin=2
    ).generate(text)
    plt.figure(figsize=(6, 3), dpi=150) 
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str

def format_key_insights(insights, sources=None):
    if not insights:
        return ""
    if isinstance(insights, str):
        points = [point.strip() for point in insights.split('\n') if point.strip()]
    else:
        points = insights
    items = []
    if sources is None:
        sources = [None] * len(points)
    bold_pattern = re.compile(r'\*\*(.*?)\*\*')
    for point, url in zip(points, sources):
        html_point = bold_pattern.sub(r'<strong>\1</strong>', point)
        if url:
            items.append(f'<li>{html_point} <a href="{url}" target="_blank" style="color:#1a73e8;">[source]</a></li>')
        else:
            items.append(f'<li>{html_point}</li>')
    return '<ul>' + ''.join(items) + '</ul>' 