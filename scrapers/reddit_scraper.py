import praw
from typing import List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def get_reddit_client() -> Optional[praw.Reddit]:
    try:
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT', 'TrendAnalyzer/1.0')
        if not all([client_id, client_secret]):
            print("Reddit credentials not found in environment variables")
            return None
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    except Exception as e:
        print(f"Error initializing Reddit client: {e}")
        return None

def search_reddit_posts(keyword: str, max_posts: int = 100, start_date=None, end_date=None) -> List[str]:
    reddit = get_reddit_client()
    if not reddit:
        return []
    posts = []
    try:
        search_results = reddit.subreddit("all").search(keyword, limit=max_posts)
        for post in search_results:
            if len(posts) >= max_posts:
                break
            created_utc = datetime.utcfromtimestamp(post.created_utc)
            if start_date and created_utc.date() < start_date:
                continue
            if end_date and created_utc.date() > end_date:
                continue
            content = post.title
            if post.selftext:
                content += f"\n{post.selftext}"
            metadata = {
                "url": f"https://www.reddit.com{post.permalink}",
            }
            posts.append((content, metadata))
        return posts[:max_posts]
    except Exception as e:
        print(f"Error searching Reddit: {e}")
        return []

def format_key_insights(insights, sources):
    items = []
    for point, url in zip(insights, sources):
        if url:
            items.append(f'<li>{point} <a href="{url}" target="_blank" style="color:#1a73e8;">[source]</a></li>')
        else:
            items.append(f'<li>{point}</li>')
    return '<ul>' + ''.join(items) + '</ul>' 