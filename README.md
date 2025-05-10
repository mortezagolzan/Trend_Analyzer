# Trend Analyzer

A Streamlit application that analyzes trends from Twitter and Reddit using AI-powered text processing and summarization.

## Features

- Real-time trend analysis from Twitter and Reddit
- AI-powered text processing and summarization
- Interactive chat interface for asking questions
- Source citations with detailed metadata
- Caching system for improved performance
- Beautiful and responsive UI

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd trend-analyzer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=your_reddit_user_agent_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter a topic or keyword to analyze
2. Select the data source (Twitter, Reddit, or Auto)
3. Choose a date range (optional)
4. Click "Analyze Trends" to start the analysis
5. View the generated summary and key insights
6. Use the chat interface to ask questions about the topic

## Caching

The application includes a caching system to improve performance:
- Results are cached based on the topic, source, and date range
- Caching can be enabled/disabled in the sidebar settings
- Cache files are stored in the `.cache` directory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 