import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Cohere API configuration
    COHERE_API_KEY = "VdILqQqDVEPD2njZCu2cRa9xYJgPdrRBNDx6miuE"
    
    # Fallback for testing - replace with your actual API key
    if not COHERE_API_KEY:
        COHERE_API_KEY = "VdILqQqDVEPD2njZCu2cRa9xYJgPdrRBNDx6miuE"  # Replace with actual key
    EMBEDDING_MODEL = "embed-multilingual-v3.0"
    CHAT_MODEL = "command-r-plus"
    
    # Vector database configuration
    VECTOR_DB_PATH = "./data/vector_db"
    
    # Text processing configuration
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    
    # Conversation configuration
    MAX_CONVERSATION_HISTORY = 4
    TEMPERATURE = 0.1
    
    # Retrieval configuration
    RETRIEVAL_K = 5
    
    # File processing configuration
    SUPPORTED_FILE_TYPES = ['.csv', '.txt', '.pdf']
    MAX_FILE_SIZE_MB = 50

    # FAISS Specific Settings
    FAISS_INDEX_TYPE = "IndexFlatL2"  # Can be changed to IndexIVFFlat, IndexHNSW, etc.
    SEARCH_K = 5  # Default number of results to return
    SCORE_THRESHOLD = 0.7  # Default similarity threshold
    
    # === GENERATION ORCHESTRATOR SETTINGS ===
    
    # Confidence Thresholds for Strategy Selection
    HIGH_CONFIDENCE_THRESHOLD = 0.8    # Results above this are considered high quality
    MEDIUM_CONFIDENCE_THRESHOLD = 0.5  # Results above this are considered usable
    LOW_CONFIDENCE_THRESHOLD = 0.3     # Results above this might still be helpful
    
    # Strategy Selection Parameters
    MIN_HIGH_CONF_RESULTS = 2          # Minimum high-confidence results for RAG_PRIMARY
    MIN_MEDIUM_CONF_RESULTS = 1        # Minimum medium-confidence results for RAG_AUGMENTED
    
    # Generation Settings
    MAX_CONTEXT_DOCS = 3               # Maximum documents to include in context
    ENABLE_WEB_SEARCH = False          # Set to True when you have Tavily/web search configured
    WEB_SEARCH_RESULTS = 5             # Number of web search results to retrieve
    
    # Temporal Detection Settings
    TEMPORAL_KEYWORDS = [
        "today", "latest", "current", "recent", "now", "this year",
        "2024", "2025", "breaking", "news", "update", "status",
        "yesterday", "tomorrow", "this week", "this month"
    ]
    
    CURRENT_INFO_DOMAINS = [
        "weather", "news", "stock", "price", "status", "availability",
        "market", "exchange", "live", "real-time", "trending"
    ]
    
    # Query Complexity Analysis
    SIMPLE_QUERY_WORD_LIMIT = 3        # Queries with <= words are considered simple
    COMPLEX_QUERY_WORD_LIMIT = 8       # Queries with > words are considered complex
    
    # Logging Configuration
    LOG_STRATEGY_DECISIONS = True      # Log strategy selection reasoning
    LOG_METRICS = True                 # Log detailed retrieval metrics
    LOG_GENERATION_TIME = True         # Log response generation timing
    
    # Performance Settings
    ASYNC_GENERATION = True            # Enable async generation
    MAX_GENERATION_TIME = 30           # Max seconds for generation (timeout)
    
    # Quality Control
    MIN_ANSWER_LENGTH = 50             # Minimum acceptable answer length
    MAX_ANSWER_LENGTH = 2000           # Maximum answer length before truncation
    
    # Fallback Behavior
    FALLBACK_TO_KNOWLEDGE = True       # Fall back to knowledge if RAG fails
    GRACEFUL_DEGRADATION = True        # Degrade gracefully on errors
    
    # Experimental Features
    ENABLE_SCORE_NORMALIZATION = True  # Normalize retrieval scores for better comparison
    ENABLE_VARIANCE_ANALYSIS = True    # Use score variance in strategy selection
    ADAPTIVE_THRESHOLDS = False        # Adapt thresholds based on query patterns (experimental)
    
    # Web Search Configuration (for when you implement it
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
    WEB_SEARCH_TIMEOUT = 10            # Timeout for web search requests
    COMBINE_RAG_AND_WEB = True         # Allow hybrid RAG + web search
    
    @classmethod
    def get_strategy_config(cls) -> dict:
        """Get all strategy-related configuration as a dictionary"""
        return {
            'high_threshold': cls.HIGH_CONFIDENCE_THRESHOLD,
            'medium_threshold': cls.MEDIUM_CONFIDENCE_THRESHOLD,
            'low_threshold': cls.LOW_CONFIDENCE_THRESHOLD,
            'min_high_conf': cls.MIN_HIGH_CONF_RESULTS,
            'min_medium_conf': cls.MIN_MEDIUM_CONF_RESULTS,
            'max_context_docs': cls.MAX_CONTEXT_DOCS,
            'temporal_keywords': cls.TEMPORAL_KEYWORDS,
            'current_info_domains': cls.CURRENT_INFO_DOMAINS
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        issues = []
        
        # Check API keys
        if not cls.COHERE_API_KEY or cls.COHERE_API_KEY == "your_api_key_here":
            issues.append("COHERE_API_KEY not set")
        
        # Check thresholds
        if not (0 <= cls.LOW_CONFIDENCE_THRESHOLD <= cls.MEDIUM_CONFIDENCE_THRESHOLD <= cls.HIGH_CONFIDENCE_THRESHOLD <= 1):
            issues.append("Confidence thresholds must be in ascending order between 0 and 1")
        
        # Check retrieval settings
        if cls.RETRIEVAL_K <= 0:
            issues.append("RETRIEVAL_K must be positive")
        
        if cls.MAX_CONTEXT_DOCS <= 0:
            issues.append("MAX_CONTEXT_DOCS must be positive")
        
        # Check timeouts
        if cls.MAX_GENERATION_TIME <= 0:
            issues.append("MAX_GENERATION_TIME must be positive")
        
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True