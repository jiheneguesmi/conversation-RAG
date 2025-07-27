import streamlit as st
import asyncio
import logging
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go

# PROPERLY DISABLE STREAMLIT MAGIC FUNCTIONS
st.set_option('deprecation.showPyplotGlobalUse', False)

# Try to import your existing modules with flexible handling
try:
    from generation import SmartGenerationOrchestrator, CohereLLMClient, GenerationStrategy
    GENERATION_AVAILABLE = True
except ImportError:
    GENERATION_AVAILABLE = False

try:
    try:
        from retrieval import HybridRetriever
        RETRIEVAL_CLASS = HybridRetriever
    except ImportError:
        try:
            from Retrieval import ConversationalRAG
            RETRIEVAL_CLASS = ConversationalRAG
        except ImportError:
            try:
                from retrieval import ConversationalRAG
                RETRIEVAL_CLASS = ConversationalRAG
            except ImportError:
                RETRIEVAL_CLASS = None
    
    RETRIEVAL_AVAILABLE = RETRIEVAL_CLASS is not None
except Exception:
    RETRIEVAL_AVAILABLE = False

try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    class Config:
        COHERE_API_KEY = "your-cohere-api-key"
        TAVILY_API_KEY = "your-tavily-api-key" 
        CHAT_MODEL = "command-r-plus"
        TEMPERATURE = 0.1
        MAX_ANSWER_LENGTH = 1000

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Smart RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .strategy-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .source-item {
        background-color: #e8f4fd;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
    }
    /* Hide any debug information */
    .debug-info {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitRAGApp:
    """Streamlit application for the Smart RAG System"""
    
    def __init__(self):
        self.orchestrator = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        # Only initialize if not already done
        session_keys = ['initialized', 'query_history', 'current_response', 'analysis_data']
        for key in session_keys:
            if key not in st.session_state:
                if key == 'initialized':
                    st.session_state[key] = False
                elif key in ['query_history', 'analysis_data']:
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
    
    def check_system_requirements(self):
        """Check if all required components are available"""
        issues = []
        
        if not GENERATION_AVAILABLE:
            issues.append("❌ generation.py module not found")
        if not RETRIEVAL_AVAILABLE:
            issues.append("❌ retrieval module not found")
        if not COHERE_AVAILABLE:
            issues.append("❌ cohere package not installed")
        if not CONFIG_AVAILABLE:
            issues.append("⚠️ config.py not found (using defaults)")
        if not TAVILY_AVAILABLE:
            issues.append("⚠️ tavily package not installed (web search disabled)")
        
        return issues
    
    @st.cache_resource
    def load_system(_self):
        """Load and cache the RAG system components"""
        try:
            if not GENERATION_AVAILABLE or not RETRIEVAL_AVAILABLE or not COHERE_AVAILABLE:
                return None
            
            retriever = RETRIEVAL_CLASS()
            llm_client = CohereLLMClient()
            
            tavily_client = None
            if TAVILY_AVAILABLE and hasattr(Config, 'TAVILY_API_KEY') and Config.TAVILY_API_KEY:
                try:
                    tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY)
                except Exception:
                    pass
            
            orchestrator = SmartGenerationOrchestrator(
                retriever=retriever,
                llm_client=llm_client,
                tavily_client=tavily_client
            )
            
            return orchestrator
            
        except Exception as e:
            logger.error(f"System loading error: {e}")
            return None
    
    def render_header(self):
        """Render the application header"""
        st.markdown('<h1 class="main-header">🤖 Smart RAG System</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Intelligent document retrieval and generation with adaptive strategies
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_system_status(self):
        """Render system status and requirements"""
        issues = self.check_system_requirements()
        
        if not issues:
            return True
            
        st.markdown("### 🔧 System Status")
        
        for issue in issues:
            if "❌" in issue:
                st.error(issue)
            else:
                st.warning(issue)
        
        if any("❌" in issue for issue in issues):
            st.markdown("""
            ### 📋 Setup Instructions:
            
            1. **Make sure you have these files in your project:**
               - `generation.py` (with SmartGenerationOrchestrator class)
               - `retrieval.py` or `Retrieval.py` (with your retrieval class)
               - `config.py` (with API keys and settings)
            
            2. **Install required packages:**
               ```bash
               pip install cohere tavily-python streamlit plotly pandas
               ```
            
            3. **Check your imports in generation.py match your actual file structure**
            """)
            return False
        
        return True
    
    def render_sidebar(self):
        """Render the sidebar with configuration and strategy info"""
        with st.sidebar:
            st.header("⚙️ System Status")
            
            components = [
                ("Generation Module", GENERATION_AVAILABLE),
                ("Retrieval Module", RETRIEVAL_AVAILABLE),
                ("Cohere API", COHERE_AVAILABLE),
                ("Config Module", CONFIG_AVAILABLE),
                ("Tavily API", TAVILY_AVAILABLE),
            ]
            
            for name, available in components:
                if available:
                    st.success(f"✅ {name}")
                else:
                    st.error(f"❌ {name}")
            
            st.markdown("---")
            
            if self.orchestrator:
                st.header("🎯 Generation Strategies")
                strategies = {
                    "🔍 RAG Primary": "High relevance + non-temporal",
                    "🔗 RAG Augmented": "Medium relevance",
                    "🧠 Knowledge Primary": "Conceptual-only",
                    "🌐 Web Search": "Temporal-only",
                    "🔄 Hybrid": "High relevance + temporal",
                    "⚠️ Fallback": "Low relevance"
                }
                
                for strategy, description in strategies.items():
                    st.markdown(f"**{strategy}**")
                    st.caption(description)
                
                st.markdown("---")
                
                if hasattr(self.orchestrator, 'HIGH_RELEVANCE_THRESHOLD'):
                    st.subheader("📊 Thresholds")
                    st.metric("High Relevance", f"{self.orchestrator.HIGH_RELEVANCE_THRESHOLD:.2f}")
                    st.metric("Medium Relevance", f"{self.orchestrator.MEDIUM_RELEVANCE_THRESHOLD:.2f}")
    
    def render_query_interface(self):
        """Render the main query interface"""
        if not self.orchestrator:
            st.error("❌ System not available - please fix the issues above first")
            return "", False
        
        # Handle example query selection
        if 'selected_example' not in st.session_state:
            st.session_state.selected_example = ""
        
        st.subheader("💡 Example Queries")
        example_queries = [
            "What's the current status of carton P056186473?",
            "Show me documents created by user APYR5460",
            "What is document management in general?",
            "Latest news about document archiving today",
            "How do carton management systems work?"
        ]
        
        cols = st.columns(len(example_queries))
        for i, example in enumerate(example_queries):
            with cols[i]:
                if st.button(f"📝 {example[:20]}...", key=f"example_{i}", help=example):
                    st.session_state.selected_example = example
                    st.rerun()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Use the selected example as the default value
            default_value = st.session_state.selected_example if st.session_state.selected_example else ""
            query = st.text_input(
                "Enter your question:",
                value=default_value,
                placeholder="e.g., What's the current status of cartons with TAXES FONCIERES?",
                key="query_input"
            )
            # Clear the selected example after using it
            if st.session_state.selected_example:
                st.session_state.selected_example = ""
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.button("🚀 Ask", type="primary", use_container_width=True)
        
        return query, submit_button
    
    def process_query_safe(self, query: str) -> Dict[str, Any]:
        """Process query with error handling"""
        try:
            if hasattr(self.orchestrator, 'process_query'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.orchestrator.process_query(query))
                loop.close()
                return result
            elif hasattr(self.orchestrator, 'process_query_sync'):
                return self.orchestrator.process_query_sync(query)
            else:
                return {"answer": "Method not found", "strategy": "Error", "confidence": 0.0}
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "strategy": "Error",
                "confidence": 0.0,
                "sources": [],
                "reasoning": f"Error: {str(e)}",
                "query": query,
                "generation_context": {}
            }
    
    def render_response(self, response: Dict[str, Any]):
        """Render the response with detailed analysis"""
        if not response:
            return
        
        st.subheader("💬 Response")
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745;">
            {response.get('answer', 'No answer provided')}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy = response.get('strategy', 'Unknown')
            strategy_color = self.get_strategy_color(strategy)
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: {strategy_color};">🎯 Strategy</h4>
                <p style="font-size: 1.2rem; font-weight: bold;">{strategy}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence = response.get('confidence', 0.0)
            confidence_color = self.get_confidence_color(confidence)
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: {confidence_color};">📊 Confidence</h4>
                <p style="font-size: 1.2rem; font-weight: bold;">{confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            source_count = len(response.get('sources', []))
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: #1f77b4;">📚 Sources</h4>
                <p style="font-size: 1.2rem; font-weight: bold;">{source_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        gen_context = response.get('generation_context', {})
        if gen_context:
            self.render_detailed_analysis(response)
        
        if response.get('reasoning'):
            st.markdown("**🤔 Decision Reasoning:**")
            st.info(response.get('reasoning'))
        
        if response.get('sources'):
            self.render_sources(response['sources'])
    
    def render_detailed_analysis(self, response: Dict[str, Any]):
        """Render detailed analysis of the generation decision"""
        st.subheader("📈 Analysis Details")
        
        gen_context = response.get('generation_context', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            relevance_level = gen_context.get('relevance_level', 'unknown')
            st.metric("Relevance Level", relevance_level.title())
        
        with col2:
            max_score = gen_context.get('max_score', 0.0)
            st.metric("Max Score", f"{max_score:.3f}")
        
        with col3:
            temporal = gen_context.get('needs_current_info', False)
            st.metric("Temporal Need", "Yes" if temporal else "No")
        
        with col4:
            conceptual = gen_context.get('is_conceptual', False)
            st.metric("Conceptual", "Yes" if conceptual else "No")
    
    def render_sources(self, sources: List[str]):
        """Render source information"""
        st.subheader("📚 Sources")
        
        for i, source in enumerate(sources, 1):
            st.markdown(f"""
            <div class="source-item">
                <strong>Source {i}:</strong> {source}
            </div>
            """, unsafe_allow_html=True)
    
    def get_strategy_color(self, strategy: str) -> str:
        """Get color for strategy display"""
        colors = {
            'rag primary': '#28a745',
            'rag augmented': '#007bff',
            'knowledge primary': '#6f42c1',
            'web search': '#fd7e14',
            'hybrid': '#20c997',
            'fallback': '#6c757d',
            'error': '#dc3545'
        }
        
        strategy_lower = strategy.lower()
        for key, color in colors.items():
            if key in strategy_lower:
                return color
        return '#6c757d'
    
    def get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level"""
        if confidence >= 0.8:
            return '#28a745'
        elif confidence >= 0.6:
            return '#ffc107'
        else:
            return '#dc3545'
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        if not self.render_system_status():
            return
        
        if not st.session_state.initialized:
            with st.spinner("🔄 Loading RAG System..."):
                self.orchestrator = self.load_system()
                st.session_state.initialized = True
        else:
            self.orchestrator = self.load_system()
        
        self.render_sidebar()
        
        query, submit_button = self.render_query_interface()
        
        if not query and not submit_button:
            return
        
        if submit_button and query.strip():
            with st.spinner("🤔 Processing your query..."):
                response = self.process_query_safe(query.strip())
                st.session_state.current_response = response
                st.session_state.query_history.append(response)
        
        if st.session_state.current_response:
            self.render_response(st.session_state.current_response)
        
        if st.session_state.query_history:
            with st.expander("📜 Query History", expanded=False):
                for i, entry in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                    st.markdown(f"**Query {i}:** {entry.get('query', 'Unknown')}")
                    st.markdown(f"**Strategy:** {entry.get('strategy', 'Unknown')}")
                    st.markdown(f"**Confidence:** {entry.get('confidence', 0.0):.2%}")
                    st.markdown("---")

def main():
    """Main function to run the Streamlit app"""
    try:
        app = StreamlitRAGApp()
        app.run()
    except Exception as e:
        st.error(f"Error running app: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    try:
        # Ensure we're running in Streamlit context
        _ = st.session_state
        main()
    except Exception:
        print("Please run this app with: streamlit run app.py")