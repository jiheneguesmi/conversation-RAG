import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import re
# Import your retrieval components
from retrieval import RetrievalResult, HybridRetriever
from config import Config
import cohere
from tavily import TavilyClient


logger = logging.getLogger(__name__)

class GenerationStrategy(Enum):
    RAG_PRIMARY = "rag_primary"          # High relevance + non-temporal
    RAG_AUGMENTED = "rag_augmented"      # Medium relevance  
    KNOWLEDGE_PRIMARY = "knowledge_primary"  # Conceptual only
    WEB_SEARCH = "web_search"            # Temporal only
    HYBRID = "hybrid"                    # High relevance + temporal
    FALLBACK = "fallback"               # Otherwise

@dataclass
class CohereLLMClient:
    def __init__(self):
        self.client = cohere.Client(Config.COHERE_API_KEY)
        self.model = Config.CHAT_MODEL

    async def generate(self, prompt: str) -> str:
        response = self.client.chat(
            message=prompt,
            model=self.model,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_ANSWER_LENGTH
        )
        return response.text

@dataclass
class GenerationContext:
    """Context for generation decision making"""
    query: str
    retrieval_results: List[Any]
    max_score: float
    avg_score: float
    result_count: int
    needs_current_info: bool
    is_conceptual: bool
    is_specific_factual: bool
    content_relevance_score: float
    query_complexity: str
    entity_count: int
    strategy: GenerationStrategy
    confidence: float
    reasoning: str
    relevance_level: str  # NEW: "high", "medium", "low"
    supplementary_info: Dict[str, Any] = field(default_factory=dict)

class SmartGenerationOrchestrator:
    """Orchestrates generation strategy selection and execution"""
    
    def __init__(self, retriever, llm_client, tavily_client=None):
        self.retriever = retriever
        self.llm_client = llm_client
        self.tavily_client = tavily_client
        
        # Updated thresholds based on your strategy
        self.HIGH_RELEVANCE_THRESHOLD = 0.75    # High relevance
        self.MEDIUM_RELEVANCE_THRESHOLD = 0.45  # Medium relevance
        self.LOW_RELEVANCE_THRESHOLD = 0.2      # Low relevance
        
        # Temporal keywords for current info detection
        self.temporal_keywords = [
            "today", "latest", "current", "recent", "now", "this year",
            "2024", "2025", "breaking", "news", "update", "status",
            "live", "ongoing", "happening", "fresh", "new", "real-time"
        ]
        
        # Current info domains
        self.current_info_domains = [
            "weather", "news", "stock", "price", "status", "availability",
            "politics", "sports", "events", "trending", "viral", "market"
        ]
        
        # Conceptual query patterns
        self.conceptual_patterns = [
            r"what is\s+",
            r"how does\s+.*\s+work",
            r"why is\s+.*\s+important",
            r"explain\s+",
            r"define\s+",
            r"what are the principles of",
            r"how to\s+.*\s+in general",
            r"what causes\s+",
            r"relationship between\s+",
            r"implications of\s+",
            r"concept of\s+",
            r"theory of\s+"
        ]
        
        # General knowledge domains
        self.general_knowledge_domains = [
            "science", "mathematics", "physics", "chemistry", "biology",
            "history", "philosophy", "psychology", "sociology", "economics",
            "literature", "art", "music", "culture", "language", "education",
            "theory", "principle", "concept"
        ]

    def analyze_generation_context(self, query: str, retrieval_results: List[Any]) -> GenerationContext:
        """Comprehensive analysis to determine generation strategy"""
        
        # Basic metrics from RetrievalResult objects
        max_score = max(r.score for r in retrieval_results) if retrieval_results else 0.0
        avg_score = sum(r.score for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0.0
        result_count = len([r for r in retrieval_results if r.score > self.LOW_RELEVANCE_THRESHOLD])
        
        # Analysis methods
        needs_current_info = self._detect_temporal_requirements(query)
        is_conceptual = self._detect_conceptual_query(query)
        is_specific_factual = self._detect_specific_factual_query(query)
        content_relevance_score = self._assess_content_relevance(query, retrieval_results)
        
        # Determine relevance level
        relevance_level = self._determine_relevance_level(max_score, content_relevance_score, result_count)
        
        # Query complexity analysis
        query_complexity = "simple"
        entity_count = 0
        
        if hasattr(self.retriever, 'query_analyzer'):
            try:
                query_analysis = self.retriever.query_analyzer.analyze_query(query)
                query_complexity = query_analysis.query_type
                entity_count = len(query_analysis.entities) if hasattr(query_analysis, 'entities') else 0
            except Exception as e:
                logger.warning(f"Query analysis failed: {e}")
        
        # Strategy selection using your framework
        strategy, confidence, reasoning = self._select_strategy_by_framework(
            relevance_level, needs_current_info, is_conceptual, query,
            max_score, content_relevance_score, result_count
        )
        
        return GenerationContext(
            query=query,
            retrieval_results=retrieval_results,
            max_score=max_score,
            avg_score=avg_score,
            result_count=result_count,
            needs_current_info=needs_current_info,
            is_conceptual=is_conceptual,
            is_specific_factual=is_specific_factual,
            content_relevance_score=content_relevance_score,
            query_complexity=query_complexity,
            entity_count=entity_count,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            relevance_level=relevance_level
        )

    def _determine_relevance_level(self, max_score: float, content_relevance_score: float, result_count: int) -> str:
        """Determine relevance level: high, medium, or low"""
        
        # High relevance: high scores AND good content relevance AND multiple results
        if (max_score >= self.HIGH_RELEVANCE_THRESHOLD and 
            content_relevance_score >= 0.4 and 
            result_count >= 2):
            return "high"
        
        # Medium relevance: moderate scores OR decent content relevance
        elif (max_score >= self.MEDIUM_RELEVANCE_THRESHOLD or 
              content_relevance_score >= 0.25 or
              result_count >= 1):
            return "medium"
        
        # Low relevance: everything else
        else:
            return "low"

    def _select_strategy_by_framework(self, relevance_level: str, needs_current_info: bool, 
                                    is_conceptual: bool, query: str, max_score: float,
                                    content_relevance_score: float, result_count: int) -> Tuple[GenerationStrategy, float, str]:
        """
        Select strategy based on your framework:
        - High relevance + temporal → Hybrid
        - High relevance + non-temporal → RAG Primary  
        - Medium relevance → RAG Augmented
        - Temporal only → Web Search
        - Conceptual only → Knowledge Primary
        - Otherwise → Fallback
        """
        
        # HIGH RELEVANCE + TEMPORAL → HYBRID
        if relevance_level == "high" or relevance_level == "medium" and needs_current_info:
            return (
                GenerationStrategy.HYBRID,
                0.9,
                f"High relevance ({max_score:.2f}) + temporal query → Hybrid approach"
            )
        
        # HIGH RELEVANCE + NON-TEMPORAL → RAG PRIMARY
        elif relevance_level == "high" and not needs_current_info:
            return (
                GenerationStrategy.RAG_PRIMARY,
                0.95,
                f"High relevance ({max_score:.2f}) + non-temporal → RAG Primary"
            )
        
        # MEDIUM RELEVANCE → RAG AUGMENTED
        elif relevance_level == "medium":
            return (
                GenerationStrategy.RAG_AUGMENTED,
                0.8,
                f"Medium relevance (score: {max_score:.2f}, content: {content_relevance_score:.2f}) → RAG Augmented"
            )
        
        # TEMPORAL ONLY → WEB SEARCH
        elif needs_current_info and not is_conceptual and relevance_level == "low":
            return (
                GenerationStrategy.WEB_SEARCH,
                0.75,
                "Temporal query with low relevance → Web Search"
            )
        
        # CONCEPTUAL ONLY → KNOWLEDGE PRIMARY
        elif is_conceptual and not needs_current_info and relevance_level == "low":
            return (
                GenerationStrategy.KNOWLEDGE_PRIMARY,
                0.8,
                "Conceptual query with low relevance → Knowledge Primary"
            )
        
        # Edge case: High temporal need but also conceptual
        elif needs_current_info and is_conceptual:
            if relevance_level in ["high", "medium"]:
                return (
                    GenerationStrategy.HYBRID,
                    0.85,
                    "Temporal + conceptual with some relevance → Hybrid"
                )
            else:
                return (
                    GenerationStrategy.WEB_SEARCH,
                    0.7,
                    "Temporal + conceptual with low relevance → Web Search"
                )
        
        # OTHERWISE → FALLBACK
        else:
            return (
                GenerationStrategy.FALLBACK,
                0.4,
                f"No clear strategy match (relevance: {relevance_level}, temporal: {needs_current_info}, conceptual: {is_conceptual}) → Fallback"
            )

    def _detect_temporal_requirements(self, query: str) -> bool:
        """Detect if query needs current/real-time information"""
        query_lower = query.lower()
        
        # Check for temporal keywords
        has_temporal_keywords = any(keyword in query_lower for keyword in self.temporal_keywords)
        
        # Check for current info domains
        has_current_domains = any(domain in query_lower for domain in self.current_info_domains)
        
        # Check for temporal question patterns
        temporal_patterns = [
            r"what'?s happening",
            r"latest news",
            r"current status",
            r"recent developments",
            r"up to date",
            r"as of (today|now)",
            r"right now",
            r"this (week|month|year)",
        ]
        has_temporal_patterns = any(re.search(pattern, query_lower) for pattern in temporal_patterns)
        
        return has_temporal_keywords or has_current_domains or has_temporal_patterns

    def _detect_conceptual_query(self, query: str) -> bool:
        """Detect if query is asking for conceptual/general knowledge"""
        query_lower = query.lower()
        
        # Check for conceptual patterns
        has_conceptual_patterns = any(re.search(pattern, query_lower) for pattern in self.conceptual_patterns)
        
        # Check for general knowledge domains
        has_general_domains = any(domain in query_lower for domain in self.general_knowledge_domains)
        
        # Check for abstract reasoning requirements
        abstract_indicators = [
            "principle", "theory", "concept", "idea", "philosophy",
            "understanding", "meaning", "significance", "importance",
            "implication", "consequence", "effect", "cause", "mechanism"
        ]
        has_abstract_indicators = any(indicator in query_lower for indicator in abstract_indicators)
        
        return has_conceptual_patterns or has_general_domains or has_abstract_indicators

    def _detect_specific_factual_query(self, query: str) -> bool:
        """Detect if query is asking for specific factual information"""
        query_lower = query.lower()
        
        # Factual question patterns
        factual_patterns = [
            r"who is\s+",
            r"when did\s+",
            r"where is\s+",
            r"how many\s+",
            r"what happened\s+",
            r"which\s+.*\s+has",
            r"name of\s+",
            r"list of\s+",
            r"details about\s+",
            r"information about\s+"
        ]
        
        has_factual_patterns = any(re.search(pattern, query_lower) for pattern in factual_patterns)
        
        # Specific domains that usually require factual answers
        factual_domains = [
            "company", "person", "organization", "product", "service",
            "location", "address", "contact", "specification", "feature"
        ]
        has_factual_domains = any(domain in query_lower for domain in factual_domains)
        
        return has_factual_patterns or has_factual_domains

    def _assess_content_relevance(self, query: str, retrieval_results: List[Any]) -> float:
        """Assess content relevance beyond similarity scores"""
        if not retrieval_results:
            return 0.0
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        total_relevance = 0.0
        
        for result in retrieval_results:
            content = getattr(result, 'content', '') or getattr(result, 'formatted_content', '')
            content_lower = content.lower()
            content_words = set(content_lower.split())
            
            # Calculate lexical overlap
            word_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
            
            # Check for entity mentions
            query_entities = [word for word in query.split() if word[0].isupper()]
            entity_mentions = sum(1 for entity in query_entities if entity.lower() in content_lower)
            entity_score = entity_mentions / len(query_entities) if query_entities else 0
            
            # Combine scores (weighted toward lexical overlap)
            result_relevance = (word_overlap * 0.7) + (entity_score * 0.3)
            total_relevance += result_relevance
        
        return total_relevance / len(retrieval_results)

    # Generation methods remain largely the same, but with updated logging
    async def _generate_rag_primary(self, context: GenerationContext) -> Dict[str, Any]:
        """Generate using primarily RAG results - for high relevance + non-temporal"""
        
        top_results = sorted(context.retrieval_results, key=lambda x: x.score, reverse=True)[:3]
        context_text = self._format_rag_context(top_results)
        
        prompt = f"""Based on the following retrieved information, please answer the user's question: "{context.query}"

Retrieved Information:
{context_text}

Please provide a comprehensive answer based primarily on this retrieved information. The information has high relevance to the query."""

        response = await self._call_llm(prompt, context)
        
        return {
            "answer": response,
            "strategy": "RAG Primary (High Relevance + Non-Temporal)",
            "confidence": context.confidence,
            "sources": [self._get_source_info(r) for r in top_results],
            "source_count": len(top_results),
            "reasoning": context.reasoning,
            "relevance_level": context.relevance_level,
            "retrieval_methods": [r.retrieval_method for r in top_results if hasattr(r, 'retrieval_method')]
        }

    async def _generate_rag_augmented(self, context: GenerationContext) -> Dict[str, Any]:
        """Generate using RAG + knowledge augmentation - for medium relevance"""
        
        top_results = sorted(context.retrieval_results, key=lambda x: x.score, reverse=True)[:3]
        context_text = self._format_rag_context(top_results)
        
        prompt = f"""Answer the user's question: "{context.query}"

Retrieved Information (Medium Relevance):
{context_text}

Use the retrieved information where relevant, but supplement with your general knowledge to provide a complete answer. The retrieved information has medium relevance, so combine it thoughtfully with your knowledge."""

        response = await self._call_llm(prompt, context)
        
        return {
            "answer": response,
            "strategy": "RAG Augmented (Medium Relevance)",
            "confidence": context.confidence,
            "sources": [self._get_source_info(r) for r in top_results],
            "reasoning": context.reasoning,
            "relevance_level": context.relevance_level,
            "retrieval_methods": [r.retrieval_method for r in top_results if hasattr(r, 'retrieval_method')]
        }

    async def _generate_knowledge_primary(self, context: GenerationContext) -> Dict[str, Any]:
        """Generate using primarily pretrained knowledge - for conceptual only"""
        
        prompt = f"""Answer the following conceptual question using your general knowledge: "{context.query}"

Provide a comprehensive explanation based on established knowledge and principles. Focus on clear explanations of concepts, theories, and general understanding."""

        response = await self._call_llm(prompt, context)
        
        return {
            "answer": response,
            "strategy": "Knowledge Primary (Conceptual Only)",
            "confidence": context.confidence,
            "sources": [],
            "reasoning": context.reasoning,
            "relevance_level": context.relevance_level
        }

    async def _generate_web_search(self, context: GenerationContext) -> Dict[str, Any]:
        """Generate using web search results - for temporal only"""
        if not self.tavily_client:
            self.tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY)
        
        if not self.tavily_client:
            logger.warning("Tavily client not available, falling back to knowledge")
            return await self._generate_knowledge_primary(context)
        
        try:
            # Perform web search
            search_results = self.tavily_client.search(
                query=context.query,
                max_results=5,
                include_domains=None,
                exclude_domains=None
            )
            
            # Format search results
            web_context = self._format_web_context(search_results.get('results', []))
            
            prompt = f"""Based on the following current web search results, answer the user's temporal question: "{context.query}"

Current Web Information:
{web_context}

Provide an up-to-date answer based on the search results. Focus on the most recent and relevant information."""

            response = await self._call_llm(prompt, context)
            
            return {
                "answer": response,
                "strategy": "Web Search (Temporal Only)",
                "confidence": context.confidence,
                "sources": [result.get('url') for result in search_results.get('results', [])],
                "reasoning": context.reasoning,
                "relevance_level": context.relevance_level,
                "search_results": search_results
            }
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return await self._generate_knowledge_primary(context)

    async def _generate_hybrid(self, context: GenerationContext) -> Dict[str, Any]:
        """Generate using hybrid approach - for high relevance + temporal"""
        
        # Get RAG context
        top_results = sorted(context.retrieval_results, key=lambda x: x.score, reverse=True)[:2]
        rag_context = self._format_rag_context(top_results) if top_results else ""
        
        # Get web search results
        web_context = ""
        web_sources = []
        
        if not self.tavily_client:
            self.tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY)
        
        if self.tavily_client:
            try:
                search_results = self.tavily_client.search(
                    query=context.query,
                    max_results=3
                )
                web_context = self._format_web_context(search_results.get('results', []))
                web_sources = [result.get('url') for result in search_results.get('results', [])]
            except Exception as e:
                logger.error(f"Web search in hybrid mode failed: {e}")
        
        # Combine contexts
        combined_prompt = f"""Answer the user's question: "{context.query}"

This question has high relevance to retrieved information AND needs current information.

Retrieved High-Relevance Information:
{rag_context}

Current Web Information:
{web_context}

Provide a comprehensive answer that combines both the relevant retrieved information and current web information. Prioritize the most accurate and up-to-date details."""

        response = await self._call_llm(combined_prompt, context)
        
        return {
            "answer": response,
            "strategy": "Hybrid (High Relevance + Temporal)",
            "confidence": context.confidence,
            "sources": [self._get_source_info(r) for r in top_results] + web_sources,
            "reasoning": context.reasoning,
            "relevance_level": context.relevance_level,
            "retrieval_methods": [r.retrieval_method for r in top_results if hasattr(r, 'retrieval_method')]
        }

    async def _generate_fallback(self, context: GenerationContext) -> Dict[str, Any]:
        """Fallback generation - otherwise"""
        
        prompt = f"""I need to provide information about: "{context.query}"

While I don't have highly relevant specific information available, I'll provide what general information I can based on my knowledge. Please note this may not be the most specific or current information for your particular context."""

        response = await self._call_llm(prompt, context)
        
        return {
            "answer": response,
            "strategy": "Fallback (Otherwise)",
            "confidence": 0.4,
            "sources": [],
            "reasoning": context.reasoning,
            "relevance_level": context.relevance_level
        }

    # Utility methods remain the same
    def _format_rag_context(self, results: List[Any]) -> str:
        """Format RAG results for context"""
        formatted_parts = []
        
        for i, result in enumerate(results, 1):
            content = getattr(result, 'formatted_content', None) or getattr(result, 'content', '')
            source_info = self._get_source_info(result)
            
            formatted_parts.append(f"""
Source {i} (from {source_info}, relevance: {result.score:.3f}):
{content}

Retrieval method: {getattr(result, 'retrieval_method', 'Unknown')}
""")
        
        return "\n".join(formatted_parts)

    def _get_source_info(self, result) -> str:
        """Extract source information from RetrievalResult"""
        source = (
            getattr(result, 'source_table', None) or 
            getattr(result, 'source', None) or 
            getattr(result, 'table_name', None) or 
            'Unknown Source'
        )
        return str(source)

    def _format_web_context(self, search_results: List[Dict]) -> str:
        """Format web search results for context"""
        formatted_parts = []
        
        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'No title')
            content = result.get('content', result.get('snippet', 'No content'))
            url = result.get('url', 'No URL')
            
            formatted_parts.append(f"""
Web Result {i} - {title}:
{content}
Source: {url}
""")
        
        return "\n".join(formatted_parts)

    async def _call_llm(self, prompt: str, context: GenerationContext) -> str:
        """Call your LLM with the given prompt"""
        try:
            if hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat(
                    model="command-r-plus",
                    message=prompt,
                    temperature=0.1,
                    max_tokens=1000
                )
                return response.text
            else:
                response = await self.llm_client.generate(prompt)
                return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"I apologize, but I'm unable to generate a response at the moment due to a technical issue: {str(e)}"

    async def process_query(self, query: str, retrieval_results: List[Any] = None) -> Dict[str, Any]:
        """Main method to process a query end-to-end"""
        
        try:
            # Get retrieval results if not provided
            if retrieval_results is None:
                retrieval_results = self.retriever.retrieve(query, top_k=5)
            
            # Analyze context
            context = self.analyze_generation_context(query, retrieval_results)
            
            # Enhanced logging with your strategy framework
            logger.info(f"=== STRATEGY SELECTION ===")
            logger.info(f"Query: {query}")
            logger.info(f"Relevance Level: {context.relevance_level}")
            logger.info(f"Temporal Need: {context.needs_current_info}")
            logger.info(f"Conceptual: {context.is_conceptual}")
            logger.info(f"Selected Strategy: {context.strategy.value}")
            logger.info(f"Confidence: {context.confidence:.2f}")
            logger.info(f"Reasoning: {context.reasoning}")
            logger.info(f"Max Score: {context.max_score:.3f}, Content Relevance: {context.content_relevance_score:.3f}")
            
            # Generate response
            response = await self.generate_response(context)
            
            # Add metadata
            response.update({
                "query": query,
                "generation_context": {
                    "relevance_level": context.relevance_level,
                    "max_score": context.max_score,
                    "avg_score": context.avg_score,
                    "result_count": context.result_count,
                    "needs_current_info": context.needs_current_info,
                    "is_conceptual": context.is_conceptual,
                    "content_relevance_score": context.content_relevance_score,
                    "query_complexity": context.query_complexity,
                    "total_retrieval_results": len(context.retrieval_results)
                },
                "timestamp": datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "strategy": "Error",
                "confidence": 0.0,
                "sources": [],
                "reasoning": f"Error occurred: {str(e)}",
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

    def process_query_sync(self, query: str, retrieval_results: List[Any] = None) -> Dict[str, Any]:
        """Synchronous version of process_query for easier integration"""
        return asyncio.run(self.process_query(query, retrieval_results))

    async def generate_response(self, context: GenerationContext) -> Dict[str, Any]:
        """Generate response based on selected strategy"""
        
        try:
            if context.strategy == GenerationStrategy.RAG_PRIMARY:
                return await self._generate_rag_primary(context)
            
            elif context.strategy == GenerationStrategy.RAG_AUGMENTED:
                return await self._generate_rag_augmented(context)
            
            elif context.strategy == GenerationStrategy.KNOWLEDGE_PRIMARY:
                return await self._generate_knowledge_primary(context)
            
            elif context.strategy == GenerationStrategy.WEB_SEARCH:
                return await self._generate_web_search(context)
            
            elif context.strategy == GenerationStrategy.HYBRID:
                return await self._generate_hybrid(context)
            
            else:  # FALLBACK
                return await self._generate_fallback(context)
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return await self._generate_fallback(context)

def test_generation_orchestrator():
    """Test the updated strategy framework"""
    async def _run():
        retriever = HybridRetriever()
        llm_client = CohereLLMClient()
        orchestrator = SmartGenerationOrchestrator(retriever, llm_client)

        print("Updated Strategy Framework Test")
        print("=" * 50)
        print("Strategy Rules:")
        print("• High relevance + temporal → Hybrid")
        print("• High relevance + non-temporal → RAG Primary")
        print("• Medium relevance → RAG Augmented")
        print("• Temporal only → Web Search")
        print("• Conceptual only → Knowledge Primary")
        print("• Otherwise → Fallback")
        print("=" * 50)
        
        while True:
            query = input("\nEnter your test query (or 'exit'): ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue
            
            result = await orchestrator.process_query(query)

            print(f"\n ANALYSIS:")
            print(f"   Relevance Level: {result['generation_context']['relevance_level']}")
            print(f"   Temporal Need: {result['generation_context']['needs_current_info']}")
            print(f"   Conceptual: {result['generation_context']['is_conceptual']}")
            print(f"   Max Score: {result['generation_context']['max_score']:.3f}")
            print(f"   Content Relevance: {result['generation_context']['content_relevance_score']:.3f}")
            
            print(f"\n STRATEGY: {result['strategy']}")
            print(f" CONFIDENCE: {result['confidence']:.2f}")
            print(f" REASONING: {result['reasoning']}")
            
            if result.get('sources'):
                print(f"📚 SOURCES: {len(result['sources'])} sources")
            
            print(f"\n ANSWER:\n{result['answer']}")

    asyncio.run(_run())

if __name__ == "__main__":
    test_generation_orchestrator()