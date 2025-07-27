import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from langchain.docstore.document import Document
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured result from retrieval system"""

    content: str
    metadata: Dict[str, Any]
    score: float
    source_table: str
    retrieval_method: str
    relevance_explanation: str = ""
    formatted_content: str = ""  # Add formatted content for better display

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary with parsed columns"""
        if " | " not in self.content:
            return {
                "content": self.content,
                "source": self.source_table,
                "score": self.score,
            }

        row_dict = {}
        parts = self.content.split(" | ")
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                row_dict[key.strip()] = value.strip()

        row_dict["_source"] = self.source_table
        row_dict["_score"] = self.score
        row_dict["_method"] = self.retrieval_method

        return row_dict


@dataclass
class QueryAnalysis:
    """Analysis of user query for optimal retrieval strategy"""

    original_query: str
    query_type: str
    entities: List[str]
    keywords: List[str]
    potential_columns: List[str]
    confidence: float


class QueryAnalyzer:
    """Analyzes queries to determine optimal retrieval strategy"""

    def __init__(self):
        # Enhanced column patterns for business/organizational data
        self.column_patterns = {
            "identity": [
                "id",
                "name",
                "title",
                "nom",
                "prenom",
                "firstname",
                "lastname",
                "codebarre",
                "barcode",
            ],
            "contact": ["email", "phone", "telephone", "address", "adresse"],
            "temporal": [
                "date",
                "time",
                "year",
                "month",
                "created",
                "updated",
                "timestamp",
                "2011",
                "2012",
            ],
            "financial": [
                "price",
                "cost",
                "amount",
                "salary",
                "budget",
                "revenue",
                "tax",
                "taxes",
            ],
            "location": [
                "city",
                "country",
                "region",
                "location",
                "ville",
                "pays",
                "emplacement",
            ],
            "status": ["status", "state", "active", "enabled", "type", "category"],
            "description": [
                "description",
                "comment",
                "notes",
                "details",
                "summary",
                "carton",
                "document",
            ],
            "document_types": [
                "pdf",
                "doc",
                "excel",
                "foncier",
                "property",
                "carton",
                "dossier",
            ],
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine retrieval strategy"""
        query_lower = query.lower()

        # Extract potential entities
        entities = self._extract_entities(query)

        # Extract keywords
        keywords = self._extract_keywords(query)

        # Determine query type
        query_type = self._classify_query_type(query_lower)

        # Map to potential columns
        potential_columns = self._map_to_columns(query_lower, keywords)

        # Calculate confidence
        confidence = self._calculate_confidence(entities, keywords, potential_columns)

        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            entities=entities,
            keywords=keywords,
            potential_columns=potential_columns,
            confidence=confidence,
        )

    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query"""
        entities = []

        # Email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        entities.extend(re.findall(email_pattern, query))

        # Barcodes and IDs (alphanumeric codes)
        barcode_pattern = r"\b[A-Z0-9]{6,}\b"
        entities.extend(re.findall(barcode_pattern, query))

        # Phone numbers
        phone_pattern = (
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
        )
        entities.extend(re.findall(phone_pattern, query))

        # Years (4-digit numbers)
        year_pattern = r"\b(19|20)\d{2}\b"
        entities.extend(re.findall(year_pattern, query))

        # Other numbers (potentially IDs, amounts)
        number_pattern = r"\b\d{3,}\b"  # At least 3 digits
        entities.extend(re.findall(number_pattern, query))

        # Names (capitalized words, but filter common words)
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", query)
        common_words = {
            "Show",
            "Find",
            "List",
            "Get",
            "From",
            "For",
            "All",
            "Tax",
            "Document",
            "File",
        }
        entities.extend([word for word in capitalized if word not in common_words])

        # Quoted strings
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)

        return list(set(entities))

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "what",
            "where",
            "when",
            "why",
            "how",
            "who",
            "which",
            "all",
            "any",
            "some",
            "more",
            "most",
        }

        # Split and clean words
        words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 1]

        return keywords

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        if any(
            indicator in query
            for indicator in ["find", "search", "look for", "get", "who is", "what is"]
        ):
            if any(comp in query for comp in ["compare", "vs", "versus", "difference"]):
                return "comparison"
            return "specific"

        if any(
            agg in query
            for agg in [
                "count",
                "sum",
                "total",
                "average",
                "how many",
                "list all",
                "show all",
            ]
        ):
            return "aggregation"

        if any(
            comp in query
            for comp in ["compare", "vs", "versus", "better", "worse", "different"]
        ):
            return "comparison"

        return "general"

    def _map_to_columns(self, query: str, keywords: List[str]) -> List[str]:
        """Map query terms to likely column names"""
        potential_columns = []

        for category, columns in self.column_patterns.items():
            for column in columns:
                if column in query or any(
                    keyword == column or column in keyword or keyword in column
                    for keyword in keywords
                ):
                    potential_columns.append(column)

        return potential_columns

    def _calculate_confidence(
        self, entities: List[str], keywords: List[str], potential_columns: List[str]
    ) -> float:
        """Calculate confidence in query analysis"""
        base_score = 0.2

        # More entities = higher confidence
        entity_score = min(0.4, len(entities) * 0.1)

        # More keywords = higher confidence
        keyword_score = min(0.2, len(keywords) * 0.05)

        # Column matches = higher confidence
        column_score = min(0.2, len(potential_columns) * 0.08)

        return min(1.0, base_score + entity_score + keyword_score + column_score)


class HybridRetriever:
    """Advanced retrieval system combining semantic and structured search"""

    def __init__(
        self, vector_store_path: str = None, embeddings_model: CohereEmbeddings = None
    ):
        self.vector_store_path = vector_store_path or Config.VECTOR_DB_PATH
        self.embeddings = embeddings_model or CohereEmbeddings(
            model=Config.EMBEDDING_MODEL, cohere_api_key=Config.COHERE_API_KEY
        )
        self.vector_store = None
        self.query_analyzer = QueryAnalyzer()

        self._load_vector_store()
        self.table_schemas = {}
        self._analyze_table_schemas()

    def _load_vector_store(self):
        """Load the FAISS vector store"""
        try:
            if os.path.exists(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"Vector store loaded from {self.vector_store_path}")
            else:
                logger.error(f"Vector store not found at {self.vector_store_path}")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")

    def _analyze_table_schemas(self):
        """Analyze schemas of all tables in the vector store"""
        if not self.vector_store:
            return

        try:
            # Get more documents for better schema analysis
            all_docs = self.vector_store.similarity_search("", k=2000)

            tables = {}
            for doc in all_docs:
                source = doc.metadata.get("source", "unknown")
                if source not in tables:
                    tables[source] = {
                        "columns": set(),
                        "sample_content": [],
                        "column_types": {},
                        "sample_values": {},
                    }

                # Enhanced column extraction
                content = doc.page_content
                if " | " in content:
                    parts = content.split(" | ")
                    for part in parts:
                        if ":" in part:
                            column_name = part.split(":")[0].strip()
                            column_value = part.split(":", 1)[1].strip()

                            tables[source]["columns"].add(column_name)

                            # Store sample values for each column
                            if column_name not in tables[source]["sample_values"]:
                                tables[source]["sample_values"][column_name] = set()
                            tables[source]["sample_values"][column_name].add(
                                column_value[:100]
                            )

                # Store more sample content
                if len(tables[source]["sample_content"]) < 10:
                    tables[source]["sample_content"].append(content[:500])

            # Convert sets to lists
            for table in tables:
                tables[table]["columns"] = list(tables[table]["columns"])
                for col in tables[table]["sample_values"]:
                    tables[table]["sample_values"][col] = list(
                        tables[table]["sample_values"][col]
                    )[:5]

            self.table_schemas = tables
            logger.info(f"Analyzed schemas for {len(tables)} tables")

        except Exception as e:
            logger.error(f"Failed to analyze table schemas: {e}")

    def retrieve_all_columns(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve results showing ALL available columns"""
        return self.retrieve(
            query, top_k, retrieval_strategy="auto", display_mode="full"
        )

    def _format_result_content(
        self, content: str, query_analysis: QueryAnalysis, display_mode: str = "smart"
    ) -> str:
        """Format result content to show most relevant information"""
        if " | " not in content:
            return content

        parts = content.split(" | ")
        row_dict = {}

        # Parse content into dictionary
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                row_dict[key.strip()] = value.strip()

        # Show ALL columns when display_mode is "full"
        if display_mode == "full":
            formatted_parts = []
            for key, value in row_dict.items():
                formatted_parts.append(f"{key}: {value}")
            return " | ".join(formatted_parts)

        # Original smart formatting logic for other modes
        always_show = [
            "id",
            "name",
            "nom",
            "prenom",
            "firstname",
            "lastname",
            "codebarre",
            "barcode",
            "title",
        ]
        query_relevant = []
        other_important = []

        query_lower = query_analysis.original_query.lower()

        for key, value in row_dict.items():
            key_lower = key.lower()
            value_lower = value.lower()

            # Always show identity columns first
            if key_lower in [col.lower() for col in always_show]:
                continue

            # Skip metadata columns unless specifically requested
            if (
                key_lower in ["filename", "insert_date", "update_date"]
                and key_lower not in query_lower
            ):
                continue

            # Check if column is query-relevant
            is_relevant = False
            if key_lower in query_lower:
                is_relevant = True
            else:
                # Check if value contains any query keywords or entities
                for entity in query_analysis.entities:
                    if entity.lower() in value_lower:
                        is_relevant = True
                        break
                if not is_relevant:
                    for keyword in query_analysis.keywords:
                        if keyword in key_lower or keyword in value_lower:
                            is_relevant = True
                            break

            if is_relevant:
                query_relevant.append(key)
            else:
                other_important.append(key)

        # Select columns to display (max 12 for smart mode)
        display_columns = []

        # Add identity columns first
        for key in row_dict.keys():
            if key.lower() in [col.lower() for col in always_show]:
                display_columns.append(key)

        # Add query-relevant columns
        display_columns.extend(query_relevant)

        # Fill remaining slots with other important columns
        remaining = 12 - len(display_columns)
        display_columns.extend(other_important[:remaining])

        # Format selected columns
        formatted_parts = []
        for col in display_columns:
            if col in row_dict:
                value = row_dict[col]
                formatted_parts.append(f"{col}: {value}")

        return " | ".join(formatted_parts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_strategy: str = "auto",
        display_mode: str = "smart",
    ) -> List[RetrievalResult]:
        """Main retrieval function with multiple strategies"""

        if not self.vector_store:
            logger.error("Vector store not available")
            return []

        query_analysis = self.query_analyzer.analyze_query(query)
        logger.info(
            f"Query analysis: {query_analysis.query_type}, confidence: {query_analysis.confidence:.2f}"
        )

        if retrieval_strategy == "auto":
            retrieval_strategy = self._choose_strategy(query_analysis)

        results = []

        try:
            if retrieval_strategy == "semantic":
                results = self._semantic_search(query, query_analysis, top_k)
            elif retrieval_strategy == "structured":
                results = self._structured_search(query, query_analysis, top_k)
            elif retrieval_strategy == "hybrid":
                results = self._hybrid_search(query, query_analysis, top_k)
            else:
                results = self._hybrid_search(query, query_analysis, top_k)

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            try:
                results = self._semantic_search(query, query_analysis, top_k)
            except Exception as e2:
                logger.error(f"Fallback retrieval also failed: {e2}")
                return []

        # Post-process results to remove duplicates and improve formatting
        return self._post_process_results(results, query_analysis, top_k, display_mode)

    def _post_process_results(
        self,
        results: List[RetrievalResult],
        query_analysis: QueryAnalysis,
        top_k: int,
        display_mode: str = "smart",
    ) -> List[RetrievalResult]:
        """Post-process results to remove duplicates and improve display"""

        # Remove duplicates based on content similarity
        unique_results = []
        seen_content = set()

        for result in results:
            # Create a hash of the core content (without metadata prefixes)
            core_content = result.content
            if " | " in core_content:
                # Extract just the data part, skip filename and metadata
                content_parts = core_content.split(" | ")
                data_parts = [
                    part
                    for part in content_parts
                    if not part.startswith(
                        ("filename:", "insert_date:", "update_date:")
                    )
                ]
                core_content = " | ".join(data_parts)

            content_hash = hash(
                core_content[:200]
            )  # Use first 200 chars for similarity

            if content_hash not in seen_content:
                seen_content.add(content_hash)

                # Format the content for better display
                result.formatted_content = self._format_result_content(
                    result.content, query_analysis, display_mode
                )
                unique_results.append(result)

        # Re-sort by score and limit to top_k
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:top_k]

    def _choose_strategy(self, query_analysis: QueryAnalysis) -> str:
        """Choose optimal retrieval strategy"""
        # If we have specific entities (names, IDs, barcodes), use structured
        if query_analysis.confidence > 0.6 and len(query_analysis.entities) > 0:
            return "structured"

        # If medium confidence, use hybrid
        if query_analysis.confidence > 0.4:
            return "hybrid"

        return "semantic"

    def _semantic_search(
        self, query: str, query_analysis: QueryAnalysis, top_k: int
    ) -> List[RetrievalResult]:
        """Enhanced semantic similarity search"""
        try:
            # Get more results initially for better filtering
            initial_k = min(top_k * 5, 50)
            docs = self.vector_store.similarity_search_with_score(query, k=initial_k)

            results = []
            for doc, distance_score in docs:
                # Convert distance to similarity (higher is better)
                similarity_score = max(0, 1 - distance_score)

                # Apply minimum threshold
                if similarity_score > 0.1:  # Only include reasonably relevant results
                    result = RetrievalResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=float(similarity_score),
                        source_table=doc.metadata.get("source", "unknown"),
                        retrieval_method="semantic",
                        relevance_explanation="Semantic similarity match",
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _structured_search(
        self, query: str, query_analysis: QueryAnalysis, top_k: int
    ) -> List[RetrievalResult]:
        """Enhanced structured search with better scoring"""
        try:
            # Get more candidates for filtering
            initial_k = min(top_k * 8, 100)
            docs = self.vector_store.similarity_search_with_score(query, k=initial_k)

            scored_results = []

            for doc, base_distance in docs:
                content_lower = doc.page_content.lower()
                metadata = doc.metadata

                # Start with semantic similarity as base
                semantic_similarity = max(0, 1 - base_distance)

                # Calculate structured relevance score
                structured_score = 0.0
                explanation_parts = []

                # Entity matching with higher weight
                entity_matches = 0
                total_entity_score = 0
                for entity in query_analysis.entities:
                    if entity.lower() in content_lower:
                        entity_matches += 1
                        # Give higher score for exact matches in key fields
                        if any(
                            field in content_lower
                            for field in ["codebarre:", "id:", "name:", "nom:"]
                        ):
                            total_entity_score += 2.0
                        else:
                            total_entity_score += 1.0
                        explanation_parts.append(f"Entity '{entity}'")

                if query_analysis.entities:
                    entity_score = total_entity_score / len(query_analysis.entities)
                    structured_score += entity_score * 0.5

                # Keyword matching
                keyword_matches = 0
                for keyword in query_analysis.keywords:
                    if keyword in content_lower:
                        keyword_matches += 1
                        explanation_parts.append(f"Keyword '{keyword}'")

                if query_analysis.keywords:
                    keyword_score = keyword_matches / len(query_analysis.keywords)
                    structured_score += keyword_score * 0.3

                # Column relevance
                column_matches = 0
                for column in query_analysis.potential_columns:
                    if f"{column.lower()}:" in content_lower:
                        column_matches += 1
                        explanation_parts.append(f"Column '{column}'")

                if query_analysis.potential_columns:
                    column_score = column_matches / len(
                        query_analysis.potential_columns
                    )
                    structured_score += column_score * 0.2

                # Combine scores with better weighting
                if structured_score > 0:
                    final_score = (structured_score * 0.8) + (semantic_similarity * 0.2)
                else:
                    final_score = (
                        semantic_similarity * 0.5
                    )  # Lower score if no structural matches

                # Only include results with reasonable scores
                if final_score > 0.05:
                    result = RetrievalResult(
                        content=doc.page_content,
                        metadata=metadata,
                        score=final_score,
                        source_table=metadata.get("source", "unknown"),
                        retrieval_method="structured",
                        relevance_explanation=(
                            "; ".join(explanation_parts)
                            if explanation_parts
                            else "Semantic match"
                        ),
                    )
                    scored_results.append(result)

            # Sort by score and return top results
            scored_results.sort(key=lambda x: x.score, reverse=True)
            return scored_results

        except Exception as e:
            logger.error(f"Structured search failed: {e}")
            return []

    def _hybrid_search(
        self, query: str, query_analysis: QueryAnalysis, top_k: int
    ) -> List[RetrievalResult]:
        """Enhanced hybrid search"""
        try:
            # Get results from both methods
            semantic_results = self._semantic_search(query, query_analysis, top_k * 3)
            structured_results = self._structured_search(
                query, query_analysis, top_k * 3
            )

            # Merge results with better deduplication
            combined_results = {}

            # Add semantic results
            for result in semantic_results:
                key = self._generate_result_key(result)
                combined_results[key] = result
                combined_results[key].retrieval_method = "hybrid_semantic"

            # Add/update with structured results
            for result in structured_results:
                key = self._generate_result_key(result)
                if key in combined_results:
                    # Weighted average favoring higher scores
                    existing_score = combined_results[key].score
                    new_score = result.score
                    combined_results[key].score = (
                        max(existing_score, new_score) * 0.7
                        + min(existing_score, new_score) * 0.3
                    )
                    combined_results[key].retrieval_method = "hybrid_both"
                    combined_results[key].relevance_explanation = (
                        f"{combined_results[key].relevance_explanation}; {result.relevance_explanation}"
                    )
                else:
                    combined_results[key] = result
                    combined_results[key].retrieval_method = "hybrid_structured"

            # Sort and return
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x.score, reverse=True)

            return final_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self._semantic_search(query, query_analysis, top_k)

    def _generate_result_key(self, result: RetrievalResult) -> str:
        """Generate a unique key for result deduplication"""
        # Use table name and a hash of the core content
        content = result.content
        if " | " in content:
            # Extract meaningful part (skip metadata)
            content_parts = content.split(" | ")
            data_parts = [
                part
                for part in content_parts
                if not part.startswith(("filename:", "insert_date:", "update_date:"))
            ]
            content = " | ".join(data_parts[:5])  # Use first 5 meaningful parts

        return f"{result.source_table}_{hash(content[:300])}"

    def search_by_table(
        self, query: str, table_name: str, top_k: int = 5, display_mode: str = "smart"
    ) -> List[RetrievalResult]:
        """Enhanced table-specific search"""
        try:
            docs = self.vector_store.similarity_search_with_score(query, k=200)

            # Filter by table and apply minimum score threshold
            table_docs = []
            for doc, distance in docs:
                if doc.metadata.get("source") == table_name:
                    similarity = max(0, 1 - distance)
                    if similarity > 0.05:  # Minimum relevance threshold
                        table_docs.append((doc, similarity))

            # Sort by similarity and limit
            table_docs.sort(key=lambda x: x[1], reverse=True)
            table_docs = table_docs[:top_k]

            results = []
            query_analysis = self.query_analyzer.analyze_query(query)

            for doc, score in table_docs:
                result = RetrievalResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score),
                    source_table=table_name,
                    retrieval_method="table_filtered",
                    relevance_explanation=f"Table-specific search in {table_name}",
                )
                result.formatted_content = self._format_result_content(
                    result.content, query_analysis, display_mode="smart"
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Table search failed: {e}")
            return []

    def get_table_info(self) -> Dict[str, Dict]:
        """Get information about available tables"""
        return self.table_schemas.copy()

    def explain_retrieval(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Explain how retrieval would work for a given query"""
        query_analysis = self.query_analyzer.analyze_query(query)
        strategy = self._choose_strategy(query_analysis)

        explanation = {
            "query": query,
            "analysis": {
                "query_type": query_analysis.query_type,
                "confidence": query_analysis.confidence,
                "entities_found": query_analysis.entities,
                "keywords_found": query_analysis.keywords,
                "potential_columns": query_analysis.potential_columns,
            },
            "strategy_chosen": strategy,
            "available_tables": list(self.table_schemas.keys()),
            "strategy_explanation": self._get_strategy_explanation(
                strategy, query_analysis
            ),
            "table_recommendations": self._recommend_tables(query_analysis),
        }

        return explanation

    def _recommend_tables(self, query_analysis: QueryAnalysis) -> List[str]:
        """Recommend most relevant tables based on query"""
        recommendations = []

        # Simple heuristics for table recommendation
        query_lower = query_analysis.original_query.lower()

        if any(word in query_lower for word in ["carton", "document", "doc"]):
            recommendations.append("dlk_dim_dig_dm_ged_carton.csv")
            recommendations.append("dlk_dim_dig_dm_ged_docs.csv")

        if any(word in query_lower for word in ["property", "tax", "foncier"]):
            recommendations.append("dlk_dim_dig_pat_property.csv")

        if any(word in query_lower for word in ["building", "site", "equipment"]):
            recommendations.extend(
                [
                    "dlk_dim_dig_exp_bldgops_eq_propdef.csv",
                    "dlk_dim_dig_exp_bldgops_servcont.csv",
                ]
            )

        return recommendations[:3]  # Top 3 recommendations

    def _get_strategy_explanation(
        self, strategy: str, query_analysis: QueryAnalysis
    ) -> str:
        """Enhanced strategy explanation"""
        base_explanations = {
            "semantic": f"Using semantic search because confidence is {query_analysis.confidence:.2f} and query appears general.",
            "structured": f"Using structured search because confidence is {query_analysis.confidence:.2f} with {len(query_analysis.entities)} specific entities found.",
            "hybrid": f"Using hybrid search because confidence is {query_analysis.confidence:.2f}, combining semantic and structured approaches.",
        }

        explanation = base_explanations.get(strategy, "Using hybrid search as default.")

        if query_analysis.entities:
            explanation += (
                f" Detected entities: {', '.join(query_analysis.entities[:3])}."
            )

        return explanation

    def get_results_as_table(
        self, query: str, top_k: int = 5, display_mode: str = "smart"
    ) -> List[Dict[str, Any]]:
        """Get results formatted as table rows"""
        results = self.retrieve(query, top_k)

        if display_mode == "full":
            # Return all columns for each result
            return [result.to_dict() for result in results]

        elif display_mode == "smart":
            # Return intelligently selected columns
            table_rows = []
            for result in results:
                row_dict = result.to_dict()

                # Select most informative columns (similar logic to _format_result_content)
                query_analysis = self.query_analyzer.analyze_query(query)

                # You can add smart column selection logic here
                # For now, return the parsed dictionary
                table_rows.append(row_dict)

            return table_rows

        return [result.to_dict() for result in results]

    def retrieve_with_full_columns(
        self, query: str, top_k: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve results showing all available columns"""
        return self.retrieve(query, top_k, display_mode="full")
    
    def get_sample_queries(self) -> List[str]:
        """Get sample queries based on available data"""
        samples = [
            "Find tax documents from 2011",
            "Show documents for Bruno Chavent", 
            "List documents in location 327",
            "Find cartons with barcode P056186762",
            "Show all property tax files",
            "Documents from 2012",
            "Find PDF files",
            "Show building equipment"
        ]
        return samples

    def show_help(self):
        """Show help information"""
        help_text = """
    Available Commands:
    - <query>: Search for documents matching the query
    - explain <query>: Show how the query will be processed
    - table <table_name> <query>: Search within a specific table
    - full <query>: Show results with all columns
    - tables: List available tables and their columns
    - samples: Show sample queries
    - help: Show this help message
    - quit/exit: Exit the program
        """
        print(help_text)
        
        


@dataclass
class QueryRequest:
    """Structured query request with configuration options"""
    
    query: str
    top_k: int = 5
    retrieval_strategy: str = "auto"  # auto, semantic, structured, hybrid
    display_mode: str = "smart"  # smart, full, compact
    table_filter: Optional[str] = None  # Filter by specific table
    min_score: float = 0.1  # Minimum relevance score
    include_metadata: bool = True
    explain_results: bool = False


def process_query(self, query_request: QueryRequest) -> Dict[str, Any]:
    """Main entry point for processing queries with full configuration"""
    
    try:
        # Validate input
        if not query_request.query.strip():
            return {
                "success": False,
                "error": "Empty query provided",
                "results": []
            }
        
        # Analyze the query
        query_analysis = self.query_analyzer.analyze_query(query_request.query)
        
        # Get retrieval results
        if query_request.table_filter:
            results = self.search_by_table(
                query_request.query, 
                query_request.table_filter, 
                query_request.top_k,
                query_request.display_mode
            )
        else:
            results = self.retrieve(
                query_request.query,
                query_request.top_k,
                query_request.retrieval_strategy,
                query_request.display_mode
            )
        
        # Filter by minimum score
        results = [r for r in results if r.score >= query_request.min_score]
        
        # Prepare response
        response = {
            "success": True,
            "query": query_request.query,
            "total_results": len(results),
            "results": []
        }
        
        # Format results
        for result in results:
            result_dict = {
                "content": result.formatted_content if hasattr(result, 'formatted_content') and result.formatted_content else result.content,
                "score": result.score,
                "source_table": result.source_table,
                "retrieval_method": result.retrieval_method
            }
            
            if query_request.include_metadata:
                result_dict["metadata"] = result.metadata
                result_dict["relevance_explanation"] = result.relevance_explanation
            
            response["results"].append(result_dict)
        
        # Add explanation if requested
        if query_request.explain_results:
            response["explanation"] = self.explain_retrieval(query_request.query, query_request.top_k)
            response["query_analysis"] = {
                "query_type": query_analysis.query_type,
                "confidence": query_analysis.confidence,
                "entities": query_analysis.entities,
                "keywords": query_analysis.keywords,
                "potential_columns": query_analysis.potential_columns
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


def simple_query(self, query: str, **kwargs) -> List[Dict[str, Any]]:
    """Simplified query interface that returns just the results"""
    
    query_request = QueryRequest(
        query=query,
        top_k=kwargs.get('top_k', 5),
        retrieval_strategy=kwargs.get('strategy', 'auto'),
        display_mode=kwargs.get('display_mode', 'smart'),
        table_filter=kwargs.get('table', None),
        min_score=kwargs.get('min_score', 0.1)
    )
    
    response = self.process_query(query_request)
    return response.get('results', [])


def interactive_query(self) -> None:
    """Interactive query interface for testing"""
    
    print("=== Interactive RAG Query Interface ===")
    print("Type your queries below. Commands:")
    print("  'tables' - Show available tables")
    print("  'explain <query>' - Explain retrieval strategy")
    print("  'quit' or 'exit' - Exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nQuery: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'tables':
                tables = self.get_table_info()
                print(f"\nAvailable tables ({len(tables)}):")
                for table_name, info in tables.items():
                    print(f"  - {table_name}: {len(info['columns'])} columns")
                continue
            
            if user_input.lower().startswith('explain '):
                query = user_input[8:].strip()
                if query:
                    explanation = self.explain_retrieval(query)
                    print(f"\nQuery Analysis for: '{query}'")
                    print(f"  Strategy: {explanation['strategy_chosen']}")
                    print(f"  Confidence: {explanation['analysis']['confidence']:.2f}")
                    print(f"  Entities: {explanation['analysis']['entities_found']}")
                    print(f"  Keywords: {explanation['analysis']['keywords_found']}")
                continue
            
            # Process regular query
            results = self.simple_query(user_input, top_k=3)
            
            if not results:
                print("No results found.")
                continue
            
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [{result['source_table']}] Score: {result['score']:.3f}")
                print(f"   Content: {result['content'][:200]}...")
                print(f"   Method: {result['retrieval_method']}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_query(self, queries: List[str], **kwargs) -> Dict[str, List[Dict[str, Any]]]:
    """Process multiple queries at once"""
    
    results = {}
    
    for query in queries:
        try:
            query_results = self.simple_query(query, **kwargs)
            results[query] = query_results
        except Exception as e:
            logger.error(f"Batch query failed for '{query}': {e}")
            results[query] = []
    
    return results


def query_with_filters(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Query with advanced filtering options"""
    
    # Build query request from filters
    query_request = QueryRequest(
        query=query,
        top_k=filters.get('limit', 5),
        retrieval_strategy=filters.get('strategy', 'auto'),
        display_mode=filters.get('display', 'smart'),
        table_filter=filters.get('table'),
        min_score=filters.get('min_score', 0.1),
        include_metadata=filters.get('include_metadata', True),
        explain_results=filters.get('explain', False)
    )
    
    response = self.process_query(query_request)
    
    # Apply additional filters
    results = response.get('results', [])
    
    # Filter by source table (if multiple specified)
    if 'tables' in filters and isinstance(filters['tables'], list):
        results = [r for r in results if r['source_table'] in filters['tables']]
    
    # Filter by score range
    if 'score_range' in filters:
        min_s, max_s = filters['score_range']
        results = [r for r in results if min_s <= r['score'] <= max_s]
    
    # Sort by different criteria
    if 'sort_by' in filters:
        reverse = filters.get('sort_desc', True)
        if filters['sort_by'] == 'score':
            results.sort(key=lambda x: x['score'], reverse=reverse)
        elif filters['sort_by'] == 'table':
            results.sort(key=lambda x: x['source_table'], reverse=reverse)
    
    return results[:filters.get('limit', len(results))]

# Usage example and testing functions
def test_retrieval_system():
    """Interactive test of the retrieval system"""
    print("Interactive RAG Retrieval System")
    print("=" * 50)

    try:
        # Initialize retriever
        retriever = HybridRetriever()

        if not retriever.vector_store:
            print("Vector store not available. Please ensure embeddings are created first.")
            return

        # Show available tables
        print(f"Available tables: {list(retriever.get_table_info().keys())}")
        print("\nCommands:")
        print("  - Enter a query to search")
        print("  - 'tables' to show table info")
        print("  - 'explain <query>' to see retrieval strategy")
        print("  - 'table <table_name> <query>' to search specific table")
        print("  - 'full <query>' to show all columns")
        print("  - 'quit' to exit")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nEnter query (or command): ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'tables':
                    print("\nTable Information:")
                    for table_name, info in retriever.get_table_info().items():
                        print(f"\n  {table_name}:")
                        print(f"    Columns: {', '.join(info['columns'][:10])}{'...' if len(info['columns']) > 10 else ''}")
                        print(f"    Total columns: {len(info['columns'])}")
                    continue
                
                if user_input.lower().startswith('explain '):
                    query = user_input[8:].strip()
                    if query:
                        explanation = retriever.explain_retrieval(query)
                        print(f"\nQuery Analysis for: '{query}'")
                        print(f"  Strategy: {explanation['strategy_chosen']}")
                        print(f"  Confidence: {explanation['analysis']['confidence']:.2f}")
                        print(f"  Query type: {explanation['analysis']['query_type']}")
                        print(f"  Entities: {explanation['analysis']['entities_found']}")
                        print(f"  Keywords: {explanation['analysis']['keywords_found']}")
                        print(f"  Potential columns: {explanation['analysis']['potential_columns']}")
                        print(f"  Explanation: {explanation['strategy_explanation']}")
                        if explanation['table_recommendations']:
                            print(f"  Recommended tables: {explanation['table_recommendations']}")
                    continue
                
                if user_input.lower().startswith('table '):
                    parts = user_input[6:].strip().split(' ', 1)
                    if len(parts) == 2:
                        table_name, query = parts
                        print(f"\nSearching in table '{table_name}' for: '{query}'")
                        results = retriever.search_by_table(query, table_name, top_k=5)
                        _display_results(results, query)
                    else:
                        print("Usage: table <table_name> <query>")
                    continue
                
                if user_input.lower().startswith('full '):
                    query = user_input[5:].strip()
                    if query:
                        print(f"\nSearching with full columns for: '{query}'")
                        results = retriever.retrieve_with_full_columns(query, top_k=5)
                        _display_results(results, query, show_all_columns=True)
                    continue
                if user_input.lower() == 'help':
                        retriever.show_help()
                        continue
                
                if user_input.lower() == 'samples':
                    print("\nSample queries to try:")
                    for i, sample in enumerate(retriever.get_sample_queries(), 1):
                        print(f"  {i}. {sample}")
                    continue        
                # Regular search
                query = user_input
                print(f"\nSearching for: '{query}'")
                
                # Show strategy explanation
                explanation = retriever.explain_retrieval(query)
                print(f"Using {explanation['strategy_chosen']} strategy (confidence: {explanation['analysis']['confidence']:.2f})")
                
                # Perform search
                results = retriever.retrieve(query, top_k=5, display_mode="smart")
                _display_results(results, query)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
                logger.error(f"Query processing error: {e}")

    except Exception as e:
        logger.error(f"Test initialization failed: {e}")
        print(f"Test failed: {e}")


def _display_results(results: List[RetrievalResult], query: str, show_all_columns: bool = False):
    """Helper function to display search results"""
    if not results:
        print("  No results found.")
        return
    
    print(f"  Found {len(results)} results:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.source_table}] Score: {result.score:.3f}")
        
        # Choose content to display
        if show_all_columns:
            display_content = result.content
        else:
            display_content = (
                result.formatted_content 
                if hasattr(result, 'formatted_content') and result.formatted_content 
                else result.content
            )
        
        # Truncate very long content for readability
        if len(display_content) > 500:
            display_content = display_content[:500] + "..."
        
        print(f"     Content: {display_content}")
        print(f"     Method: {result.retrieval_method}")
        
        if result.relevance_explanation:
            print(f"     Relevance: {result.relevance_explanation}")
        
        print("-" * 40)

def interactive_query_session():
    """Start an interactive query session"""
    test_retrieval_system()
    

if __name__ == "__main__":
    interactive_query_session()
