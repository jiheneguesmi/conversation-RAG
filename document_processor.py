import os
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import chardet
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from config import Config

# Configure logging - errors only
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('document_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_requests_per_minute: int = 100, max_monthly_calls: int = 1000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_monthly_calls = max_monthly_calls
        self.request_times = []
        self.monthly_call_count = 0
        self.monthly_reset_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Load previous state if exists
        self.load_state()
        
    def load_state(self):
        """Load rate limiter state from file"""
        try:
            if os.path.exists('rate_limiter_state.json'):
                with open('rate_limiter_state.json', 'r') as f:
                    state = json.load(f)
                    self.monthly_call_count = state.get('monthly_call_count', 0)
                    reset_date_str = state.get('monthly_reset_date')
                    if reset_date_str:
                        self.monthly_reset_date = datetime.fromisoformat(reset_date_str)
        except Exception as e:
            logger.error(f"Could not load rate limiter state: {e}")
    
    def save_state(self):
        """Save rate limiter state to file"""
        try:
            state = {
                'monthly_call_count': self.monthly_call_count,
                'monthly_reset_date': self.monthly_reset_date.isoformat()
            }
            with open('rate_limiter_state.json', 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Could not save rate limiter state: {e}")
    
    def check_monthly_reset(self):
        """Check if monthly limit should be reset"""
        now = datetime.now()
        if now >= self.monthly_reset_date + timedelta(days=31):
            self.monthly_call_count = 0
            self.monthly_reset_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def can_make_request(self) -> bool:
        """Check if a request can be made"""
        self.check_monthly_reset()
        
        now = datetime.now()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
        
        # Check per-minute limit
        if len(self.request_times) >= self.max_requests_per_minute:
            return False
        
        # Check monthly limit
        if self.monthly_call_count >= self.max_monthly_calls:
            return False
        
        return True
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        if not self.can_make_request():
            # If monthly limit exceeded, raise exception
            if self.monthly_call_count >= self.max_monthly_calls:
                raise Exception(f"Monthly API limit of {self.max_monthly_calls} calls exceeded")
            
            # Wait for per-minute limit
            if self.request_times:
                wait_time = 60 - (datetime.now() - min(self.request_times)).total_seconds()
                if wait_time > 0:
                    time.sleep(wait_time + 1)
    
    def record_request(self):
        """Record that a request was made"""
        now = datetime.now()
        self.request_times.append(now)
        self.monthly_call_count += 1
        self.save_state()
    
    def get_remaining_quota(self):
        """Get remaining monthly quota"""
        self.check_monthly_reset()
        return max(0, self.max_monthly_calls - self.monthly_call_count)

class DocumentProcessor:
    def __init__(self, is_trial_key: bool = True):
        self.is_trial_key = is_trial_key
        
        # Set rate limits based on key type
        if is_trial_key:
            self.rate_limiter = RateLimiter(max_requests_per_minute=90, max_monthly_calls=1000)
        else:
            self.rate_limiter = RateLimiter(max_requests_per_minute=1800, max_monthly_calls=999999)
        
        # Fix: Check API key before initializing embeddings
        if not Config.COHERE_API_KEY or Config.COHERE_API_KEY == "your_api_key_here":
            logger.error("Cohere API key not found. Please set COHERE_API_KEY environment variable.")
            self.embeddings = None
        else:
            try:
                self.embeddings = CohereEmbeddings(
                    model=Config.EMBEDDING_MODEL,
                    cohere_api_key=Config.COHERE_API_KEY
                )
            except Exception as e:
                logger.error(f"Cohere embeddings failed: {e}")
                self.embeddings = None
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': [],
            'total_documents': 0,
            'total_chunks': 0,
            'file_types': {},
            'api_calls_made': 0,
            'processing_time': 0
        }
    
    def load_documents(self, folder_path: str, max_rows_per_file: Optional[int] = None):
        """Load documents from folder"""
        documents = []
        start_time = time.time()
        
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': [],
            'total_documents': 0,
            'total_chunks': 0,
            'file_types': {},
            'api_calls_made': 0,
            'processing_time': 0
        }
        
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return documents
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.processing_stats['total_files'] = len(files)
        
        if not files:
            logger.error("No CSV files found")
            return documents
        
        for i, filename in enumerate(files, 1):
            file_path = os.path.join(folder_path, filename)
            
            try:
                docs = self._load_csv_file(file_path, max_rows_per_file)
                documents.extend(docs)
                self.processing_stats['processed_files'] += 1
                self.processing_stats['total_documents'] += len(docs)
                
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
                self.processing_stats['failed_files'].append({'filename': filename, 'error': str(e)})
        
        self.processing_stats['processing_time'] = time.time() - start_time
        return documents
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet with better fallback handling"""
        try:
            # First try to detect with chardet on a larger sample
            with open(file_path, 'rb') as f:
                raw_data = f.read(50000)  # Read first 50KB for better detection
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', '').lower()
                confidence = result.get('confidence', 0)
                
                print(f"Detected encoding for {os.path.basename(file_path)}: {encoding} (confidence: {confidence:.2f})")
                
                # If chardet detected UTF-8 or similar, use it
                if encoding in ['utf-8', 'utf-8-sig'] and confidence > 0.5:
                    return encoding
                
                # If detected ISO-8859-1 or latin-1 with high confidence, use it
                if encoding in ['iso-8859-1', 'latin-1', 'cp1252', 'windows-1252'] and confidence > 0.7:
                    return 'iso-8859-1'  # Use iso-8859-1 as it's more compatible
                
                # Test common encodings in order of preference for French text
                encodings_to_try = [
                    'utf-8-sig',      # UTF-8 with BOM
                    'utf-8',          # UTF-8
                    'iso-8859-1',     # Latin-1 (common for French)
                    'cp1252',         # Windows-1252 (common for Windows French)
                    'latin1',         # Alias for iso-8859-1
                    'cp437',          # IBM PC encoding
                ]
                
                # Try each encoding by attempting to read the file
                for test_encoding in encodings_to_try:
                    try:
                        with open(file_path, 'r', encoding=test_encoding, errors='strict') as test_file:
                            # Try to read a substantial portion
                            content = test_file.read(10000)
                            # If we can read without errors, this encoding works
                            print(f"Successfully tested encoding {test_encoding} for {os.path.basename(file_path)}")
                            return test_encoding
                    except (UnicodeDecodeError, UnicodeError) as e:
                        print(f"Failed encoding {test_encoding} for {os.path.basename(file_path)}: {e}")
                        continue
                
                # If all else fails, use utf-8 with error replacement
                print(f"Falling back to utf-8 with error replacement for {os.path.basename(file_path)}")
                return 'utf-8'
                
        except Exception as e:
            logger.error(f"Encoding detection failed for {file_path}: {e}")
            return 'utf-8'
    
    def _load_csv_file(self, file_path: str, max_rows: Optional[int] = None):
        """Load CSV file with improved encoding detection and error handling"""
        documents = []
        filename = os.path.basename(file_path)
        
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            print(f"Processing {filename} with encoding: {encoding}")
            
            # Try to read with detected encoding
            try:
                # Use errors='replace' to handle any remaining problematic characters
                read_kwargs = {
                    'encoding': encoding,
                    'on_bad_lines': 'skip',  # Skip bad lines instead of failing
                    'engine': 'python',     # Use Python engine for better error handling
                    'sep': ',',             # Explicitly set separator
                    'quoting': 1,           # QUOTE_ALL
                    'skipinitialspace': True, # Skip whitespace after delimiter
                }
                
                # First, count total rows for progress tracking
                try:
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        total_rows = sum(1 for _ in f) - 1  # Subtract header row
                except:
                    total_rows = 1000  # Fallback estimate
                
                # Try to read first few rows to test
                test_df = pd.read_csv(file_path, nrows=5, **read_kwargs)
                print(f"Successfully read test rows from {filename}")
                
            except Exception as e:
                # If that fails, try with utf-8 and more aggressive error handling
                print(f"Failed to read {filename} with {encoding}, trying utf-8 with error replacement: {e}")
                encoding = 'utf-8'
                read_kwargs = {
                    'encoding': encoding,
                    'encoding_errors': 'replace',  # Replace bad characters
                    'on_bad_lines': 'skip',
                    'engine': 'python',
                    'sep': ',',
                    'quoting': 1,
                    'skipinitialspace': True,
                }
                
                try:
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        total_rows = sum(1 for _ in f) - 1
                except:
                    total_rows = 1000
                    
                test_df = pd.read_csv(file_path, nrows=5, **read_kwargs)
                print(f"Successfully read test rows from {filename} with utf-8 fallback")
            
            # Calculate rows to process
            rows_to_process = min(max_rows, total_rows) if max_rows else total_rows
            print(f"Processing {rows_to_process} rows from {filename} (total: {total_rows})")
            
            # Process in chunks to handle memory better
            chunk_size = 1000
            processed_rows = 0
            
            try:
                for chunk_df in pd.read_csv(file_path, chunksize=chunk_size, **read_kwargs):
                    if processed_rows >= rows_to_process:
                        break
                    
                    # Limit chunk if needed
                    if processed_rows + len(chunk_df) > rows_to_process:
                        chunk_df = chunk_df.head(rows_to_process - processed_rows)
                    
                    # Clean the dataframe
                    chunk_df = chunk_df.fillna('')  # Replace NaN with empty string
                    
                    # Process each row
                    for idx, row in chunk_df.iterrows():
                        # Clean and combine all columns into a single text
                        text_parts = []
                        metadata = {
                            "source": filename,
                            "row_index": int(idx),
                            "file_type": "csv",
                            "encoding_used": encoding
                        }
                        
                        for col, val in row.items():
                            if pd.notna(val) and str(val).strip() != '':  # Skip empty/null values
                                # Clean the value - handle various problematic characters
                                clean_val = str(val).replace('\x00', '').replace('\ufffd', '?').replace('\r', '').replace('\n', ' ').strip()
                                
                                if clean_val:  # Only add if there's actual content
                                    text_parts.append(f"{col}: {clean_val}")
                                    
                                    # Add to metadata (limit to avoid memory issues)
                                    if len(metadata) < 50:  # Limit metadata size
                                        metadata[str(col)] = clean_val[:200]  # Limit field length
                        
                        if text_parts:  # Only create document if there's content
                            text_content = " | ".join(text_parts)
                            
                            # Create document
                            doc = Document(
                                page_content=text_content,
                                metadata=metadata
                            )
                            documents.append(doc)
                    
                    processed_rows += len(chunk_df)
                    
                    # Progress indicator
                    if processed_rows % 5000 == 0:
                        print(f"Processed {processed_rows}/{rows_to_process} rows from {filename}")
                        
            except Exception as e:
                logger.error(f"Error processing chunks from {filename}: {e}")
                # Don't raise here, return what we have
                
            print(f"Successfully processed {len(documents)} documents from {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"CSV processing error in {filename}: {e}")
            raise
    
    def process_documents(self, documents: List[Document], batch_size: int = 100, quota_limit: Optional[int] = None):
        """Split documents into chunks with batch processing and quota awareness"""
        if not documents:
            logger.error("No documents to process")
            return []
        
        start_time = time.time()
        
        # If quota_limit is specified, limit the number of documents we process
        if quota_limit:
            remaining_quota = self.rate_limiter.get_remaining_quota()
            if quota_limit > remaining_quota:
                quota_limit = remaining_quota
            
            # Estimate documents per chunk (roughly 1.8 chunks per document based on your data)
            max_documents = int(quota_limit / 1.8)  # Conservative estimate
            
            if len(documents) > max_documents:
                print(f"Limiting processing to {max_documents} documents due to API quota ({quota_limit} calls remaining)")
                documents = documents[:max_documents]
        
        all_chunks = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_chunks = self.text_splitter.split_documents(batch)
            all_chunks.extend(batch_chunks)
            
            # If we have a quota limit, check if we're approaching it
            if quota_limit and len(all_chunks) >= quota_limit:
                print(f"Stopping chunk creation at {len(all_chunks)} chunks to stay within quota limit")
                all_chunks = all_chunks[:quota_limit]
                break
        
        self.processing_stats['total_chunks'] = len(all_chunks)
        
        return all_chunks
    
    def create_vector_store(self, chunks: List[Document], batch_size: int = 50, max_chunks: Optional[int] = None):
        """Create vector database from chunks with rate limiting and quota awareness"""
        if not chunks:
            logger.error("No chunks to create vector store")
            return None
        
        if not self.embeddings:
            logger.error("No embeddings model available - check your API key")
            return None
        
        start_time = time.time()
        
        # Create directory if it doesn't exist
        os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)
        
        # Check quota and limit chunks if necessary
        remaining_quota = self.rate_limiter.get_remaining_quota()
        
        if max_chunks:
            chunks_to_process = min(len(chunks), max_chunks, remaining_quota)
        else:
            chunks_to_process = min(len(chunks), remaining_quota)
        
        if chunks_to_process < len(chunks):
            print(f"Limiting processing to {chunks_to_process} chunks due to API quota (remaining: {remaining_quota})")
            chunks = chunks[:chunks_to_process]
        
        if chunks_to_process == 0:
            logger.error("No API quota remaining for vector store creation")
            raise Exception("No API quota remaining for vector store creation")
        
        print(f"Creating vector store with {len(chunks)} chunks...")
        
        # Process in batches
        vector_store = None
        processed_chunks = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Rate limiting
            for _ in range(len(batch)):
                self.rate_limiter.wait_if_needed()
                self.rate_limiter.record_request()
            
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(
                        documents=batch,
                        embedding=self.embeddings
                    )
                else:
                    batch_vector_store = FAISS.from_documents(
                        documents=batch,
                        embedding=self.embeddings
                    )
                    vector_store.merge_from(batch_vector_store)
                
                processed_chunks += len(batch)
                self.processing_stats['api_calls_made'] += len(batch)
                
                # Progress update
                print(f"Processed {processed_chunks}/{len(chunks)} chunks...")
                
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print("Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    logger.error(f"Vector store error: {e}")
                    raise
        
        # Save final vector store
        if vector_store:
            try:
                vector_store.save_local(Config.VECTOR_DB_PATH)
                print(f"Vector store saved to {Config.VECTOR_DB_PATH}")
            except Exception as e:
                logger.error(f"Failed to save vector store: {e}")
        
        return vector_store
    
    def load_vector_store(self):
        """Load existing vector database"""
        if not os.path.exists(Config.VECTOR_DB_PATH):
            logger.error(f"Vector store not found: {Config.VECTOR_DB_PATH}")
            return None
        
        # Check if index files exist
        index_file = os.path.join(Config.VECTOR_DB_PATH, "index.faiss")
        pkl_file = os.path.join(Config.VECTOR_DB_PATH, "index.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(pkl_file):
            logger.error(f"Vector store files missing in {Config.VECTOR_DB_PATH}")
            return None
        
        if not self.embeddings:
            logger.error("No embeddings model available - check your API key")
            return None
        
        try:
            vector_store = FAISS.load_local(
                Config.VECTOR_DB_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            logger.error(f"Vector store load failed: {e}")
            return None
    
    def create_incremental_vector_store(self, folder_path: str, max_rows_per_file: int = 100, chunks_per_batch: int = 50):
        """Create vector store incrementally to work within quota limits"""
        print("Creating incremental vector store...")
        
        # Load a limited number of documents
        print(f"Loading documents with limit of {max_rows_per_file} rows per file...")
        documents = self.load_documents(folder_path, max_rows_per_file)
        
        if not documents:
            print("No documents loaded")
            return None
        
        print(f"Loaded {len(documents)} documents")
        
        # Get remaining quota
        remaining_quota = self.rate_limiter.get_remaining_quota()
        print(f"Remaining API quota: {remaining_quota}")
        
        if remaining_quota <= 0:
            print("No API quota remaining. Cannot create vector store.")
            return None
        
        # Process documents with quota awareness
        chunks = self.process_documents(documents, quota_limit=remaining_quota)
        
        if not chunks:
            print("No chunks created")
            return None
        
        print(f"Created {len(chunks)} chunks")
        
        # Create vector store with available quota
        vector_store = self.create_vector_store(chunks, batch_size=chunks_per_batch, max_chunks=remaining_quota)
        
        # Print summary
        self.print_processing_summary()
        
        return vector_store
    
    def test_vector_store(self, query: str, k: int = 3):
        """Test vector store with a query"""
        vector_store = self.load_vector_store()
        
        if not vector_store:
            print("No vector store available for testing")
            return
        
        try:
            results = vector_store.similarity_search(query, k=k)
            
            print(f"\nQuery: {query}")
            print(f"Found {len(results)} results:\n")
            
            for i, doc in enumerate(results, 1):
                print(f"Result {i}:")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Content: {doc.page_content[:200]}...")
                print("-" * 50)
                
        except Exception as e:
            logger.error(f"Vector store test failed: {e}")
    
    def get_processing_stats(self):
        """Get processing statistics"""
        return self.processing_stats
    
    def print_processing_summary(self):
        """Print a summary of processing results"""
        stats = self.processing_stats
        
        print("\n" + "="*50)
        print("DOCUMENT PROCESSING SUMMARY")
        print("="*50)
        print(f"Files processed: {stats['processed_files']}/{stats['total_files']}")
        print(f"Documents created: {stats['total_documents']:,}")
        print(f"Chunks created: {stats['total_chunks']:,}")
        print(f"API calls made: {stats['api_calls_made']:,}")
        print(f"Processing time: {stats['processing_time']:.1f}s")
        print(f"Quota remaining: {self.rate_limiter.get_remaining_quota():,}")
        
        if stats['failed_files']:
            print(f"\nFailed files ({len(stats['failed_files'])}):")
            for failed in stats['failed_files']:
                error_short = failed['error'][:80] + "..." if len(failed['error']) > 80 else failed['error']
                print(f"  {failed['filename']}: {error_short}")
        
        print("="*50)
    
    def estimate_processing_cost(self, folder_path: str, max_rows_per_file: Optional[int] = None):
        """Estimate API calls needed for processing"""
        total_estimated_calls = 0
        
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return 0
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not files:
            logger.error("No CSV files found")
            return 0
        
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Use encoding detection for estimation too
                encoding = self._detect_encoding(file_path)
                
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    total_rows = sum(1 for _ in f) - 1  # Subtract header
                
                rows_to_process = min(max_rows_per_file, total_rows) if max_rows_per_file else total_rows
                estimated_chunks = max(1, rows_to_process // 2)  # Rough estimate
                
                total_estimated_calls += estimated_chunks
                
            except Exception as e:
                logger.error(f"Error estimating {filename}: {e}")
                # Use a default estimate for failed files
                total_estimated_calls += 100
        
        remaining_quota = self.rate_limiter.get_remaining_quota()
        
        print(f"\nEstimated API calls needed: {total_estimated_calls:,}")
        print(f"Remaining quota: {remaining_quota:,}")
        
        if total_estimated_calls > remaining_quota:
            print("  WARNING: Estimated calls exceed remaining quota!")
            recommended_rows = max(1, int((remaining_quota / total_estimated_calls) * (max_rows_per_file or 1000)))
            print(f"  RECOMMENDATION: Limit to {recommended_rows} rows per file to stay within quota")
            return total_estimated_calls
        
        return total_estimated_calls

# Standalone testing functions
def run_quota_aware_processing(folder_path="./data/documents", max_rows_per_file=50, is_trial_key=True):
    """Run document processing that respects API quota limits"""
    print("Initializing document processor...")
    processor = DocumentProcessor(is_trial_key=is_trial_key)
    
    # Check if API key is available
    if not processor.embeddings:
        print(" Cannot proceed without valid Cohere API key")
        print("Please set COHERE_API_KEY environment variable or update config.py")
        return processor
    
    # Check remaining quota first
    remaining_quota = processor.rate_limiter.get_remaining_quota()
    print(f"Remaining API quota: {remaining_quota}")
    
    if remaining_quota <= 0:
        print("No API quota remaining. Please wait for quota reset or upgrade your plan.")
        return processor
    
    # Estimate cost with current settings
    print("Estimating processing cost...")
    estimated_calls = processor.estimate_processing_cost(folder_path, max_rows_per_file)
    
    # If estimated calls exceed quota, suggest a smaller limit
    if estimated_calls > remaining_quota:
        suggested_rows = max(1, int(max_rows_per_file * remaining_quota / estimated_calls))
        print(f"Reducing max_rows_per_file from {max_rows_per_file} to {suggested_rows} to fit within quota")
        max_rows_per_file = suggested_rows
    
    if estimated_calls > 0:
        response = input(f"\nContinue with processing up to {min(estimated_calls, remaining_quota)} chunks? (y/N): ")
        if response.lower() != 'y':
            print("Processing cancelled by user")
            return processor
    
    # Use the incremental processing method
    vector_store = processor.create_incremental_vector_store(
        folder_path=folder_path,
        max_rows_per_file=max_rows_per_file,
        chunks_per_batch=25  # Smaller batch size for better error handling
    )
    
    if vector_store:
        print("Vector store created successfully!")
    else:
        print("Vector store creation failed or incomplete")
    
    return processor

if __name__ == "__main__":
    # Configuration - Adjusted for quota constraints
    IS_TRIAL_KEY = True  # Set to False if you have a production key
    MAX_ROWS_PER_FILE = 50  # Start small to fit within quota
    
    print("Quota-Aware Document Processing Script")
    print("=" * 40)
    
    # Run quota-aware processing
    test_processor = run_quota_aware_processing(
        folder_path="./data/documents",
        max_rows_per_file=MAX_ROWS_PER_FILE,
        is_trial_key=IS_TRIAL_KEY
    )
    
    # Test vector store if it exists
    if test_processor.load_vector_store():
        print("\n No vector store available for testing")