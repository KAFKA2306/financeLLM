# vector.py
"""
Enhanced Vector Processing System for Document Management and Retrieval

This module provides a robust implementation for processing documents into vector representations
for efficient similarity search and retrieval. Key features include:

- Multilingual support using the E5 large model
- Intelligent text chunking with configurable parameters
- Comprehensive error handling and recovery
- Detailed logging and progress tracking
- Memory-efficient processing for large documents
- Support for multiple document formats
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
import json
import hashlib
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
import json
import hashlib
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class VectorProcessor:
    def __init__(
        self,
        base_dir: str = "M:/ML/signatejpx",
        model_name: str = "intfloat/multilingual-e5-large",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize the Vector Processing System with enhanced configuration options.

        Args:
            base_dir: Root directory for the project
            model_name: Name of the HuggingFace embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
            batch_size: Number of chunks to process simultaneously
        """
        # First, set up basic logging before any other methods
        self.setup_logging()
        
        # Initialize path structure
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output"
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Configuration parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.model_name = model_name
        
        # Initialize other components after logging is set up
        self.setup_directories()
        self.initialize_text_splitter()
        self.initialize_embeddings()
        
        # Track processed files to avoid duplicates
        self.processed_files: Set[str] = set()
        
        self.logger.info("=== Vector Processing System Initialization Complete ===")

    def setup_directories(self):
        """
        Create and validate the directory structure with enhanced error checking.
        """
        required_dirs = {
            "logs": "Log files",
            "raw_data": "Original text data",
            "chunks": "Processed text chunks",
            "vector_store": "FAISS indices",
            "embeddings": "Computed embeddings",
            "temp": "Temporary processing files"
        }
        
        for dir_name, description in required_dirs.items():
            dir_path = self.output_dir / dir_name
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                # Use print here as a fallback if logger setup fails
                print(f"Verified directory: {dir_path} ({description})") 
                # Add a log message if logger is available
                if hasattr(self, 'logger'):
                    self.logger.info(f"Verified directory: {dir_path} ({description})")
            except Exception as e:
                # Use print as a fallback error reporting
                print(f"Failed to create directory {dir_path}: {str(e)}")
                # Log the error if logger is available
                if hasattr(self, 'logger'):
                    self.logger.error(f"Failed to create directory {dir_path}: {str(e)}")
                raise

    def setup_logging(self):
        """
        Configure comprehensive logging with rotation and structured formatting.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output_dir and logs subdirectory exist
        output_dir = Path("M:/ML/signatejpx/output")
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / f"vector_processing_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("VectorProcessor")


    def initialize_text_splitter(self):
        """
        Initialize the text splitter with optimized settings for multilingual content.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", ". ", ", ", " "],
            keep_separator=True
        )
        self.logger.info(f"Initialized text splitter: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def initialize_embeddings(self):
        """
        Initialize the embedding model with error handling and validation.
        """
        self.logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            # Validate the model by encoding a test string
            test_embedding = self.embeddings.embed_query("Test string")
            self.logger.info(f"Embedding model loaded successfully. Vector size: {len(test_embedding)}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def read_file(self, file_path: Path) -> Optional[str]:
        """
        Read file content with comprehensive encoding handling and validation.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Optional[str]: File content if successful, None if failed
        """
        # Skip MacOS system files and other hidden files
        if file_path.name.startswith('.') or '__MACOSX' in str(file_path):
            self.logger.info(f"Skipping system file: {file_path}")
            return None
            
        encodings = ['utf-8', 'cp932', 'shift_jis', 'euc-jp', 'iso2022_jp']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                self.logger.info(f"Successfully read {file_path} using {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {str(e)}")
                return None
                
        self.logger.error(f"Failed to read {file_path} with any supported encoding")
        return None

    def process_chunks(self, chunks: List[Document], source_id: str) -> Optional[FAISS]:
        """
        Process text chunks into vector store with batching and progress tracking.
        
        Args:
            chunks: List of text chunks to process
            source_id: Identifier for the source document
            
        Returns:
            Optional[FAISS]: Vector store if successful, None if failed
        """
        try:
            self.logger.info(f"Processing {len(chunks)} chunks for {source_id}")
            
            # Process chunks in batches
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                if i == 0:
                    vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    batch_store = FAISS.from_documents(batch, self.embeddings)
                    vector_store.merge_from(batch_store)
                
                self.logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(chunks)-1)//self.batch_size + 1}")
            
            return vector_store
            
        except Exception as e:
            self.logger.error(f"Error processing chunks for {source_id}: {str(e)}")
            return None

    def process_file(self, file_path: Path) -> bool:
        """
        Process a single file with comprehensive error handling and progress tracking.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        source_id = file_path.stem
        
        if source_id in self.processed_files:
            self.logger.info(f"Skipping already processed file: {source_id}")
            return True
            
        self.logger.info(f"\n=== Starting processing: {source_id} ===")
        
        try:
            # Read and validate content
            content = self.read_file(file_path)
            if content is None:
                return False
                
            # Save raw content
            raw_path = self.output_dir / "raw_data" / file_path.name
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Create and save chunks
            chunks = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[{"source": source_id, "file_path": str(file_path)}]
            )
            
            # Save chunk information
            chunks_data = [{
                'content': chunk.page_content,
                'metadata': {
                    'source': source_id,
                    'chunk_id': i,
                    'length': len(chunk.page_content)
                }
            } for i, chunk in enumerate(chunks)]
            
            chunks_file = self.output_dir / "chunks" / f"{source_id}_chunks.json"
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            
            # Process chunks and create vector store
            vector_store = self.process_chunks(chunks, source_id)
            if vector_store is None:
                return False
                
            # Save vector store
            store_path = self.output_dir / "vector_store" / source_id
            vector_store.save_local(str(store_path))
            
            self.processed_files.add(source_id)
            self.logger.info(f"=== Completed processing: {source_id} ===\n")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {source_id}: {str(e)}")
            return False

    def process_all_files(self):
        """
        Process all files in the data directory with parallel execution support.
        """
        self.logger.info("Starting batch processing of all files")
        
        # Find all text files
        text_files = []
        for ext in ['.txt', '.md']:
            text_files.extend(self.data_dir.rglob(f"*{ext}"))
        
        if not text_files:
            self.logger.warning("No text files found for processing")
            return
            
        self.logger.info(f"Found {len(text_files)} files to process")
        
        # Process files with progress tracking
        with tqdm(total=len(text_files), desc="Processing files") as pbar:
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.process_file, file_path): file_path 
                          for file_path in text_files}
                
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        success = future.result()
                        if success:
                            self.logger.info(f"Successfully processed {file_path}")
                        else:
                            self.logger.warning(f"Failed to process {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {str(e)}")
                    pbar.update(1)

    def cleanup(self):
        """
        Clean up temporary files and resources.
        """
        try:
            shutil.rmtree(self.temp_dir)
            self.logger.info("Cleaned up temporary files")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

def main():
    """
    Main execution function with enhanced error handling and reporting.
    """
    processor = None
    try:
        # Initialize processor with custom settings
        processor = VectorProcessor(
            chunk_size=1000,
            chunk_overlap=100,
            batch_size=32
        )
        
        # Process all files
        processor.process_all_files()
        
        # Report processing statistics
        processor.logger.info("\n=== Processing Summary ===")
        processor.logger.info(f"Total files processed: {len(processor.processed_files)}")
        
    except Exception as e:
        logging.error(f"Fatal error in vector processing: {str(e)}", exc_info=True)
        raise
    finally:
        if processor:
            processor.cleanup()


def main():
    """
    Main execution function with enhanced error handling and reporting.
    """
    processor = None
    try:
        # Initialize processor with custom settings
        processor = VectorProcessor(
            chunk_size=1000,
            chunk_overlap=100,
            batch_size=32
        )
        
        # Process all files
        processor.process_all_files()
        
        # Report processing statistics
        processor.logger.info("\n=== Processing Summary ===")
        processor.logger.info(f"Total files processed: {len(processor.processed_files)}")
        
    except Exception as e:
        logging.error(f"Fatal error in vector processing: {str(e)}", exc_info=True)
        raise
    finally:
        if processor:
            processor.cleanup()

if __name__ == "__main__":
    main()