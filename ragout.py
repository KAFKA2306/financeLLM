import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import urllib3
import PyPDF2
from tqdm import tqdm
import time

class EnhancedRAG:
    def __init__(self, base_dir: str = "M:/ML/signatejpx", output_dir: str = "M:/ML/signatejpx/output"):
        """
        Initialize Enhanced RAG system with proper directory structure and comprehensive logging
        
        Args:
            base_dir: Base directory containing data files
            output_dir: Directory for saving outputs
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.docs_dir = self.base_dir / "data" / "documents" / "documents"
        
        # Setup logging and directories
        self.setup_logging()
        self.setup_directories()
        
        self.logger.info("Initializing EnhancedRAG system...")
        
        # Initialize core components
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        
        # Suppress insecure request warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Load configuration
        self.config = self.load_config()
        self.logger.info("Initialization completed successfully")
        
    def setup_logging(self):
        """Configure logging system"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rag_run_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directory structure"""
        dirs = [
            "raw_data",      # Store raw input data
            "processed",     # Store processed texts and embeddings
            "results",       # Store RAG results
            "analysis",      # Store analysis and metrics
            "logs"          # Store execution logs
        ]
        
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
        self.logger.info("Directory structure verified and created")
        
    def load_config(self) -> Dict[str, Any]:
        """Load API configuration"""
        try:
            config_path = self.base_dir / "secret" / "config.py"
            self.logger.info(f"Loading configuration from: {config_path}")
            
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", str(config_path))
            if spec is None or spec.loader is None:
                raise ImportError("Failed to load config module")
                
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            
            return {
                'model': config.QWEN_CONFIG['model'],
                'api_url': config.QWEN_CONFIG['api_url'],
                'api_key': config.QWEN_CONFIG['api_key']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
            
    def read_pdf(self, pdf_path: str) -> Tuple[str, bool]:
        """
        Read and extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple[str, bool]: (Extracted text content, Success flag)
        """
        try:
            text = []
            start_time = time.time()
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text.append(page_text)
                    
            duration = time.time() - start_time
            self.logger.info(f"Successfully read PDF {pdf_path} ({total_pages} pages) in {duration:.2f} seconds")
            
            return "\n".join(text), True
            
        except Exception as e:
            self.logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return "", False
            
    def load_documents(self) -> List[str]:
        """
        Load all PDF documents from the correct documents directory
        
        Returns:
            List[str]: List of extracted text content from PDFs
        """
        self.logger.info(f"Loading documents from: {self.docs_dir}")
        pdf_files = sorted(self.docs_dir.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.error(f"No PDF files found in {self.docs_dir}")
            raise FileNotFoundError(f"No PDF files found in {self.docs_dir}")
            
        texts = []
        successful_loads = 0
        failed_loads = 0
        
        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            if pdf_file.name.startswith('.'):  # Skip hidden files
                continue
                
            text, success = self.read_pdf(str(pdf_file))
            if success and text:
                texts.append(text)
                successful_loads += 1
                
                # Save raw text for reference
                raw_path = self.output_dir / "raw_data" / f"{pdf_file.stem}.txt"
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                failed_loads += 1
                
        self.logger.info(f"Document loading completed: {successful_loads} successful, {failed_loads} failed")
        return texts
        
    def create_vector_store(self, texts: List[str]) -> FAISS:
        """Create vector store from input texts"""
        self.logger.info("Creating vector store...")
        chunks = self.text_splitter.create_documents(texts)
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.logger.info(f"Vector store created with {len(chunks)} chunks")
        return vector_store
        
    def retrieve_documents(self, query: str, vector_store: FAISS, k: int = 3) -> List[str]:
        """Retrieve relevant documents for the query"""
        docs = vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
        
    def parse_llm_response(self, response_text: str) -> str:
        """Parse LLM API response with enhanced error handling"""
        try:
            if not response_text.strip().startswith('{'):
                return response_text.strip('"\'')
                
            response_json = json.loads(response_text)
            if isinstance(response_json, dict):
                if "response" in response_json:
                    return response_json["response"]
                elif "generated_text" in response_json:
                    return response_json["generated_text"]
                elif "choices" in response_json and isinstance(response_json["choices"], list):
                    return response_json["choices"][0]["text"]
                    
            return str(response_json)
            
        except Exception as e:
            self.logger.warning(f"Error parsing response: {str(e)}")
            return response_text
            
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using LLM API with improved error handling"""
        prompt = f"""Based on the following context, please answer the question. If you cannot find relevant information in the context, respond with "分かりません".

Context:
{chr(10).join(context)}

Question: {query}

Answer:"""

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.config['api_key']
        }
        
        data = {
            "temperature": 0.7,
            "prompt": prompt,
            "repeat_penalty": 1.0
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.config['api_url'],
                headers=headers,
                json=data,
                verify=False,
                timeout=30
            )
            
            duration = time.time() - start_time
            self.logger.info(f"API request completed in {duration:.2f} seconds")
            
            if response.status_code != 200:
                self.logger.error(f"API error: Status {response.status_code}")
                raise Exception(f"API returned status {response.status_code}")
                
            parsed_response = self.parse_llm_response(response.text)
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise RuntimeError(f"API request failed: {str(e)}")
            
    def save_results(self, query: str, response: str, context: List[str]) -> None:
        """Save results with metadata and logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "timestamp": timestamp,
            "query": query,
            "context": context,
            "response": response
        }
        
        # Save individual result
        result_path = self.output_dir / "results" / f"result_{timestamp}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved result to {result_path}")
        
    def process_queries(self, queries: pd.DataFrame, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries with progress tracking and error handling"""
        self.logger.info(f"Starting query processing for {len(queries)} queries")
        vector_store = self.create_vector_store(texts)
        results = []
        
        for _, row in tqdm(queries.iterrows(), total=len(queries), desc="Processing queries"):
            try:
                query = row["problem"]
                self.logger.info(f"Processing query: {query}")
                
                context = self.retrieve_documents(query, vector_store)
                response = self.generate_response(query, context)
                
                result = {
                    "index": row["index"],
                    "response": response
                }
                results.append(result)
                
                # Save individual result
                self.save_results(query, response, context)
                
            except Exception as e:
                self.logger.error(f"Error processing query {row['index']}: {str(e)}")
                # Continue with next query instead of failing completely
                results.append({
                    "index": row["index"],
                    "response": "分かりません"
                })
                
        self.logger.info("Query processing completed")
        return results

def main():
    try:
        # Initialize RAG system
        rag = EnhancedRAG()
        
        # Load queries
        queries = pd.read_csv(rag.base_dir / "data" / "query.csv")
        rag.logger.info(f"Loaded {len(queries)} queries")
        
        # Load documents
        texts = rag.load_documents()
        rag.logger.info(f"Loaded {len(texts)} documents")
        
        # Process queries
        results = rag.process_queries(queries, texts)
        
        # Save final predictions
        predictions_df = pd.DataFrame(results)
        output_path = rag.output_dir / "processed" / "predictions.csv"
        predictions_df.to_csv(output_path, index=False, header=False)
        rag.logger.info(f"Final results saved to {output_path}")
        
    except Exception as e:
        if hasattr(rag, 'logger'):
            rag.logger.error(f"Fatal error: {str(e)}", exc_info=True)
        else:
            print(f"Initialization error: {str(e)}")
            
if __name__ == "__main__":
    main()