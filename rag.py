import json
import requests
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from requests.exceptions import RequestException
import warnings
import urllib3

# Suppress insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load LLM configuration from the specified path"""
    if config_path is None:
        config_path = os.path.join('M:\\', 'ML', 'signatejpx', 'secret', 'config.py')
    
    config_path = Path(config_path).resolve()
    print(f"Loading configuration from: {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

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

def initialize_embedding_model() -> Tuple[HuggingFaceEmbeddings, RecursiveCharacterTextSplitter]:
    """Initialize the embedding model and text splitter"""
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    
    return embeddings, text_splitter

def create_vectorstore(texts: List[str], embeddings: HuggingFaceEmbeddings, 
                      text_splitter: RecursiveCharacterTextSplitter) -> FAISS:
    """Create vector store from input texts"""
    if not texts:
        raise ValueError("Empty text list provided")
    
    chunks = text_splitter.create_documents(texts)
    return FAISS.from_documents(chunks, embeddings)

def retrieve_documents(query: str, vector_store: FAISS, k: int = 3) -> List[str]:
    """Retrieve relevant documents for the query"""
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

def parse_llm_response(response_json: Dict[str, Any]) -> str:
    """Parse LLM API response with detailed error handling"""
    # Print response for debugging
    print(f"API Response structure: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
    
    # Try different known response formats
    if "choices" in response_json and isinstance(response_json["choices"], list):
        choice = response_json["choices"][0]
        if isinstance(choice, dict):
            if "text" in choice:
                return choice["text"]
            elif "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
    
    if "response" in response_json:
        return response_json["response"]
        
    if "generated_text" in response_json:
        return response_json["generated_text"]
        
    if "output" in response_json:
        return response_json["output"]
    
    # If we get here, we couldn't parse the response
    available_keys = list(response_json.keys())
    raise ValueError(f"Unexpected API response format. Available keys: {available_keys}")

def generate_response(query: str, context: List[str], config: Dict[str, str]) -> str:
    """Generate response using LLM API with improved error handling"""
    prompt = f"""Based on the following context, please answer the question.

Context:
{chr(10).join(context)}

Question: {query}

Answer:"""

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": config['api_key']
    }
    
    data = {
        "temperature": 0.7,
        "prompt": prompt,
        "repeat_penalty": 1.0
    }
    
    print(f"Sending request to API: {config['api_url']}")
    print(f"Request data: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(
            config['api_url'],
            headers=headers,
            json=data,
            verify=False,
            timeout=30
        )
        
        print(f"API Status Code: {response.status_code}")
        
        if response.status_code != 200:
            error_detail = response.text if response.text else "No error details provided"
            raise RequestException(f"API returned status {response.status_code}: {error_detail}")
        
        result = response.json()
        return parse_llm_response(result)
        
    except json.JSONDecodeError as e:
        print(f"Raw API Response: {response.text}")
        raise RuntimeError(f"Failed to decode API response: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"API request failed: {str(e)}")

def query_rag(query: str, vector_store: FAISS, config: Dict[str, str]) -> str:
    """Execute the complete RAG pipeline"""
    context = retrieve_documents(query, vector_store)
    return generate_response(query, context, config)

def main():
    """Main function demonstrating RAG usage"""
    try:
        # Initialize system
        config = load_config()
        print("Configuration loaded successfully")
        
        embeddings, text_splitter = initialize_embedding_model()
        print("Embedding model initialized")
        
        # Prepare sample texts
        texts = [
            "This is a sample document for testing.",
            "Another example text to demonstrate RAG.",
            "A third document with different content."
        ]
        
        # Create vector store and test the system
        vector_store = create_vectorstore(texts, embeddings, text_splitter)
        print("Vector store created successfully")
        
        query = "What are these documents about?"
        print(f"\nProcessing query: {query}")
        
        answer = query_rag(query, vector_store, config)
        print(f"\nQuestion: {query}")
        print(f"Answer: {answer}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(f"\nFull error traceback:")
        print(traceback.format_exc())









def parse_llm_response(response_text: str) -> str:
    """
    Parse the LLM API response, handling both string and JSON responses.
    
    Args:
        response_text: Raw response text from the API
        
    Returns:
        str: Parsed response text
        
    This improved version handles:
    - Direct string responses
    - JSON responses in various formats
    - Proper error reporting
    """
    try:
        # If response is already a clean string, return it
        if not response_text.strip().startswith('{'):
            return response_text.strip('"\'')
            
        # Try parsing as JSON if it looks like JSON
        response_json = json.loads(response_text)
        
        # Handle various JSON response formats
        if isinstance(response_json, dict):
            if "choices" in response_json and isinstance(response_json["choices"], list):
                return response_json["choices"][0]["text"]
            elif "response" in response_json:
                return response_json["response"]
            elif "generated_text" in response_json:
                return response_json["generated_text"]
            elif "output" in response_json:
                return response_json["output"]
            # If none of the expected fields are found, convert the whole response to string
            return str(response_json)
            
        # If response_json is not a dict (e.g., string), return it as is
        return str(response_json)
        
    except json.JSONDecodeError:
        # If JSON parsing fails, return the cleaned string
        return response_text.strip('"\'')
    except Exception as e:
        # Log the error but don't raise it - return the original text
        print(f"Warning: Error while parsing response: {str(e)}")
        return response_text

def generate_response(query: str, context: List[str], config: Dict[str, str]) -> str:
    """
    Generate response using LLM API with improved error handling and response parsing.
    
    Args:
        query: User's question
        context: Retrieved relevant documents
        config: API configuration
        
    Returns:
        str: Generated response from the LLM
    """
    prompt = f"""Based on the following context, please answer the question.

Context:
{chr(10).join(context)}

Question: {query}

Answer:"""

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": config['api_key']
    }
    
    data = {
        "temperature": 0.7,
        "prompt": prompt,
        "repeat_penalty": 1.0
    }
    
    print(f"Sending request to API: {config['api_url']}")
    print(f"Request data: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(
            config['api_url'],
            headers=headers,
            json=data,
            verify=False,
            timeout=30
        )
        
        print(f"API Status Code: {response.status_code}")
        
        if response.status_code != 200:
            error_detail = response.text if response.text else "No error details provided"
            raise RequestException(f"API returned status {response.status_code}: {error_detail}")
        
        # Get raw response text
        response_text = response.text
        print(f"Raw API Response: {response_text}")
        
        # Parse the response using our improved parser
        parsed_response = parse_llm_response(response_text)
        if not parsed_response:
            raise ValueError("Empty response from API")
            
        return parsed_response
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails but we have response text, try to use it directly
        if response.text:
            return parse_llm_response(response.text)
        raise RuntimeError(f"Failed to decode API response: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"API request failed: {str(e)}")
    
def parse_llm_response(response_text: str) -> str:
    """
    chat.pyの実装を基にした、シンプルで堅牢な応答解析
    """
    try:
        # JSON形式かどうかをチェック
        if response_text.strip().startswith('{'):
            response_json = json.loads(response_text)
            if isinstance(response_json, dict):
                # 既知の応答フォーマットを順番にチェック
                if "response" in response_json:
                    return response_json["response"]
                elif "generated_text" in response_json:
                    return response_json["generated_text"]
                elif "choices" in response_json and isinstance(response_json["choices"], list):
                    return response_json["choices"][0]["text"]
            # 上記以外の場合は文字列として返す
            return str(response_json)
        
        # JSON以外は単純な文字列として処理
        return response_text.strip('"\'')
        
    except json.JSONDecodeError:
        # JSONパースに失敗しても文字列として返す
        return response_text.strip('"\'')
    except Exception as e:
        print(f"Warning: Error while parsing response: {str(e)}")
        return response_text

def generate_response(query: str, context: List[str], config: Dict[str, str]) -> str:
    """
    LLM APIを使用して応答を生成する関数（改善版）
    """
    prompt = f"""Based on the following context, please answer the question.

Context:
{chr(10).join(context)}

Question: {query}

Answer:"""

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": config['api_key']
    }
    
    data = {
        "temperature": 0.7,
        "prompt": prompt,
        "repeat_penalty": 1.0
    }
    
    print(f"Sending request to API: {config['api_url']}")
    
    try:
        response = requests.post(
            config['api_url'],
            headers=headers,
            json=data,
            verify=False,
            timeout=30
        )
        
        print(f"API Status Code: {response.status_code}")
        
        if response.status_code != 200:
            error_detail = response.text if response.text else "No error details provided"
            raise RequestException(f"API returned status {response.status_code}: {error_detail}")
        
        # 応答テキストを直接parse_llm_responseに渡す
        parsed_response = parse_llm_response(response.text)
        if not parsed_response:
            raise ValueError("Empty response from API")
            
        return parsed_response
        
    except Exception as e:
        raise RuntimeError(f"API request failed: {str(e)}")

def query_rag(query: str, vector_store: FAISS, config: Dict[str, str]) -> str:
    """
    RAGパイプライン全体を実行する関数
    """
    try:
        context = retrieve_documents(query, vector_store)
        return generate_response(query, context, config)
    except Exception as e:
        print(f"Error in RAG pipeline: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()