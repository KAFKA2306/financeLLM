# rag.py

import logging
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class RAGSystem:
    def __init__(self, base_dir: str = "M:/ML/signatejpx"):
        """RAGシステムの初期化"""
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "output"
        self.setup_logging()
        
        self.logger.info("Initializing RAG system...")
        
        # 埋め込みモデルの初期化
        self.logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'}
        )
        self.logger.info("Initialization complete")

    def setup_logging(self):
        """詳細なログ設定"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / "logs" / f"rag_system_{timestamp}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("RAGSystem")

    def get_available_sources(self) -> List[str]:
        """利用可能なソースIDの取得"""
        vector_store_dir = self.output_dir / "vector_store"
        if not vector_store_dir.exists():
            self.logger.warning("Vector store directory not found")
            return []
            
        return [p.name for p in vector_store_dir.iterdir() if p.is_dir()]

    def load_vector_store(self, source_id: str) -> Optional[FAISS]:
        """ベクトルストアの読み込み"""
        vector_store_path = self.output_dir / "vector_store" / source_id
        
        if not vector_store_path.exists():
            self.logger.error(f"Vector store not found at {vector_store_path}")
            available_sources = self.get_available_sources()
            if available_sources:
                self.logger.info(f"Available sources: {available_sources}")
            return None
            
        self.logger.info(f"Loading vector store from {vector_store_path}")
        return FAISS.load_local(str(vector_store_path), self.embeddings)



    def load_vector_store(self, source_id: str) -> Optional[FAISS]:
        """ベクトルストアの読み込み"""
        vector_store_path = self.output_dir / "vector_store" / source_id
        
        if not vector_store_path.exists():
            self.logger.error(f"Vector store not found at {vector_store_path}")
            available_sources = self.get_available_sources()
            if available_sources:
                self.logger.info(f"Available sources: {available_sources}")
            return None
            
        self.logger.info(f"Loading vector store from {vector_store_path}")
        return FAISS.load_local(
            str(vector_store_path), 
            self.embeddings, 
            allow_dangerous_deserialization=True  # Add this parameter
        )


    def process_query(self, query: str, source_id: str) -> Dict[str, Any]:
        """クエリの処理"""
        self.logger.info(f"Processing query for source: {source_id}")
        
        # ベクトルストアの読み込み
        vector_store = self.load_vector_store(source_id)
        if vector_store is None:
            return {
                'error': f"Vector store not found for {source_id}",
                'timestamp': datetime.now().isoformat()
            }
        
        # 関連文書の検索
        self.logger.info("Searching for relevant documents...")
        relevant_docs = vector_store.similarity_search(query, k=3)
        
        # 結果の整形
        result = {
            'query': query,
            'source_id': source_id,
            'retrieved_docs': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in relevant_docs
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        # 結果の保存
        self.save_result(result)
        self.logger.info("Query processing complete")
        
        return result

    def save_result(self, result: Dict[str, Any]):
        """結果の保存"""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"rag_result_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved result to {result_file}")

def main():
    """メイン実行関数"""
    try:
        rag = RAGSystem()
        
        # 利用可能なソースの確認
        available_sources = rag.get_available_sources()
        if not available_sources:
            print("No vector stores found. Please run vector.py first to create vector stores.")
            return
            
        print("Available sources:", available_sources)
        
        # テスト用のクエリ実行
        test_query = "テスト用クエリ"
        source_id = available_sources[0]  # 最初のソースを使用
        
        result = rag.process_query(test_query, source_id)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("\nRetrieved documents:")
            for i, doc in enumerate(result['retrieved_docs'], 1):
                print(f"\n{i}. Content preview:")
                print(f"{doc['content'][:200]}...")
                print(f"Metadata: {doc['metadata']}")
        
    except Exception as e:
        logging.error(f"Fatal error in RAG system: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()