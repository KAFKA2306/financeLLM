import os
import logging
from pathlib import Path
import pypdfium2 as pdfium
from tqdm import tqdm
import json
import unicodedata
import chardet
import re
from typing import List, Dict, Optional, Union
import datetime

# ベクトル化関連のライブラリ
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pandas as pd

class RobustPDFVectorizer:
    """
    PDFからテキストを抽出し、意味的埋め込みを生成するクラス
    """
    def __init__(
        self, 
        base_dir: str = "M:/ML/signatejpx", 
        output_dir: Optional[str] = None,
        model_name: str = "intfloat/multilingual-e5-large",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        ベクトル化プロセッサの初期化
        
        Args:
            base_dir: プロジェクトのルートディレクトリ
            output_dir: 出力ディレクトリ（未指定の場合はbase_dirに作成）
            model_name: 埋め込みモデル名
            chunk_size: テキストチャンクのサイズ
            chunk_overlap: チャンク間のオーバーラップサイズ
        """
        # ディレクトリパスの設定
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "data" / "documents" / "documents"
        self.output_dir = Path(output_dir or self.base_dir) / "output"
        
        # サブディレクトリの作成
        self.raw_dir = self.output_dir / "raw_pdfs"
        self.processed_dir = self.output_dir / "processed_pdfs"
        self.vector_dir = self.output_dir / "vector_store"
        self.log_dir = self.output_dir / "logs"
        
        # ディレクトリ作成
        for dir_path in [self.raw_dir, self.processed_dir, self.vector_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # ロギング設定
        self._setup_logging()
        
        # テキスト分割設定
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", ". ", ", ", " "],
            keep_separator=True
        )
        
        # 埋め込みモデルの初期化
        self.embedding_model = SentenceTransformer(model_name, device='cpu')
        self.logger.info(f"埋め込みモデルを初期化: {model_name}")

    def _setup_logging(self):
        """ロギングの設定"""
        log_file = self.log_dir / "vector_processing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _extract_pdf_text(self, pdf_path: Path) -> Optional[str]:
        """
        PDFからテキストを抽出
        
        Args:
            pdf_path: PDFファイルのパス
        
        Returns:
            抽出されたテキストまたはNone
        """
        try:
            # PDFからテキスト抽出
            pdf = pdfium.PdfDocument(pdf_path)
            
            # ページごとにテキスト抽出
            texts = []
            for page_num, page in enumerate(pdf, 1):
                try:
                    text_page = page.get_textpage()
                    page_text = text_page.get_text_bounded()
                    texts.append(page_text)
                except Exception as page_error:
                    self.logger.warning(f"ページ {page_num} からテキストを抽出できませんでした: {page_error}")
            
            # テキストの結合
            full_text = "\n".join(texts)
            
            # テキストの検証
            if not full_text.strip():
                self.logger.warning(f"有意義なテキストがPDF {pdf_path} から抽出できませんでした")
                return None
            
            return full_text
        
        except Exception as e:
            self.logger.error(f"PDFテキスト抽出エラー {pdf_path}: {e}")
            return None

    def process_pdfs(
        self, 
        pdf_files: Optional[List[Union[str, Path]]] = None,
        ignore_macos_files: bool = True
    ) -> Dict[str, List[Dict[str, Union[str, List[float]]]]]:
        """
        PDFを処理し、ベクトル化を実行
        
        Args:
            pdf_files: 処理するPDFファイルのリスト。Noneの場合は全PDFを処理
            ignore_macos_files: macOSシステムファイルを無視するかどうか
        
        Returns:
            ファイル名をキーとしたベクトル化された文書の辞書
        """
        # 処理するPDFファイルの決定
        if pdf_files is None:
            pdf_files = list(self.input_dir.glob("*.pdf"))
        else:
            pdf_files = [Path(f) for f in pdf_files]
        
        # macOSシステムファイルの除外
        if ignore_macos_files:
            pdf_files = [f for f in pdf_files if not f.name.startswith('._')]
        
        # 結果を格納する辞書
        document_vectors = {}
        
        # PDFの処理
        for pdf_path in tqdm(pdf_files, desc="PDFを処理中"):
            try:
                # テキスト抽出
                text = self._extract_pdf_text(pdf_path)
                
                if not text:
                    self.logger.warning(f"テキスト抽出に失敗: {pdf_path}")
                    continue
                
                # テキストの分割
                text_chunks = self.text_splitter.split_text(text)
                
                # テキストチャンクのベクトル化
                chunk_vectors = self._vectorize_chunks(text_chunks)
                
                # 結果の保存
                document_vectors[pdf_path.name] = chunk_vectors
                
                # チャンクのテキストと埋め込みの保存
                self._save_document_chunks(pdf_path.stem, text_chunks, chunk_vectors)
            
            except Exception as e:
                self.logger.error(f"PDFの処理中にエラー: {pdf_path}, {e}")
        
        # FAISSインデックスの作成と保存
        self._create_faiss_index(document_vectors)
        
        return document_vectors

    def _vectorize_chunks(self, text_chunks: List[str]) -> List[Dict[str, Union[str, List[float]]]]:
        """
        テキストチャンクをベクトル化
        
        Args:
            text_chunks: ベクトル化するテキストチャンク
        
        Returns:
            チャンクのテキストと埋め込みベクトルを含む辞書のリスト
        """
        # ベクトル化
        embeddings = self.embedding_model.encode(text_chunks, show_progress_bar=True)
        
        # チャンクとベクトルをペアにして返す
        return [
            {
                "text": chunk,
                "embedding": embedding.tolist()
            }
            for chunk, embedding in zip(text_chunks, embeddings)
        ]

    def _save_document_chunks(
        self, 
        document_name: str, 
        text_chunks: List[str], 
        chunk_vectors: List[Dict[str, Union[str, List[float]]]]
    ):
        """
        チャンクのテキストと埋め込みを保存
        
        Args:
            document_name: ドキュメント名
            text_chunks: テキストチャンク
            chunk_vectors: チャンクのベクトル情報
        """
        # テキストチャンクの保存
        chunks_text_path = self.processed_dir / f"{document_name}_chunks.txt"
        with open(chunks_text_path, "w", encoding="utf-8") as f:
            for chunk in text_chunks:
                f.write(chunk + "\n\n")
        
        # ベクトルの保存
        chunks_vector_path = self.vector_dir / f"{document_name}_vectors.json"
        with open(chunks_vector_path, "w", encoding="utf-8") as f:
            json.dump(chunk_vectors, f, ensure_ascii=False, indent=2)

    def _create_faiss_index(self, document_vectors: Dict[str, List[Dict[str, Union[str, List[float]]]]]):
        """
        FAISSインデックスの作成と保存
        
        Args:
            document_vectors: 文書のベクトル情報
        """
        # 全ベクトルの抽出
        all_embeddings = []
        embedding_metadata = []
        
        for doc_name, chunks in document_vectors.items():
            for chunk in chunks:
                all_embeddings.append(chunk['embedding'])
                embedding_metadata.append({
                    'document': doc_name,
                    'text': chunk['text']
                })
        
        # ベクトルをNumpy配列に変換
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # FAISSインデックスの作成（L2距離）
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # インデックスの保存
        faiss_index_path = self.vector_dir / "document_index.faiss"
        faiss.write_index(index, str(faiss_index_path))
        
        # メタデータの保存
        metadata_path = self.vector_dir / "index_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(embedding_metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"FAISSインデックスを作成: {len(all_embeddings)}チャンク")

    def retrieve_relevant_documents(self, query: str, k: int = 3) -> List[Dict[str, any]]:
        """
        クエリに関連するドキュメントを検索
        
        Args:
            query: 検索クエリ
            k: 返す関連文書の数
        
        Returns:
            関連する文書の情報
        """
        # クエリのベクトル化
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # FAISSインデックスの読み込み
        faiss_index_path = self.vector_dir / "document_index.faiss"
        metadata_path = self.vector_dir / "index_metadata.json"
        
        index = faiss.read_index(str(faiss_index_path))
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            embedding_metadata = json.load(f)
        
        # 類似度検索
        distances, indices = index.search(query_embedding, k)
        
        # 結果の整形
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            metadata = embedding_metadata[idx]
            results.append({
                'document': metadata['document'],
                'text': metadata['text'],
                'distance': dist
            })
        
        return results

def main():
    """
    ベクトル化プロセスの実行
    """
    try:
        # ベクトライザーの初期化
        vectorizer = RobustPDFVectorizer()
        
        # PDFの処理とベクトル化
        document_vectors = vectorizer.process_pdfs()
        
        # クエリ例の処理
        query = "企業の財務戦略について"
        relevant_docs = vectorizer.retrieve_relevant_documents(query)
        
        # 結果の表示
        print("\n=== 関連文書 ===")
        for doc in relevant_docs:
            print(f"文書: {doc['document']}")
            print(f"テキスト: {doc['text'][:200]}...")
            print(f"距離: {doc['distance']}\n")
    
    except Exception as e:
        logging.error(f"処理中にエラーが発生: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()