# RAG統合システム - SIGNATEデータ活用チャレンジ

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# 機械学習関連ライブラリ
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# PDF処理
import PyPDF2
from tqdm import tqdm

# API関連
import requests
import urllib3

class IntegratedRAGSystem:
    def __init__(
        self, 
        base_dir: str = "M:/ML/signatejpx", 
        model_name: str = "intfloat/multilingual-e5-large",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        統合RAGシステムの初期化
        
        Args:
            base_dir: プロジェクトのベースディレクトリ
            model_name: 埋め込みモデルの名前
            chunk_size: テキストチャンクのサイズ
            chunk_overlap: チャンク間のオーバーラップサイズ
        """
        # パスの設定
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output"
        
        # ログ設定
        self._setup_logging()
        
        # コンポーネント初期化
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", ". ", ", ", " "],
            keep_separator=True
        )
        
        # 埋め込みモデルの初期化
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 設定の読み込み
        self.config = self._load_config()
        
        # 評価基準の定義
        self.evaluation_criteria = {
            "Perfect": {
                "score": 1.0,
                "description": "質問に対して正確に答え、虚偽の内容が含まれていない回答"
            },
            "Acceptable": {
                "score": 0.5, 
                "description": "質問に対して有用な答えを提供しているが、軽微な誤りが含まれている回答"
            },
            "Missing": {
                "score": 0.0,
                "description": "質問に対して「わかりません」「見つけられませんでした」などの具体的な答えがない回答"
            },
            "Incorrect": {
                "score": -1.0,
                "description": "質問に対して間違った、または関連性のない回答"
            }
        }
        
        self.logger.info("統合RAGシステムの初期化が完了しました")

    def _setup_logging(self):
        """ログ設定の詳細な構成"""
        # ログディレクトリの作成
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # ログファイル名にタイムスタンプ付与
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"integrated_rag_{timestamp}.log"
        
        # ロガーの設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("IntegratedRAGSystem")

    def _load_config(self) -> Dict[str, Any]:
        """
        秘密設定ファイルの読み込み
        
        Returns:
            設定辞書
        """
        try:
            config_path = self.base_dir / "secret" / "config.py"
            
            # 動的に設定ファイルをインポート
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", str(config_path))
            
            if spec is None or spec.loader is None:
                raise ImportError("設定モジュールの読み込みに失敗しました")
                
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            
            # 複数のモデル設定を返す
            return {
                'llama': getattr(config, 'LLAMA_CONFIG', {}),
                'qwen': getattr(config, 'QWEN_CONFIG', {})
            }
            
        except Exception as e:
            self.logger.error(f"設定ファイルの読み込みに失敗: {str(e)}")
            raise

    def load_documents(self) -> List[str]:
        """
        PDFドキュメントの読み込みと前処理
        
        Returns:
            抽出されたテキストのリスト
        """
        # ドキュメントディレクトリの設定
        docs_dir = self.data_dir / "documents" / "documents"
        self.logger.info(f"ドキュメントを読み込んでいます: {docs_dir}")
        
        # PDFファイルの検索
        pdf_files = sorted(docs_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"PDFファイルが見つかりませんでした: {docs_dir}")
        
        # テキスト抽出用リスト
        texts = []
        successful_loads = 0
        failed_loads = 0
        
        # PDFファイルの処理
        for pdf_file in tqdm(pdf_files, desc="PDFドキュメントの読み込み"):
            # システムファイルのスキップ
            if pdf_file.name.startswith('.'):
                continue
            
            try:
                # PDFからテキスト抽出
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    page_texts = [page.extract_text() for page in pdf_reader.pages]
                    full_text = "\n".join(page_texts)
                
                # テキストの保存
                if full_text.strip():
                    texts.append(full_text)
                    successful_loads += 1
                    
                    # 生テキストの保存
                    raw_path = self.output_dir / "raw_data" / f"{pdf_file.stem}.txt"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(raw_path, "w", encoding="utf-8") as f:
                        f.write(full_text)
                else:
                    failed_loads += 1
                    
            except Exception as e:
                self.logger.error(f"PDFの読み込みエラー {pdf_file}: {str(e)}")
                failed_loads += 1
        
        self.logger.info(f"ドキュメント読み込み完了: 成功 {successful_loads}, 失敗 {failed_loads}")
        return texts

    def create_vector_store(self, texts: List[str]) -> FAISS:
        """
        テキストからベクターストアを作成
        
        Args:
            texts: 処理するテキストのリスト
            
        Returns:
            FAISSベクターストア
        """
        self.logger.info("ベクターストアの作成を開始...")
        
        # テキストをドキュメントに変換
        chunks = self.text_splitter.create_documents(texts)
        
        # ベクターストアの作成
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        self.logger.info(f"ベクターストアを作成しました: {len(chunks)} チャンク")
        return vector_store

    def retrieve_documents(self, query: str, vector_store: FAISS, k: int = 3) -> List[str]:
        """
        クエリに関連する文書の検索
        
        Args:
            query: 検索クエリ
            vector_store: 検索対象のベクターストア
            k: 返す関連文書の数
            
        Returns:
            関連文書のリスト
        """
        self.logger.info(f"文書の類似性検索を実行: {query}")
        
        # 類似文書の検索
        relevant_docs = vector_store.similarity_search(query, k=k)
        
        return [doc.page_content for doc in relevant_docs]

    def generate_response(self, query: str, context: List[str], model_config: Dict[str, str]) -> str:
        """
        LLMを使用した回答の生成
        
        Args:
            query: 質問
            context: 関連文書のコンテキスト
            model_config: モデル設定
            
        Returns:
            生成された回答
        """
        # プロンプトの準備
        prompt = f"""以下の文脈に基づいて、質問に答えてください。文脈から明確な答えが見つからない場合は「分かりません」と回答してください。

文脈:
{chr(10).join(context)}

質問: {query}

回答:"""

        # APIリクエストの準備
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": model_config['api_key']
        }
        
        # リクエストデータ
        data = {
            "temperature": 0.7,
            "prompt": prompt,
            "repeat_penalty": 1.0
        }
        
        try:
            # APIリクエストの送信
            response = requests.post(
                model_config['api_url'],
                headers=headers,
                json=data,
                verify=False,
                timeout=30
            )
            
            # レスポンスの処理
            if response.status_code != 200:
                self.logger.error(f"API エラー: ステータス {response.status_code}")
                return "分かりません"
            
            # レスポンスの解析
            result = response.json()
            
            if "response" in result:
                return result["response"]
            elif "generated_text" in result:
                return result["generated_text"]
            
            return str(result)
        
        except Exception as e:
            self.logger.error(f"回答生成中のエラー: {str(e)}")
            return "分かりません"

    def process_queries(self, queries: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        クエリの処理と回答生成
        
        Args:
            queries: 処理するクエリのDataFrame
            
        Returns:
            生成された回答のリスト
        """
        # ドキュメントの読み込みとベクターストアの作成
        texts = self.load_documents()
        vector_store = self.create_vector_store(texts)
        
        # 結果の保存用リスト
        all_results = []
        
        # 各モデルでの処理
        for model_name, model_config in self.config.items():
            model_results = []
            
            # クエリの処理
            for _, row in tqdm(queries.iterrows(), total=len(queries), 
                             desc=f"{model_name}モデルでクエリを処理中"):
                try:
                    query = row['problem']
                    
                    # 関連文書の取得
                    context = self.retrieve_documents(query, vector_store)
                    
                    # 回答の生成
                    response = self.generate_response(query, context, model_config)
                    
                    # 結果の整形
                    result = {
                        'model': model_name,
                        'index': row['index'],
                        'query': query,
                        'response': response,
                        'context': context
                    }
                    
                    model_results.append(result)
                    
                    # 個別の結果を保存
                    self._save_individual_result(result)
                    
                except Exception as e:
                    self.logger.error(f"クエリ {row['index']} の処理中にエラー: {str(e)}")
                    model_results.append({
                        'model': model_name,
                        'index': row['index'],
                        'query': query,
                        'response': "分かりません",
                        'context': []
                    })
            
            all_results.extend(model_results)
            
            # モデルごとの結果を保存
            self._save_model_results(model_name, model_results)
        
        return all_results

    def _save_individual_result(self, result: Dict[str, Any]) -> None:
        """
        個々の結果の保存
        
        Args:
            result: 保存する結果
        """
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"result_{result['model']}_{result['index']}_{timestamp}.json"
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)



    def _save_model_results(self, model_name: str, results: List[Dict[str, Any]]) -> None:
            """
            モデルごとの結果をCSVに保存
            
            Args:
                model_name: モデルの名前
                results: 結果のリスト
            """
            processed_dir = self.output_dir / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 予測結果をDataFrameに変換
            df = pd.DataFrame(results)
            
            # CSVファイルとして保存（ヘッダーなし）
            csv_file = processed_dir / f"predictions_{model_name}_{timestamp}.csv"
            df[['index', 'response']].to_csv(csv_file, index=False, header=False)
            
            self.logger.info(f"{model_name}モデルの結果を {csv_file} に保存しました")

    def evaluate_responses(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成された回答を評価
        
        Args:
            results: 評価する結果のリスト
            
        Returns:
            評価サマリー
        """
        # 評価カテゴリごとの集計
        category_counts = {
            "Perfect": 0,
            "Acceptable": 0,
            "Missing": 0,
            "Incorrect": 0
        }
        
        # 詳細な評価結果の保存用リスト
        detailed_evaluations = []
        
        # 全体のスコア計算
        total_score = 0
        for result in results:
            # 回答の評価
            evaluation = self._evaluate_single_response(result['response'])
            
            # メトリクスの更新
            category = evaluation['rating']
            category_counts[category] += 1
            
            # スコアの累積
            total_score += evaluation['score']
            
            # 詳細評価情報の追加
            detailed_evaluation = {
                **result,
                **evaluation
            }
            detailed_evaluations.append(detailed_evaluation)
        
        # 総合評価の計算
        total_queries = len(results)
        average_score = total_score / total_queries if total_queries > 0 else 0
        
        # 評価サマリーの作成
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": total_queries,
            "average_score": average_score,
            "category_distribution": {
                category: {
                    "count": count,
                    "percentage": count / total_queries * 100 if total_queries > 0 else 0
                }
                for category, count in category_counts.items()
            },
            "evaluation_criteria": self.evaluation_criteria
        }
        
        # 結果の保存
        self._save_evaluation_results(detailed_evaluations, summary)
        
        return summary

    def _evaluate_single_response(self, response: str) -> Dict[str, Any]:
        """
        単一の回答を評価
        
        Args:
            response: 評価する回答
            
        Returns:
            評価結果の辞書
        """
        # 空白または「分かりません」の場合
        if not response or response.strip() in ["分かりません", "見つけられませんでした"]:
            return {
                "rating": "Missing",
                "score": self.evaluation_criteria["Missing"]["score"],
                "reasoning": "具体的な回答が見つかりませんでした"
            }
        
        # 回答の長さに基づく評価
        response_len = len(response)
        if response_len < 10:
            return {
                "rating": "Incorrect",
                "score": self.evaluation_criteria["Incorrect"]["score"],
                "reasoning": "回答が非常に短く、内容が不十分です"
            }
        elif response_len < 50:
            return {
                "rating": "Acceptable",
                "score": self.evaluation_criteria["Acceptable"]["score"],
                "reasoning": "回答は有用ですが、さらなる詳細が望まれます"
            }
        else:
            return {
                "rating": "Perfect",
                "score": self.evaluation_criteria["Perfect"]["score"],
                "reasoning": "質問に対して十分で正確な回答が提供されています"
            }

    def _save_evaluation_results(self, 
                                  detailed_evaluations: List[Dict[str, Any]], 
                                  summary: Dict[str, Any]):
        """
        評価結果の保存
        
        Args:
            detailed_evaluations: 詳細な評価結果
            summary: 評価サマリー
        """
        # 評価結果用ディレクトリの作成
        results_dir = self.output_dir / "evaluation_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 詳細な評価結果のJSON保存
        detailed_results_path = results_dir / f"detailed_evaluations_{timestamp}.json"
        with open(detailed_results_path, "w", encoding="utf-8") as f:
            json.dump(detailed_evaluations, f, ensure_ascii=False, indent=2)
        
        # サマリーのJSON保存
        summary_path = results_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Markdownレポートの生成
        self._generate_markdown_report(summary, results_dir / f"evaluation_report_{timestamp}.md")
        
        self.logger.info(f"評価結果を保存: {detailed_results_path}, {summary_path}")

    def _generate_markdown_report(self, summary: Dict[str, Any], report_path: Path):
        """
        評価サマリーからMarkdownレポートを生成
        
        Args:
            summary: 評価サマリー
            report_path: レポート保存パス
        """
        report_lines = [
            "# RAG評価レポート",
            f"## 評価日時: {summary['timestamp']}",
            "",
            "## 総合評価",
            f"- **総クエリ数**: {summary['total_queries']}",
            f"- **平均スコア**: {summary['average_score']:.2f}",
            "",
            "## カテゴリ別分布",
        ]
        
        # カテゴリごとの詳細
        for category, data in summary['category_distribution'].items():
            report_lines.append(f"### {category}")
            report_lines.append(f"- **件数**: {data['count']}")
            report_lines.append(f"- **割合**: {data['percentage']:.1f}%")
            report_lines.append(f"- **説明**: {summary['evaluation_criteria'][category]['description']}")
            report_lines.append("")
        
        report_lines.append("## 評価基準")
        for category, criteria in summary['evaluation_criteria'].items():
            report_lines.append(f"### {category}")
            report_lines.append(f"- **スコア**: {criteria['score']}")
            report_lines.append(f"- **説明**: {criteria['description']}")
            report_lines.append("")
        
        # レポートをファイルに保存
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

def main():
    """
    メイン実行関数
    RAGシステムの統合プロセスを実行
    """
    try:
        # RAGシステムの初期化
        rag_system = IntegratedRAGSystem()
        
        # クエリの読み込み
        queries = pd.read_csv(rag_system.data_dir / "query.csv")
        rag_system.logger.info(f"{len(queries)}件のクエリを読み込みました")
        
        # クエリの処理と回答生成
        results = rag_system.process_queries(queries)
        
        # 結果の評価
        evaluation_summary = rag_system.evaluate_responses(results)
        
        # 評価サマリーの表示
        print("\n=== 評価サマリー ===")
        print(json.dumps(evaluation_summary, ensure_ascii=False, indent=2))
        
    except Exception as e:
        logging.error(f"処理中に致命的なエラーが発生しました: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()