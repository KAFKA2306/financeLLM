import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

@dataclass
class LLMConfig:
    name: str
    api_url: str
    api_key: str
    model: str

@dataclass
class EvaluationMetrics:
    perfect_count: int = 0
    acceptable_count: int = 0
    missing_count: int = 0
    incorrect_count: int = 0
    
    def calculate_score(self) -> float:
        total = self.perfect_count + self.acceptable_count + self.missing_count + self.incorrect_count
        if total == 0:
            return 0.0
        return (self.perfect_count + 0.5 * self.acceptable_count - self.incorrect_count) / total

class LLMComparator:
    def __init__(self, base_dir: str = "evaluation_results"):
        """Initialize the LLM comparison system"""
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_logging()
        
        # 評価結果の保存用
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.metrics: Dict[str, EvaluationMetrics] = {}
        
    def setup_directories(self):
        """Create necessary directories for results and logs"""
        dirs = ["logs", "results", "plots", "raw_responses"]
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Configure logging system"""
        log_file = self.base_dir / "logs" / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def generate_response(self, query: str, context: List[str], config: LLMConfig) -> Optional[str]:
        """Generate response from LLM with proper error handling"""
        prompt = f"""以下の文脈に基づいて、質問に答えてください。

文脈:
{chr(10).join(context)}

質問: {query}

回答:"""

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": config.api_key
        }
        
        data = {
            "temperature": 0.7,
            "prompt": prompt,
            "repeat_penalty": 1.0
        }
        
        try:
            response = requests.post(
                config.api_url,
                headers=headers,
                json=data,
                verify=False,
                timeout=30
            )
            
            if response.status_code != 200:
                self.logger.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
            result = response.json()
            
            # Save raw response for analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_response_file = self.base_dir / "raw_responses" / f"{config.name}_{timestamp}.json"
            with open(raw_response_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Parse response based on model type
            if "choices" in result:
                return result["choices"][0]["text"]
            elif "response" in result:
                return result["response"]
            elif "generated_text" in result:
                return result["generated_text"]
            
            self.logger.warning(f"Unexpected response format from {config.name}")
            return str(result)
            
        except Exception as e:
            self.logger.error(f"Error generating response with {config.name}: {str(e)}")
            return None

    def evaluate_response(self, ground_truth: str, response: str) -> Tuple[str, float]:
        """Evaluate response quality"""
        if response is None or response.strip() == "":
            return "Missing", 0.0
            
        if "分かりません" in response or "見つかりません" in response:
            return "Missing", 0.0
            
        # Here you would typically use a more sophisticated evaluation method
        # For demonstration, we'll use a simple length-based heuristic
        response_len = len(response)
        if response_len < 10:
            return "Incorrect", -1.0
        elif response_len < 50:
            return "Acceptable", 0.5
        else:
            return "Perfect", 1.0

    def compare_models(self, queries: List[Dict[str, str]], configs: List[LLMConfig], 
                      contexts: List[List[str]]) -> None:
        """Compare multiple LLM models"""
        for config in configs:
            self.logger.info(f"Evaluating model: {config.name}")
            self.results[config.name] = []
            self.metrics[config.name] = EvaluationMetrics()
            
            for query_data, context in tqdm(zip(queries, contexts), total=len(queries)):
                response = self.generate_response(query_data["question"], context, config)
                rating, score = self.evaluate_response(query_data.get("ground_truth"), response)
                
                result = {
                    "query": query_data["question"],
                    "response": response,
                    "rating": rating,
                    "score": score
                }
                
                self.results[config.name].append(result)
                
                # Update metrics
                if rating == "Perfect":
                    self.metrics[config.name].perfect_count += 1
                elif rating == "Acceptable":
                    self.metrics[config.name].acceptable_count += 1
                elif rating == "Missing":
                    self.metrics[config.name].missing_count += 1
                else:
                    self.metrics[config.name].incorrect_count += 1

    def generate_comparison_report(self) -> str:
        """Generate detailed comparison report"""
        report = ["# LLM Model Comparison Report", ""]
        
        # Overall scores
        report.append("## Overall Scores")
        for model_name, metric in self.metrics.items():
            score = metric.calculate_score()
            report.append(f"\n### {model_name}")
            report.append(f"- Overall Score: {score:.3f}")
            report.append(f"- Perfect: {metric.perfect_count}")
            report.append(f"- Acceptable: {metric.acceptable_count}")
            report.append(f"- Missing: {metric.missing_count}")
            report.append(f"- Incorrect: {metric.incorrect_count}")
            
        # Save report
        report_path = self.base_dir / "results" / f"comparison_report_{datetime.now():%Y%m%d_%H%M%S}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
            
        return "\n".join(report)

    def plot_comparison_results(self):
        """Generate visualization of comparison results"""
        plt.figure(figsize=(12, 6))
        
        models = list(self.metrics.keys())
        scores = [metric.calculate_score() for metric in self.metrics.values()]
        
        plt.bar(models, scores)
        plt.title("Model Comparison - Overall Scores")
        plt.ylabel("Score (-1 to 1)")
        plt.ylim(-1, 1)
        
        # Save plot
        plot_path = self.base_dir / "plots" / f"comparison_plot_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(plot_path)
        plt.close()

def main():
    # サンプルの設定
    configs = [
        LLMConfig(
            name="Llama-2-70B",
            api_url="http://localhost:8000/generate",
            api_key="your_key_here",
            model="llama2-70b"
        ),
        LLMConfig(
            name="Qwen-72B",
            api_url="http://localhost:8001/generate",
            api_key="your_key_here",
            model="qwen-72b"
        )
    ]
    
    # サンプルのクエリとコンテキスト
    queries = [
        {"question": "企業のESG活動について説明してください", "ground_truth": "..."}
        # Add more queries here
    ]
    
    contexts = [
        ["ESGに関する文脈1", "ESGに関する文脈2"]
        # Add more contexts here
    ]
    
    # 比較実行
    comparator = LLMComparator()
    comparator.compare_models(queries, configs, contexts)
    
    # レポート生成
    report = comparator.generate_comparison_report()
    print("\nComparison Report:")
    print(report)
    
    # 可視化
    comparator.plot_comparison_results()
import importlib.util
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class LLMConfig:
    name: str
    api_url: str
    api_key: str
    model: str
    
    @classmethod
    def from_dict(cls, name: str, config_dict: Dict[str, str]) -> 'LLMConfig':
        """設定辞書からLLMConfigインスタンスを作成"""
        return cls(
            name=name,
            api_url=config_dict['api_url'],
            api_key=config_dict['api_key'],
            model=config_dict['model']
        )

def load_config(config_path: str = None) -> Dict[str, List[LLMConfig]]:
    """
    外部設定ファイルから複数のLLM設定を読み込む
    
    Args:
        config_path: 設定ファイルのパス。Noneの場合はデフォルトパスを使用
        
    Returns:
        Dict[str, List[LLMConfig]]: モデルグループごとの設定リスト
    """
    if config_path is None:
        config_path = str(Path('M:/ML/signatejpx/secret/config.py'))
    
    config_path = Path(config_path).resolve()
    print(f"Loading configuration from: {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    # 設定ファイルを動的にインポート
    spec = importlib.util.spec_from_file_location("config", str(config_path))
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load config module")
    
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # 設定を検証して変換
    models_config = {}
    
    # Llamaモデルの設定を読み込み
    if hasattr(config, 'LLAMA_CONFIG'):
        models_config['llama'] = [
            LLMConfig.from_dict("Llama-2-70B", config.LLAMA_CONFIG)
        ]
    
    # Qwenモデルの設定を読み込み
    if hasattr(config, 'QWEN_CONFIG'):
        models_config['qwen'] = [
            LLMConfig.from_dict("Qwen-72B", config.QWEN_CONFIG)
        ]
    
    # 設定の存在を確認
    if not models_config:
        raise ValueError("No valid model configurations found in config file")
    
    return models_config



import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

class LLMComparator:
    def __init__(self, config_path: Optional[str] = None, base_dir: str = "evaluation_results"):
        """
        Initialize the LLM comparison system with external configuration
        
        Args:
            config_path: Path to the configuration file
            base_dir: Base directory for storing results
        """
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_logging()
        
        # 設定ファイルの読み込み
        try:
            self.model_configs = load_config(config_path)
            self.logger.info(f"Loaded configurations for models: {list(self.model_configs.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {str(e)}")
            raise
        
        # 評価結果の保存用
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.metrics: Dict[str, EvaluationMetrics] = {}
    
    def setup_directories(self):
        """Create necessary directories for results and logs"""
        dirs = ["logs", "results", "plots", "raw_responses", "metrics"]
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Configure logging system with rotation and formatting"""
        log_file = self.base_dir / "logs" / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def evaluate_models(self, queries: List[Dict[str, str]], contexts: List[List[str]]) -> None:
        """
        全設定済みモデルを評価
        
        Args:
            queries: 評価用クエリのリスト
            contexts: 各クエリに対応するコンテキストのリスト
        """
        for model_type, configs in self.model_configs.items():
            self.logger.info(f"Evaluating {model_type} models...")
            for config in configs:
                self.logger.info(f"Starting evaluation of {config.name}")
                try:
                    self._evaluate_single_model(config, queries, contexts)
                except Exception as e:
                    self.logger.error(f"Error evaluating {config.name}: {str(e)}")
                    continue

    def _evaluate_single_model(self, config: LLMConfig, 
                             queries: List[Dict[str, str]], 
                             contexts: List[List[str]]) -> None:
        """単一モデルの評価を実行"""
        self.results[config.name] = []
        self.metrics[config.name] = EvaluationMetrics()
        
        for query_data, context in tqdm(zip(queries, contexts), 
                                      desc=f"Evaluating {config.name}",
                                      total=len(queries)):
            response = self.generate_response(query_data["question"], context, config)
            rating, score = self.evaluate_response(query_data.get("ground_truth"), response)
            
            result = {
                "query": query_data["question"],
                "response": response,
                "rating": rating,
                "score": score
            }
            
            self.results[config.name].append(result)
            self._update_metrics(config.name, rating)
            
            # 結果をリアルタイムで保存
            self._save_intermediate_result(config.name, result)

    def _update_metrics(self, model_name: str, rating: str) -> None:
        """評価メトリクスの更新"""
        metrics = self.metrics[model_name]
        if rating == "Perfect":
            metrics.perfect_count += 1
        elif rating == "Acceptable":
            metrics.acceptable_count += 1
        elif rating == "Missing":
            metrics.missing_count += 1
        else:
            metrics.incorrect_count += 1

    def _save_intermediate_result(self, model_name: str, result: Dict[str, Any]) -> None:
        """中間結果の保存"""
        results_dir = self.base_dir / "results" / model_name
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"result_{timestamp}.json"
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def generate_comparison_report(self) -> str:
        """
        詳細な比較レポートの生成
        
        Returns:
            str: マークダウン形式のレポート
        """
        report = ["# LLMモデル比較レポート", "", "## 評価概要", ""]
        report.append("各モデルは以下の基準で評価されました：")
        report.append("- Perfect (1.0): 完全に正確な回答")
        report.append("- Acceptable (0.5): 軽微な誤りを含む有用な回答")
        report.append("- Missing (0.0): 回答なし、または「分かりません」")
        report.append("- Incorrect (-1.0): 不正確または無関係な回答")
        report.append("")
        
        # モデルごとの詳細な結果
        report.append("## モデル別評価結果")
        for model_name, metrics in self.metrics.items():
            score = metrics.calculate_score()
            report.append(f"\n### {model_name}")
            report.append(f"総合スコア: {score:.3f}")
            report.append("")
            report.append("詳細な内訳:")
            report.append(f"- Perfect回答数: {metrics.perfect_count}")
            report.append(f"- Acceptable回答数: {metrics.acceptable_count}")
            report.append(f"- Missing回答数: {metrics.missing_count}")
            report.append(f"- Incorrect回答数: {metrics.incorrect_count}")
            
            # 成功率の計算
            total = sum([metrics.perfect_count, metrics.acceptable_count, 
                        metrics.missing_count, metrics.incorrect_count])
            success_rate = (metrics.perfect_count + metrics.acceptable_count) / total * 100
            report.append(f"有用回答率: {success_rate:.1f}%")
        
        # レポートの保存
        report_path = self.base_dir / "results" / f"comparison_report_{datetime.now():%Y%m%d_%H%M%S}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        
        return "\n".join(report)

def main():
    # 設定ファイルのパスを指定
    config_path = "M:/ML/signatejpx/secret/config.py"
    
    # 評価用データの準備
    queries = [
        {
            "question": "企業のESG活動について説明してください",
            "ground_truth": "企業のESG活動は環境・社会・ガバナンスの3つの観点から..."
        }
        # さらにクエリを追加
    ]
    
    contexts = [
        [
            "ESGは、Environment（環境）、Social（社会）、Governance（企業統治）の頭文字を取った言葉です。",
            "企業のESG活動は、持続可能な社会の実現に向けた取り組みとして重要視されています。"
        ]
        # さらにコンテキストを追加
    ]
    
    try:
        # 比較システムの初期化と実行
        comparator = LLMComparator(config_path=config_path)
        comparator.evaluate_models(queries, contexts)
        
        # レポート生成と表示
        report = comparator.generate_comparison_report()
        print("\n=== 比較レポート ===")
        print(report)
        
        # 可視化の生成
        comparator.plot_comparison_results()
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback
        print(traceback.format_exc())
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import PyPDF2
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import urllib3
from dataclasses import dataclass

@dataclass
class ProjectPaths:
    """プロジェクトの標準ディレクトリ構造を管理するクラス"""
    base_dir: Path
    data_dir: Path
    output_dir: Path
    
    @classmethod
    def create(cls, base_dir: str = "M:/ML/signatejpx") -> 'ProjectPaths':
        """標準的なプロジェクトパス構造を作成"""
        base = Path(base_dir)
        return cls(
            base_dir=base,
            data_dir=base / "data",
            output_dir=base / "output"
        )
    
    def validate(self) -> None:
        """必要なディレクトリの存在を確認し、存在しない場合は作成"""
        # 必須ディレクトリの確認
        required_dirs = [
            self.output_dir / "logs",
            self.output_dir / "raw_data",
            self.output_dir / "processed",
            self.output_dir / "results",
            self.output_dir / "analysis"
        ]
        
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)

class UnifiedRAGSystem:
    def __init__(self, paths: ProjectPaths):
        """
        統一されたディレクトリ構造を持つRAGシステムの初期化
        
        Args:
            paths: プロジェクトのパス構造
        """
        self.paths = paths
        self.paths.validate()
        self.setup_logging()
        
        # 核となるコンポーネントの初期化
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        
        # 設定の読み込み
        self.config = self.load_config()
        self.logger.info("UnifiedRAGSystem initialized successfully")
        
    def setup_logging(self):
        """ログ設定の統一化"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.paths.output_dir / "logs" / f"rag_run_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        try:
            config_path = self.paths.base_dir / "secret" / "config.py"
            self.logger.info(f"Loading configuration from: {config_path}")
            
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", str(config_path))
            if spec is None or spec.loader is None:
                raise ImportError("Failed to load config module")
            
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            
            return {
                'llama': config.LLAMA_CONFIG,
                'qwen': config.QWEN_CONFIG
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
            
    def read_pdf(self, pdf_path: Path) -> Tuple[str, bool]:
        """PDFファイルの読み込みと処理"""
        try:
            text = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in tqdm(pdf_reader.pages, desc=f"Reading {pdf_path.name}"):
                    text.append(page.extract_text())
                    
            # 抽出したテキストを保存
            raw_text_path = self.paths.output_dir / "raw_data" / f"{pdf_path.stem}.txt"
            with open(raw_text_path, "w", encoding="utf-8") as f:
                f.write("\n".join(text))
                
            return "\n".join(text), True
            
        except Exception as e:
            self.logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return "", False
            
    def load_documents(self) -> List[str]:
        """文書の読み込みと前処理"""
        docs_dir = self.paths.data_dir / "documents" / "documents"
        self.logger.info(f"Loading documents from: {docs_dir}")
        
        pdf_files = sorted(docs_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {docs_dir}")
            
        texts = []
        successful_loads = 0
        failed_loads = 0
        
        for pdf_file in pdf_files:
            if pdf_file.name.startswith('.'):  # Macのシステムファイルをスキップ
                continue
                
            text, success = self.read_pdf(pdf_file)
            if success and text:
                texts.append(text)
                successful_loads += 1
            else:
                failed_loads += 1
                
        self.logger.info(f"Document loading completed: {successful_loads} successful, {failed_loads} failed")
        return texts
        
    def process_queries(self, queries: pd.DataFrame, texts: List[str]) -> List[Dict[str, Any]]:
        """クエリの処理と結果の保存"""
        self.logger.info(f"Processing {len(queries)} queries")
        
        # ベクトルストアの作成
        chunks = self.text_splitter.create_documents(texts)
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        results = []
        
        # 各モデルでの評価
        for model_name, model_config in self.config.items():
            model_results = []
            self.logger.info(f"Evaluating with {model_name} model")
            
            for _, row in tqdm(queries.iterrows(), total=len(queries), 
                             desc=f"Processing queries with {model_name}"):
                try:
                    query = row["problem"]
                    # 関連文書の取得
                    context = self.retrieve_documents(query, vector_store)
                    # 応答の生成
                    response = self.generate_response(query, context, model_config)
                    
                    result = {
                        "model": model_name,
                        "index": row["index"],
                        "query": query,
                        "response": response,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    model_results.append(result)
                    
                    # 個別の結果を保存
                    self.save_result(result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing query {row['index']}: {str(e)}")
                    continue
                    
            results.extend(model_results)
            
            # モデルごとの結果をCSVとして保存
            self.save_model_results(model_name, model_results)
            
        return results
        
    def save_result(self, result: Dict[str, Any]) -> None:
        """個別の結果を保存"""
        results_dir = self.paths.output_dir / "results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で保存
        result_file = results_dir / f"result_{result['model']}_{result['index']}_{timestamp}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
    def save_model_results(self, model_name: str, results: List[Dict[str, Any]]) -> None:
        """モデルごとの結果をCSVとして保存"""
        processed_dir = self.paths.output_dir / "processed"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # DataFrameに変換してCSV保存
        df = pd.DataFrame(results)
        csv_file = processed_dir / f"predictions_{model_name}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
    def generate_analysis_report(self, results: List[Dict[str, Any]]) -> None:
        """分析レポートの生成と保存"""
        analysis_dir = self.paths.output_dir / "analysis"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 基本的な分析の実行
        df = pd.DataFrame(results)
        analysis = {
            "total_queries": len(df),
            "queries_per_model": df.groupby("model").size().to_dict(),
            "average_response_length": df.groupby("model")["response"].apply(lambda x: sum(len(r) for r in x)/len(x)).to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 分析結果の保存
        analysis_file = analysis_dir / f"analysis_report_{timestamp}.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

def main():
    try:
        # プロジェクトパスの初期化
        paths = ProjectPaths.create()
        
        # RAGシステムの初期化
        rag_system = UnifiedRAGSystem(paths)
        
        # クエリの読み込み
        queries = pd.read_csv(paths.data_dir / "query.csv")
        rag_system.logger.info(f"Loaded {len(queries)} queries")
        
        # 文書の読み込み
        texts = rag_system.load_documents()
        rag_system.logger.info(f"Loaded {len(texts)} documents")
        
        # クエリの処理と結果の保存
        results = rag_system.process_queries(queries, texts)
        
        # 分析レポートの生成
        rag_system.generate_analysis_report(results)
        
        rag_system.logger.info("Processing completed successfully")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()