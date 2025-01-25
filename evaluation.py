# RAG評価システム

import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class RAGEvaluator:
    def __init__(self, base_dir: str = "M:/ML/signatejpx"):
        """
        RAGシステムの評価を行うクラス
        
        Args:
            base_dir: プロジェクトのベースディレクトリ
        """
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "output"
        self.data_dir = self.base_dir / "data"
        self.setup_logging()
        
        # 評価用の設定
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

    def setup_logging(self):
        """ログ設定"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / "logs" / f"rag_evaluation_{timestamp}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("RAGEvaluator")

    def find_latest_predictions_file(self) -> Optional[Path]:
        """
        最新の予測ファイルを検索
        
        Returns:
            最新の予測ファイルのパス、見つからない場合はNone
        """
        # 可能性のあるディレクトリを検索
        search_dirs = [
            self.output_dir / "processed",
            self.output_dir / "results",
            self.data_dir / "sample_submit" / "sample_submit"
        ]
        
        prediction_files = []
        for search_dir in search_dirs:
            if search_dir.exists():
                prediction_files.extend(search_dir.glob("predictions*.csv"))
        
        # タイムスタンプで最新のファイルを選択
        if prediction_files:
            latest_file = max(prediction_files, key=lambda f: f.stat().st_mtime)
            self.logger.info(f"Found predictions file: {latest_file}")
            return latest_file
        
        self.logger.warning("No predictions file found")
        return None

    def evaluate_responses(self, 
                            predictions_path: Optional[Path] = None, 
                            ground_truth_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        予測された回答を評価する
        
        Args:
            predictions_path: 予測回答のCSVファイルパス（オプション）
            ground_truth_path: 正解データのCSVファイルパス（オプション）
        
        Returns:
            評価結果の詳細情報
        """
        self.logger.info("Starting response evaluation")
        
        # 予測ファイルのパスが指定されていない場合は自動検索
        if predictions_path is None:
            predictions_path = self.find_latest_predictions_file()
            
        if predictions_path is None:
            raise FileNotFoundError("予測ファイルが見つかりませんでした。パスを確認してください。")
        
        # 正解データのデフォルトパス設定
        if ground_truth_path is None:
            ground_truth_path = self.data_dir / "validation" / "ans_txt.csv"
        
        # 予測回答の読み込み
        try:
            predictions_df = pd.read_csv(predictions_path, header=None, names=['index', 'response'])
        except Exception as e:
            self.logger.error(f"予測回答の読み込みに失敗: {e}")
            raise
        
        # 正解データの読み込み（オプション）
        ground_truth_df = None
        if ground_truth_path and ground_truth_path.exists():
            try:
                ground_truth_df = pd.read_csv(ground_truth_path)
            except Exception as e:
                self.logger.warning(f"正解データの読み込みに失敗: {e}")
        
        # 評価結果を格納するリスト
        evaluation_results = []
        
        # 評価プロセス
        for _, row in predictions_df.iterrows():
            evaluation = self._evaluate_single_response(row['response'])
            
            # インデックス情報の追加
            evaluation['index'] = row['index']
            
            # 正解データがある場合は追加情報を付与
            if ground_truth_df is not None:
                ground_truth_row = ground_truth_df[ground_truth_df['index'] == row['index']]
                if not ground_truth_row.empty:
                    evaluation['ground_truth'] = ground_truth_row['ground_truth'].values[0]
            
            evaluation_results.append(evaluation)
        
        # 評価サマリーの生成
        summary = self._generate_evaluation_summary(evaluation_results)
        
        # 結果の保存
        self._save_evaluation_results(evaluation_results, summary)
        
        return summary

    def _evaluate_single_response(self, response: str) -> Dict[str, Any]:
        """
        単一の回答を評価する（簡易的な評価ロジック）
        
        Args:
            response: 評価する回答
        
        Returns:
            評価結果の辞書
        """
        # 回答が空白または「分かりません」の場合
        if not response or response.strip() in ["分かりません", "見つけられませんでした"]:
            return {
                "rating": "Missing",
                "score": self.evaluation_criteria["Missing"]["score"],
                "reasoning": "具体的な回答が見つかりませんでした"
            }
        
        # 回答長に基づく簡易評価
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

    def _generate_evaluation_summary(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        評価結果から総合的なサマリーを生成
        
        Args:
            evaluation_results: 個々の回答の評価結果
        
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
        
        total_score = 0
        for result in evaluation_results:
            rating = result['rating']
            category_counts[rating] += 1
            total_score += result.get('score', 0)
        
        # 総合スコアの計算
        total_queries = len(evaluation_results)
        average_score = total_score / total_queries if total_queries > 0 else 0
        
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
        
        return summary

    def _save_evaluation_results(self, 
                                  evaluation_results: List[Dict[str, Any]], 
                                  summary: Dict[str, Any]):
        """
        評価結果と要約を保存
        
        Args:
            evaluation_results: 個々の回答の評価結果
            summary: 評価サマリー
        """
        # 個別の評価結果を保存
        results_dir = self.output_dir / "evaluation_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 詳細結果のJSON保存
        detailed_results_path = results_dir / f"detailed_results_{timestamp}.json"
        with open(detailed_results_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        # サマリーのJSON保存
        summary_path = results_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # サマリーのMarkdown生成
        self._generate_markdown_report(summary, results_dir / f"evaluation_report_{timestamp}.md")
        
        self.logger.info(f"評価結果を保存しました: {detailed_results_path}, {summary_path}")

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
        
        for category, data in summary['category_distribution'].items():
            report_lines.append(f"### {category}")
            report_lines.append(f"- **件数**: {data['count']}")
            report_lines.append(f"- **割合**: {data['percentage']:.1f}%")
            report_lines.append(f"- **説明**: {self.evaluation_criteria[category]['description']}")
            report_lines.append("")
        
        report_lines.append("## 評価基準")
        for category, criteria in self.evaluation_criteria.items():
            report_lines.append(f"### {category}")
            report_lines.append(f"- **スコア**: {criteria['score']}")
            report_lines.append(f"- **説明**: {criteria['description']}")
            report_lines.append("")
        
        # レポートをファイルに保存
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

def main():
    """評価システムのメイン実行関数"""
    try:
        # 評価器の初期化
        evaluator = RAGEvaluator()
        
        # 予測結果と正解データを自動検索
        evaluation_summary = evaluator.evaluate_responses()
        
        # サマリーの表示
        print("\n=== 評価サマリー ===")
        print(json.dumps(evaluation_summary, ensure_ascii=False, indent=2))
        
    except Exception as e:
        logging.error(f"評価プロセスでエラーが発生しました: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()