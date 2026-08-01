# financeLLM — 企業開示文書のRAG分析

**リポジトリ:** https://github.com/KAFKA2306/financeLLM

統合報告書、決算資料、財務文書などをベクトル化し、検索結果を根拠としてLLMへ回答させるRAG（Retrieval-Augmented Generation）研究プロジェクトです。

企業開示から回答を生成するだけでなく、どの文書・ページ・チャンクを参照したかを確認し、検索できなかった内容を推測で補わないことを重視します。

## 主な処理

```text
PDF・文書を読み込む
  → テキストを抽出
  → チャンクへ分割
  → 埋め込みを生成
  → FAISSへ保存
  → 質問に近いチャンクを検索
  → 検索結果をLLMへ渡す
  → 回答と根拠を保存
  → 評価スクリプトで比較
```

## 主なスクリプト

| ファイル | 役割 |
| --- | --- |
| `vector.py` | 文書抽出、チャンク分割、埋め込み、ベクトルストア作成 |
| `rag.py` / `ragout.py` | 関連チャンク検索と回答生成 |
| `evaluation.py` | 回答・検索結果の評価 |
| `chat.py` | 対話形式の実行 |

現在のコードでは、多言語E5系埋め込みモデル、FAISS、複数LLMの利用を想定しています。実際のモデル識別子、必要VRAM、量子化方式はコードと設定を正としてください。

## セットアップ

```bash
git clone https://github.com/KAFKA2306/financeLLM.git
cd financeLLM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windowsでは仮想環境の有効化コマンドを変更してください。

## 入力文書

例:

```text
data/documents/
  company-a-integrated-report.pdf
  company-a-earnings-presentation.pdf
```

文書には著作権、利用条件、個人情報が含まれる可能性があります。公開リポジトリへ第三者PDFを無条件にコミットしないでください。

## 実行

### ベクトルストアを作る

```bash
python vector.py
```

### RAG回答を生成する

```bash
python ragout.py
```

### 評価する

```bash
python evaluation.py
```

実際の引数、入力先、出力先は各スクリプトの現在のCLI定義を確認してください。

## 推奨する出力形式

回答ごとに次を保存します。

```json
{
  "question": "質問",
  "answer": "生成回答",
  "sources": [
    {
      "document": "文書名",
      "page": 12,
      "chunk_id": "...",
      "score": 0.82
    }
  ],
  "model": "使用モデル",
  "created_at": "日時",
  "status": "supported | insufficient_evidence"
}
```

## 検索品質の確認

- 正解ページが上位に検索されるか
- 文書名、ページ番号、チャンクIDが残るか
- 数値や表が途中で分断されていないか
- 年度・四半期・通貨・単位を保持しているか
- 異なる企業のチャンクが混ざっていないか
- 質問に根拠がない場合に回答を拒否できるか

## 回答品質の評価

LLMによる自己採点だけでは品質を証明できません。次を分けて評価します。

1. **Retrieval** — 必要な根拠を取得できたか
2. **Groundedness** — 回答が取得文に支持されているか
3. **Citation accuracy** — 引用先が実際に主張を支えるか
4. **Numerical accuracy** — 数値・単位・期間が正しいか
5. **Completeness** — 質問に必要な項目を満たすか
6. **Abstention** — 根拠不足時に推測しなかったか

## 財務文書特有の注意

- 年次、四半期、累計、単独四半期を混ぜない
- 売上高、受注高、ARR、利益、CFを区別する
- 連結と単体を区別する
- 通貨と単位を保存する
- 実績、会社予想、コンセンサス、独自推計を分ける
- 訂正開示がある場合は新しい版を優先する
- 表の行列構造を壊さない

## 秘密情報

以前のREADMEではAPIキーを`secret/config.py`へ保存する手順がありましたが、Pythonファイルへの平文保存は推奨しません。

```bash
export OPENAI_API_KEY=...
export HUGGINGFACE_TOKEN=...
```

- `.env`を使う場合は`.gitignore`へ追加する
- APIキーをログやNotebook出力へ表示しない
- 漏えいしたキーは削除ではなく失効・再発行する

## 主な構成

```text
financeLLM/
├── data/documents/       # 入力文書
├── output/vector_store/  # ベクトルストア
├── output/results/       # 回答・評価
├── output/logs/          # 実行ログ
├── vector.py
├── rag.py
├── ragout.py
├── evaluation.py
└── chat.py
```

## 現在の位置づけ

本リポジトリはRAGの研究・実験用です。「高精度」「隠れた洞察」などの表現は、比較データセットと再現可能な評価がなければ性能保証になりません。

生成回答は投資助言や企業の公式見解ではありません。

**README最終監査:** 2026-08-01
