# financeLLM — 企業文書RAGのlegacy実験

> **状態: legacy / 再現不能**  
> 財務・企業文書を埋め込み、FAISSで検索する試作コードは残っていますが、現在のdefault branchは依存定義とpath設定が整合しておらず、READMEの手順だけでは再現できません。検索品質、引用精度、数値精度を検証済みのRAG製品ではありません。

## 目的

文書をchunkへ分割し、多言語embeddingとvector searchを使って関連箇所を検索する研究を保存しています。将来的には、回答と参照文書・page・chunkを対応付けることを想定していました。

## 現在確認できるコード

| ファイル | 現在の役割 |
|---|---|
| `vector.py` | txt・md fileの読込、chunk作成、embedding、FAISS保存の試行 |
| `rag.py` / `ragout.py` | 検索・回答生成の試行 |
| `evaluation.py` | オフラインfixtureによる決定論的なRAG評価 |
| `chat.py` | 対話実行の試行 |
| `requirements.txt` | 旧環境の依存候補。現在はそのままinstallできない |

## 再現できない主な理由

### 依存定義

`requirements.txt`には次の問題があります。

- `python==3.8.10`は通常のpip package指定ではない
- `logging==0.5.1.2`は標準libraryと競合する不要な指定
- `intfloat/multilingual-e5-large`はpip package指定ではない
- `vector.py`は`langchain_community`をimportするが、依存定義は旧`langchain==0.0.220`のみ
- PyTorchの`+cpu` wheel取得元が固定されていない

したがって、`pip install -r requirements.txt`を現在有効なquick startとして扱いません。

### 固定path

`vector.py`は既定値として次を使用します。

```text
M:/ML/signatejpx
M:/ML/signatejpx/output/logs
```

repository相対pathではないため、別環境ではそのまま動きません。

### code品質

- import blockが重複している
- `main`関数が重複している
- README旧版はPDF処理を説明していたが、`vector.py`の探索対象は主に`.txt`と`.md`
- 文書page番号を保持する正準schemaを確認できない
- RAG本体の再現可能な依存環境は未整備

## オフラインRAG評価

`evaluation.py` の評価は回答文字数ではなく、固定fixtureと予測JSONを比較します。評価軸は retrieval、document/page/chunk citation、回答中の必須根拠、財務数値、abstention です。財務数値は値だけでなく、通貨・単位・年度・四半期を同じfactとして比較します。

固定回帰テストは外部API・LLM・vector DBを使いません。

```bash
python -m unittest discover -s tests -v
```

fixture評価CLIは次の形式です。

```bash
python evaluation.py \
  --fixture tests/fixtures/rag_eval/fixture.json \
  --predictions path/to/predictions.json \
  --output output/evaluation.json
```

結果JSONには `fixture_revision`、`evaluator_version`、`model_id` とcase別metricsを保存します。同一fixture・同一predictionでは時刻を埋め込まないため、同じ評価結果を再生成できます。

legacy CSVを評価する場合も回答長は採点に使用せず、`ground_truth` 本文との比較を行います。

## 現在できること

- 過去のRAG実装方針を参照する
- chunking、embedding、FAISS保存の試作codeを監査する
- 固定fixtureで検索・引用・財務fact・棄却の評価ロジックを回帰検証する
- 現行基盤へ移植する際の素材として利用する

## 現在できないこと

- READMEだけでRAG本体の環境を再現する
- PDFをpage citation付きで確実に処理する
- fixtureに登録していない生成回答が根拠文書に支持されていると自動保証する
- 財務数値の抽出器自体の正しさを保証する
- model間の性能比較結果を再現する

## 文書と秘密情報

- 第三者PDFを利用条件の確認なくcommitしない
- API keyやtokenをPython fileへ保存しない
- vector storeには原文の断片が含まれるため、機密文書のindexを公開しない
- FAISS indexを信頼できないsourceから読み込まない
- 入力文書、chunk、index、回答の公開範囲を同一とみなさない

## 再開する場合の最小条件

1. Python versionを`pyproject.toml`へ定義し、lock fileを作る
2. repository相対pathまたは設定fileへ移行する
3. PDF parserとpage metadataの契約を定義する
4. document ID、page、chunk ID、score、model、created_atを保存する
5. retrieval、groundedness、citation accuracy、numerical accuracy、abstentionを別々に評価する
6. 小規模fixtureと回帰testを追加する
7. 根拠不足時に回答を拒否する
8. 既存の企業知識・EDINET基盤へ統合する場合は、正準dataの所有境界を明示する

## 金融分析上の注意

生成回答は企業の公式見解、投資助言、売買推奨ではありません。年次と四半期、実績と予想、連結と単体、通貨と単位、訂正前後の文書を分離しなければなりません。

## 関連する監査

- README監査Issue: https://github.com/KAFKA2306/financeLLM/issues/1
- RAG評価Issue: https://github.com/KAFKA2306/financeLLM/issues/3
- 全repository README監査: https://github.com/KAFKA2306/com/issues/3

**README監査日:** 2026年8月5日
