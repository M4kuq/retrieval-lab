# Retrieval Lab オフラインチュートリアル

このチュートリアルは、ローカルの日本語 Markdown と graded qrels だけで、
BM25 と keyword の検索評価から JSON/CSV/HTML レポートまでを 15 分以内に
確認するためのものです。外部サービス、実モデルのダウンロード、秘密情報は使いません。

## Goal

次のことをひとつの再現可能な実験で確認します。

- 文書 relevance の qrels を使い、keyword と BM25 を同じ chunk と query で比較する。
- Hit Rate、Recall、Precision、MRR、nDCG、AP を cutoff 1 と 3 で読む。
- JSON 結果を保存し、summary CSV、per-query CSV、standalone HTML を生成する。

このサンプルの正解データは `examples/japanese/qrels.jsonl` にあります。評価用の
qrels は検索品質の検証に使うデータなので、実運用では担当者が文書内容に基づいて
レビューした ground truth を用意してください。

## Setup

Python 3.11 以上と Retrieval Lab の開発環境を使います。リポジトリのルートで
次を実行してください。

```console
uv sync --extra dev
```

入力は以下の相対パスです。`retrieval-lab.yaml` の親ディレクトリが基準になります。

```text
examples/japanese/
├── corpus/
│   ├── cache.md
│   ├── evaluation.md
│   ├── retrieval.md
│   └── safety.md
├── qrels.jsonl
└── retrieval-lab.yaml
```

設定は `schema_version: 1`、`recursive_characters` chunker、`native_jsonl` dataset、
keyword と BM25、`top_k: [1, 3]`、seed 42、空の quality gate、JSON/CSV/HTML を
明示しています。相対パスのため、設定ファイルを別の場所へコピーするときは
corpus と qrels の配置も一緒に保ってください。

## Steps

### 1. 入力を検証する

```console
uv run retrieval-lab validate \
  --config examples/japanese/retrieval-lab.yaml
```

これは YAML、corpus、query ID、gold document ID、retriever 構成を検証します。
検索評価はまだ実行しません。

### 2. 評価してレポートを保存する

出力先を一時ディレクトリにすると、リポジトリに生成物を追加せずに試せます。

```console
uv run retrieval-lab run \
  --config examples/japanese/retrieval-lab.yaml \
  --output-dir /tmp/retrieval-lab-japanese
```

設定の `report.formats` に従って、次のファイルが作られます。

```text
/tmp/retrieval-lab-japanese/
├── result.json
├── summary.csv
├── per_query.csv
└── report.html
```

HTML は単一ファイルで、CDN、外部 JavaScript、外部画像を参照しません。結果には
query ごとの metric、検索 latency、p95 の小標本 warning が含まれます。レポートの
表示目的で query 本文や retrieved ID を出力する仕様ではありません。

### 3. 結果を確認・比較する

```console
uv run retrieval-lab inspect /tmp/retrieval-lab-japanese/result.json
uv run retrieval-lab inspect /tmp/retrieval-lab-japanese/result.json --query-id q-metrics
```

自動処理では strict JSON を使えます。

```console
uv run retrieval-lab inspect /tmp/retrieval-lab-japanese/result.json --json
```

baseline と candidate がある場合は、保存済み JSON を再実行せず比較できます。

```console
uv run retrieval-lab compare baseline/result.json /tmp/retrieval-lab-japanese/result.json
```

## Checks

次の小さな確認で、入力が実際に使われていることを確かめられます。

```console
test -s /tmp/retrieval-lab-japanese/result.json
test -s /tmp/retrieval-lab-japanese/summary.csv
test -s /tmp/retrieval-lab-japanese/per_query.csv
test -s /tmp/retrieval-lab-japanese/report.html
```

Python API から同じ実験を呼ぶ場合は `examples/api_quickstart.py` を実行できます。

```console
uv run python examples/api_quickstart.py
```

metrics は query ごとの値を平均した aggregate です。検索の計測は retriever の
search と build/index に限定され、指標計算時間は latency に含まれません。少ない
サンプルでは p95 が不安定なので、latency warning を品質改善の判断と混同しないで
ください。

## Next Steps

- `examples/dense_hybrid_comparison.py` で、ネットワーク不要の deterministic local
  `EmbeddingBackend` を注入した Dense と Hybrid を試す。
- `examples/custom_callable.py` で、既存検索サービスを `CallableRetriever` に接続する。
- `examples/precomputed_ranking.py` で、すでに取得済みの ranking を共通 metric で評価する。
- `notebooks/japanese_bm25_tutorial.ipynb` を上から順に実行する。
- 本番用の qrels は、合成データや tutorial の出力を正式な ground truth とせず、
  独立した relevance レビューで作成する。
