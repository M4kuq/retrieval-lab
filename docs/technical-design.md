# Retrieval Lab 実現可能性調査・技術設計

- 状態: 実装開始可能
- 対象: OSS Pythonライブラリ `Retrieval Lab`（仮称）
- v0.1の範囲: RAGの検索部分の比較・評価
- 将来拡張: 取得コンテキストを使った回答生成と回答品質評価
- 正式インターフェース: Python API。CLIはPython APIの薄いアダプターとする

## 1. 結論

実現可能性は高い。BM25、Dense検索、RRFによるHybrid検索、標準的な検索評価指標、Pythonパッケージングは、いずれも安定したOSSと標準仕様で構成できる。v0.1ではLLMを使わず、ローカルかつ決定的に評価できるため、API料金なしで公開・利用できる。

最大の技術リスクは検索処理そのものではなく、次の3点である。

1. 正解文書IDを含む評価データセットを利用者が用意できるか
2. 文書単位の正解とチャンク単位の検索結果を、誤解なく対応付けられるか
3. 異なる検索方式を同じコーパス、同じチャンク、同じクエリで比較できるか

したがって、v0.1ではアルゴリズム数を増やすよりも、データ契約、比較可能性、再現性、CI回帰判定を中核に置く。

## 2. 一次情報による成立性確認

### 2.1 評価方法

- [BEIR](https://github.com/beir-cellar/beir) は、異なる検索方式を共通データセットと共通指標で比較できることを実証している。実装でも nDCG、MAP、Recall、Precision を複数の `k` で計算している。
- [NISTのTREC qrels仕様](https://trec.nist.gov/data/qrels_eng/) は、`query_id / iteration / document_id / relevance` という正解データの基本形を定義している。
- [ir-measures](https://ir-measur.es/en/latest/getting-started.html) は、qrelsと検索runから nDCG、Precision、Judged などをPythonで集計でき、クエリ単位の結果も返せる。
- [Sentence Transformersの評価API](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html) も MRR、nDCG、Recall などを標準的な情報検索指標として採用している。

以上から、`query -> ranked document IDs -> qrelsと照合 -> 指標集計` という評価コアは既知の方式であり、独自の評価式を発明する必要はない。

### 2.2 検索方式

- [BM25S](https://github.com/xhluca/bm25s) は、NumPy中心のPython実装で、インデックスの保存・再読込、上位k件検索を提供する。JavaやPyTorchを必須としないため、ローカルBM25の実装基盤として使える。
- [Sentence Transformers](https://sbert.net/) は埋め込み生成とSemantic Searchを提供し、[Retrieve & Re-Rank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) も公式に説明している。
- [ElasticsearchのRRF仕様](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) は、異なるスコア尺度の複数ランキングを順位だけで統合する式を公開している。BM25スコアとコサイン類似度を直接足さずにHybrid検索を構成できる。
- 日本語BM25には形態素解析が必要になるため、[SudachiPy](https://github.com/WorksApplications/SudachiPy) を任意依存として利用できる。
- Dense検索の初期モデル候補 `intfloat/multilingual-e5-small` は、[公式モデルカード](https://huggingface.co/intfloat/multilingual-e5-small)でMITライセンス、100言語対応が示されている。

### 2.3 回答生成への拡張

[Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) は、検索コンテキストだけでなく、回答のFaithfulnessやAnswer Relevancyなどを評価している。Context Precisionの一部はLLM judgeを利用するため、v0.1の無償・決定的評価とは分離し、将来の任意アダプターにする。

### 2.4 文書読み込みの範囲

PDFは技術的には読み込めるが、[PyMuPDF公式ドキュメント](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)が示すとおり、表示順と抽出順が一致しない場合がある。また[pypdf公式ドキュメント](https://pypdf.readthedocs.io/en/latest/user/extract-text.html)のとおり、画像PDFにはOCRが必要になる。PDF対応をv0.1必須にすると検索評価以前の抽出品質が結果へ混入するため、v0.1コアはTXT、Markdown、JSONLに限定し、PDFは任意拡張とする。

## 3. 既存OSSとの違い

| 既存OSS | 主目的 | Retrieval Labが補う部分 |
| --- | --- | --- |
| BEIR | 公開ベンチマーク上でRetrieverを比較 | 利用者自身の文書、チャンク設定、既存検索APIを同じ手順で評価 |
| MTEB | 埋め込みモデルを多様なタスクで評価 | 埋め込みモデルだけでなくBM25、Hybrid、チャンク条件、レイテンシを比較 |
| ir-measures | qrelsとrunから指標を計算 | 文書読込、インデックス作成、検索実行、レポート、CI判定まで統合 |
| Sentence Transformers Evaluator | Dense/Sparseモデルの評価 | 生文書からの実験管理、複数Retriever、外部検索Adapter、回帰テスト |
| Ragas | RAG全体・回答品質の評価 | v0.1ではLLM不要の検索評価に絞り、無料・決定的・CI向けにする |

新規性は評価式ではなく、次の一連の体験に置く。

`文書 -> 共通チャンク -> BM25/Dense/Hybrid -> 共通qrels評価 -> レイテンシ -> 比較レポート -> CI回帰判定`

## 4. v0.1の責務境界

### 含める

- Python 3.11以上
- Python API
- Python APIだけを呼び出すCLI
- TXT、Markdown、JSONLのコーパス読込
- 決定的なチャンクID生成
- BM25、Dense、BM25+DenseのRRF Hybrid
- 任意の既存検索処理を渡すCallable Adapter
- 文書単位・チャンク単位のrelevance
- HitRate@k、Recall@k、MRR@k、nDCG@k
- Precision@k、MAP@k、Judged@kの任意指定
- 検索レイテンシのmean、p50、p95
- JSON、CSV、単一HTMLレポート
- ベースラインとの差分とCI品質ゲート
- 実験設定、データ、モデル、依存バージョンの記録

### 含めない

- 回答生成
- LLM judge
- 自動的な正解データ生成
- PDF/OCRをコア機能として扱うこと
- Web UI、サーバー、ユーザー管理
- Elasticsearch/Qdrant等の個別SDKをコア依存にすること
- GraphRAG、Agentic RAG
- パラメータの自動最適化
- 本番向け分散インデックス

## 5. 評価の意味を固定するデータ契約

### 5.1 コーパスJSONL

1行1文書とする。`id` と `text` は必須、`metadata` は任意。

```json
{"id":"doc-001","text":"検索対象の本文...","metadata":{"source":"manual.md","section":"認証"}}
```

制約:

- `id` はコーパス内で一意
- 空文字は禁止
- 同じIDで本文が変わった場合はコーパスハッシュが変わる
- TXT/Markdown読込時は、Unicode NFC化し区切り文字を `/` に統一した相対パスを安定した文書IDへ変換する

### 5.2 評価データJSONL

```json
{
  "query_id":"q-001",
  "query":"パスワードを再設定する方法は？",
  "relevant":[
    {"id":"doc-001","relevance":2},
    {"id":"doc-014","relevance":1}
  ],
  "reference_answer":"設定画面から再設定する。",
  "metadata":{"category":"how-to"}
}
```

制約:

- `query_id` は一意
- `relevant` には少なくとも1件の正のrelevanceが必要
- native JSONLの `relevance` は1以上の整数。2以上を使えばnDCGで段階評価できる。非正解を0として列挙する必要はない
- native JSONLでは `relevant` がそのクエリの既知の正解集合であるというclosed-world契約にする
- `reference_answer` はv0.1では任意かつ未使用だが、将来の回答生成評価のため最初から保持する

### 5.3 文書評価とチャンク評価

デフォルトは `relevance_level: document` とする。

検索結果がチャンクの場合、`parent_document_id` で文書へ畳み込み、同じ文書の複数チャンクは最上位のチャンクだけを残す。これにより、1文書を細かく分割したRetrieverが見かけ上有利になることを防ぐ。

`relevance_level: chunk` の場合だけ、正解IDと検索結果IDをチャンクIDで直接照合する。このモードでは、評価データが同一のチャンク設定から生成されていることをハッシュで検証する。

### 5.4 検索結果契約

```python
@dataclass(frozen=True)
class RetrievedItem:
    id: str
    score: float
    parent_document_id: str | None = None
    text: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
```

Retrieverはスコア降順で返す。Runner側でも `(-score, id)` で安定ソートし、同点時の結果を決定的にする。不正な値（NaN、Infinity、重複ID、要求件数を大幅に超える結果）は契約違反として扱う。

## 6. 公開Python API

### 6.1 設定ファイルから実行

```python
from retrieval_lab import EvaluationRunner

runner = EvaluationRunner.from_config("retrieval-lab.yaml")
result = runner.run()

print(result.summary())
result.save_json("artifacts/result.json")
result.save_csv("artifacts")
result.save_html("artifacts/report.html")
```

### 6.2 Pythonだけで設定

```python
from retrieval_lab import (
    BM25RetrieverConfig,
    CorpusConfig,
    DatasetConfig,
    DenseRetrieverConfig,
    EvaluationConfig,
    EvaluationRunner,
    ExperimentConfig,
    HybridRetrieverConfig,
    RecursiveCharacterChunkerConfig,
)

config = ExperimentConfig(
    corpus=CorpusConfig(
        path="./documents",
        chunker=RecursiveCharacterChunkerConfig(
            size=512,
            overlap=64,
        ),
    ),
    dataset=DatasetConfig(
        path="./evaluation.jsonl",
        relevance_level="document",
    ),
    retrievers=[
        BM25RetrieverConfig(name="bm25", tokenizer="sudachi"),
        DenseRetrieverConfig(
            name="dense",
            model="intfloat/multilingual-e5-small",
            normalize_embeddings=True,
            query_prompt="query: ",
            document_prompt="passage: ",
        ),
        HybridRetrieverConfig(
            name="hybrid",
            sources=["bm25", "dense"],
            fusion="rrf",
            rrf_k=60,
            candidate_k=100,
        ),
    ],
    evaluation=EvaluationConfig(
        top_k=[1, 3, 5, 10],
        metrics=["hit_rate", "recall", "mrr", "ndcg"],
    ),
)

result = EvaluationRunner(config).run()
```

### 6.3 既存検索APIを評価

```python
from retrieval_lab import CallableRetriever, RetrievedItem, evaluate_retrievers

def search_production(query: str, top_k: int) -> list[RetrievedItem]:
    rows = existing_search_api(query=query, limit=top_k)
    return [
        RetrievedItem(
            id=row.chunk_id,
            parent_document_id=row.document_id,
            score=row.score,
            text=row.text,
        )
        for row in rows
    ]

result = evaluate_retrievers(
    dataset="./evaluation.jsonl",
    retrievers={"production": CallableRetriever(search_production)},
    top_k=[1, 3, 5, 10],
)
```

この入口ではコーパスやインデックス構築を要求しない。既存Vector DB、社内API、LangChain等は、個別SDKをコアへ追加せず同じ契約で評価できる。

### 6.4 非同期API

外部検索API向けに `AsyncRetriever` と `await runner.arun()` を用意する。`run()` の内部でイベントループを無理に起動しない。同期と非同期を明示的に分け、NotebookやFastAPI内でのイベントループ衝突を避ける。

## 7. Retrieverインターフェース

```python
class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedItem]: ...


class AsyncRetriever(Protocol):
    @property
    def name(self) -> str: ...

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> Sequence[RetrievedItem]: ...
```

Built-in Retrieverのインデックス作成は公開Retriever契約から分離し、内部の `RetrieverFactory` と `IndexBuilder` が担当する。これにより外部Retrieverは不要な `build()` を実装せずに済む。

## 8. 設定ファイル

```yaml
schema_version: 1

experiment:
  name: support-manual-baseline
  seed: 42
  workspace: .retrieval-lab

corpus:
  path: ./documents
  include: ["**/*.md", "**/*.txt"]
  chunker:
    type: recursive_characters
    size: 512
    overlap: 64

dataset:
  path: ./evaluation.jsonl
  format: native_jsonl
  relevance_level: document

retrievers:
  - name: bm25
    type: bm25
    tokenizer: sudachi

  - name: dense
    type: dense
    model: intfloat/multilingual-e5-small
    model_revision: null
    normalize_embeddings: true
    query_prompt: "query: "
    document_prompt: "passage: "
    batch_size: 32

  - name: hybrid
    type: hybrid
    sources: [bm25, dense]
    fusion: rrf
    rrf_k: 60
    candidate_k: 100

evaluation:
  top_k: [1, 3, 5, 10]
  metrics: [hit_rate, recall, mrr, ndcg]
  repetitions: 1
  concurrency: 1

quality_gates:
  - retriever: hybrid
    metric: recall@5
    min_value: 0.75
  - retriever: hybrid
    metric: latency_p95_ms
    max_value: 250

report:
  output_dir: ./artifacts
  formats: [json, csv, html]
```

`schema_version` は必須にする。未知フィールドは原則エラーとし、タイプミスした設定が黙って無視されることを防ぐ。

E5系のようにquery/document prefixを要求するモデルがあるため、Dense設定は両方のpromptを明示的に持つ。モデル名だけから暗黙推論してよいのは既知モデル用presetを選んだ場合だけとし、実際に使ったpromptをmanifestへ残す。

Hybridは、各sourceから最終表示件数より広い `candidate_k` を取得してからRRFする。`candidate_k` は全sourceで共通にし、`max(top_k)` 未満を設定した場合は検証エラーとする。

## 9. 指標仕様

| 指標 | 定義 | 用途 |
| --- | --- | --- |
| HitRate@k | 上位k件に正解が1件以上あれば1 | RAGへ最低1つ根拠を渡せたか |
| Recall@k | 取得した正解数 / 全正解数 | 必要な根拠を取りこぼしていないか |
| MRR@k | 最初の正解順位の逆数 | 最初の有用文書がどれだけ上位か |
| nDCG@k | relevanceと順位を考慮した正規化利得 | 複数正解・段階評価を含む順位品質 |
| Precision@k | 上位k件中の正解割合 | コンテキスト枠の無駄の少なさ |
| MAP@k | 各正解順位までのPrecisionの平均 | 複数正解のランキング全体 |
| Judged@k | 上位k件で判定済みの割合 | qrelsの網羅性確認 |

初期デフォルトは HitRate、Recall、MRR、nDCG とする。PrecisionとMAPは未判定文書を非正解とみなす影響が強いため任意指定にする。TREC形式を読み込む場合はJudged@kを推奨し、未判定率をレポートに表示する。

集計はクエリ単位のmacro averageとし、micro averageは採用しない。正解文書数が多い一部クエリへ全体結果が偏ることを避ける。

## 10. レイテンシ計測仕様

- インデックス構築時間とクエリ検索時間を分離
- 各Retrieverは `max(top_k)` を一度だけ検索し、各kの指標は同じランキングをsliceして計算
- デフォルトは `concurrency=1`、`repetitions=1`
- `perf_counter_ns()` で検索呼出しを計測
- mean、p50、p95、最大値、失敗数を保存
- percentileはnearest-rank法で計算し、20クエリ未満ではp95が不安定である旨を警告
- repetitionsを増やした場合、最初の結果で品質を評価し、各反復の順位一致も検証する
- 外部APIはネットワーク時間込み。Built-in Retrieverと同列の速度比較にするときはレポートへ実行環境差を明示
- 1件でも検索エラーが起きたrunはデフォルトで失敗とし、失敗クエリを黙って分母から除外しない

## 11. 再現性とキャッシュ

キャッシュキーは内容アドレス方式にする。

```text
corpus_hash = hash(normalized documents + loader version)
chunk_hash  = hash(corpus_hash + chunker config + chunker version)
index_hash  = hash(chunk_hash + retriever config + implementation version)
run_hash    = hash(index_hash + dataset hash + evaluation config)
```

ディレクトリ:

```text
.retrieval-lab/
  cache/
    corpora/<corpus_hash>/
    chunks/<chunk_hash>.jsonl
    indexes/<retriever_name>/<index_hash>/
  runs/<timestamp>-<short_hash>/
    manifest.json
    result.json
    summary.csv
    per_query.csv
    report.html
```

`manifest.json` に以下を保存する。

- Retrieval Labのバージョン
- Python、OS、主要依存のバージョン
- 完全な正規化済み設定
- コーパス、チャンク、データセットのハッシュ
- 埋め込みモデルIDとrevision
- seed
- 開始・終了時刻
- 各Retrieverのbuild時間とindexサイズ

日時や絶対パスを除いた結果部分は、同一入力なら同一JSONになるようにする。

結果JSONにも `schema_version` を持たせ、最低限次の構造を固定する。

```json
{
  "schema_version": 1,
  "run": {"id": "...", "manifest": {}},
  "retrievers": {
    "bm25": {
      "metrics": {"recall@5": 0.8, "mrr@10": 0.72},
      "latency": {"mean_ms": 4.2, "p50_ms": 3.8, "p95_ms": 7.1},
      "per_query": []
    }
  },
  "quality_gates": []
}
```

`per_query` はquery ID、各metric、検索順位、タイミング、警告を持つ。本文は別設定で許可された場合だけ含める。

`model_revision` が未指定でも、モデル取得後に解決済みcommit SHAをmanifestへ記録する。再現実験ではそのSHAを設定へ固定するよう警告する。

v0.1のBuilt-in Denseは、埋め込み行列を分割して内積するexact searchとし、対象規模を個人開発・小規模PoC（目安10万チャンクまで）に限定する。それ以上はCallable Adapterで既存Vector DBを利用する。FAISS等のANNは、速度と検索精度のパラメータが新しい比較変数になるためv0.2以降へ分離する。

## 12. 内部アーキテクチャ

```text
src/retrieval_lab/
  __init__.py              # 安定した公開APIだけを再export
  py.typed                 # PEP 561型情報
  api.py                   # EvaluationRunner / evaluate_retrievers
  config.py                # schema_version付きPydantic設定
  errors.py                # 公開例外階層
  models.py                # Document / Chunk / QueryCase / RetrievedItem
  datasets/
    native_jsonl.py
    trec.py
    validation.py
  loaders/
    text.py
    markdown.py
    jsonl.py
  chunkers/
    recursive_characters.py
    ids.py
  retrievers/
    protocols.py
    callable.py
    bm25.py
    dense.py
    hybrid.py
  evaluation/
    runner.py
    ranking.py
    metrics.py
    latency.py
    gates.py
    compare.py
  artifacts/
    cache.py
    manifest.py
    serializers.py
  reports/
    json_report.py
    csv_report.py
    html_report.py
    templates/report.html.j2
  cli/
    app.py
    commands.py
```

依存方向は `cli -> public API -> application services -> domain/protocols` とする。CLIからRetrieverやMetricを直接呼ばない。

## 13. 依存パッケージ方針

Python Packaging User Guideの[`pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)を使用し、重い機能はoptional dependenciesへ分離する。

```toml
[project]
dependencies = [
  "pydantic>=2,<3",
  "pyyaml>=6,<7",
  "typer>=0.16,<1",
  "bm25s>=0.2,<1",
  "ir-measures>=0.4,<1",
  "jinja2>=3.1,<4",
]

[project.optional-dependencies]
dense = ["sentence-transformers>=5,<6"]
ja = ["sudachipy>=0.6,<1", "sudachidict-core"]
pdf = ["pymupdf>=1.26,<2"]
all = [
  "sentence-transformers>=5,<6",
  "sudachipy>=0.6,<1",
  "sudachidict-core",
  "pymupdf>=1.26,<2",
]
```

実装開始時には最新版の互換性をロックファイルとCIで確認し、根拠なく下限・上限を広げない。`dense` を入れない利用者はPyTorchを取得しない。

プロジェクトコードはApache-2.0を第一候補とする。サンプルデータは自作または再配布可能なものだけに限定し、モデル本体は同梱しない。各モデルのID、revision、license metadataをmanifestへ記録する。`Retrieval Lab` と `retrieval-lab` は仮称なので、M5開始前にPyPI名、GitHub名、類似商標を再確認して正式名を確定する。

## 14. CLI設計

```bash
# 雛形生成
retrieval-lab init ./my-eval

# 設定・ID・qrels整合性だけを検証
retrieval-lab validate -c retrieval-lab.yaml

# 評価実行
retrieval-lab run -c retrieval-lab.yaml

# 保存済み結果を比較
retrieval-lab compare baseline/result.json candidate/result.json

# 品質ゲートをCIで判定
retrieval-lab gate candidate/result.json --baseline baseline/result.json
```

終了コード:

- `0`: 成功、品質ゲート通過
- `1`: 品質ゲート不通過
- `2`: CLI使用方法または設定不正
- `3`: 実行時エラー

`run` は必ず公開Python APIを呼び、CLI専用の評価ロジックを持たない。

## 15. 比較可能性のルール

`compare` と `gate` は次が一致しない場合、比較を拒否する。

- dataset hash
- query IDs
- relevance level
- metric名と定義バージョン
- top-k

コーパス、チャンク、Retriever設定は比較したい実験変数なので、不一致を許可するが差分を明示する。同一run内でBM25/Dense/Hybridを比較するときは、同じChunkArtifactを共有する。

## 16. エラー設計

公開例外を次に限定する。

```python
RetrievalLabError
├── ConfigurationError
├── DatasetValidationError
├── CorpusValidationError
├── RetrieverContractError
├── EvaluationError
└── IncomparableRunError
```

エラーには、問題のID、入力箇所、期待値、修正例を含める。1件目で止めず、設定・データ検証では可能な限り複数エラーをまとめて返す。

## 17. セキュリティとプライバシー

- YAMLは`safe_load`相当のみ
- 設定ファイルから任意の `module:function` をimportして実行する機能はv0.1に入れない
- custom RetrieverはPython APIからオブジェクトとして渡す
- コアはURLクロールやクラウドアップロードを行わない
- HTMLへ文書本文やクエリを出す際はエスケープ
- レポートへ全文を含めるかは `include_text` で明示し、デフォルトは先頭の短いpreviewだけ
- APIキーや環境変数の値をmanifestへ記録しない

## 18. 将来の回答生成を壊さない拡張点

v0.1で以下の型だけ先に確保する。

```python
@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    cited_item_ids: tuple[str, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

class AnswerGenerator(Protocol):
    def generate(
        self,
        query: QueryCase,
        contexts: Sequence[RetrievedItem],
    ) -> GeneratedAnswer: ...
```

将来は `retrieve -> generate -> evaluate answer` を追加するが、RetrieverとGeneratorを結合しない。回答評価Metricには必要入力を宣言させる。

```python
class MetricRequirements:
    qrels: bool = False
    reference_answer: bool = False
    generated_answer: bool = False
    retrieved_text: bool = False
    llm_judge: bool = False
```

これにより、Faithfulnessのように回答・コンテキスト・judgeを必要とする指標と、Recallのようにqrelsだけで計算できる指標を同じRunner上で事前検証できる。Ragasは将来のadapterとし、コア依存にはしない。

## 19. テスト設計

### 単体テスト

- 各指標を小さな手計算例で検証
- Recall@kがkの増加で減少しないこと
- すべての正規化指標が0〜1に収まること
- 同点スコアの順序が決定的であること
- 文書単位collapseで同一文書が重複しないこと
- RRF式のgolden test
- ハッシュが入力変更を検出すること

### 契約テスト

- Callable Retrieverが空結果、重複、NaN、不足件数を返した場合
- sync/asyncの双方
- document relevanceとchunk relevanceの取り違えを拒否

### 統合テスト

- 小型の固定コーパスでBM25/Dense/Hybridを実行
- CLIとPython APIが同じresult schemaを生成
- JSONを再読込してcompare/gateできる
- HTMLが外部CDNなしで開く
- ネットワークなしでもBM25評価が完走

### 配布テスト

- wheel/sdist build
- クリーン環境へのwheel install
- `python -c "import retrieval_lab"`
- `retrieval-lab --help`
- Linux、Windows、macOSのCI
- Python 3.11、3.12、3.13

## 20. 実装順序

### M1: 評価コア

- domain models
- native JSONL dataset
- qrels validation
- precomputed runからの指標計算
- JSON result schema

受け入れ条件: 検索処理なしでも、既知runを評価して手計算と一致する。

### M2: Built-in検索

- corpus loaders
- deterministic chunking
- BM25
- Dense exact search
- RRF Hybrid
- content-addressed cache

受け入れ条件: 同じChunkArtifactを使った3方式の比較が1回のrunで完走する。

### M3: 正式Python API

- `EvaluationRunner`
- config models
- Callable/Async Retriever
- save JSON/CSV/HTML

受け入れ条件: READMEのPythonサンプルをそのまま実行できる。

### M4: CLIとCI回帰判定

- init/validate/run/compare/gate
- quality gates
- exit codes
- GitHub Actionsサンプル

受け入れ条件: Recall低下のfixtureでCIが終了コード1になり、改善時は0になる。

### M5: 公開品質

- API reference
- tutorial notebook
- 日本語サンプルデータ
- benchmark/reproducibility docs
- CONTRIBUTING、SECURITY、LICENSE、CHANGELOG
- PyPI Test環境から本番公開

受け入れ条件: 新規環境でドキュメントだけを見て15分以内に最初のHTMLレポートを生成できる。

## 21. v0.1完了条件

- `pip install` 後にimportとCLIの双方が使える
- TXT/Markdown/JSONLから評価できる
- BM25/Dense/Hybridを同じチャンクで比較できる
- 外部RetrieverをPython callableで評価できる
- document/chunk relevanceの挙動がテストとドキュメントで固定されている
- HitRate、Recall、MRR、nDCGがクエリ単位・集計単位で出る
- latency mean/p50/p95とbuild時間が出る
- JSON/CSV/HTMLを生成できる
- baseline比較とquality gateをCIで使える
- オフラインBM25テストが通る
- wheelをクリーン環境にインストールできる
- 主要公開APIに型ヒントとdocstringがある

## 22. 最終判断

Goとする。

ただし、最初から「RAG評価全般」を名乗るとRagas等と競合し、実装範囲も曖昧になる。v0.1の説明は次の一文に固定する。

> Retrieval Lab is a local-first Python toolkit for comparing and regression-testing RAG retrieval strategies on your own corpus.

日本語では次のとおり。

> 自分の文書と正解データを使い、BM25・Dense・Hybrid検索を同じ条件で比較し、改善や劣化をレポートとCIで確認できるローカル優先のPythonライブラリ。

この範囲であれば、無料で開始でき、技術的に成立し、Pythonライブラリとして公開する価値も明確である。
