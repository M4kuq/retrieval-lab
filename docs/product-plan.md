# Retrieval Lab 開発マスタープラン

## 1. プロジェクト概要

複数の検索方式を同一条件で実行し、検索精度・速度・リソース消費を簡単に比較できるPythonライブラリを開発する。

仮称は `retrieval-lab` とする。開発開始時にGitHubおよびPyPI上の名称利用可否を確認し、競合している場合は代替名を選定する。

本プロジェクトでは、CLIツールを作ることだけを目的としない。

Pythonコードから以下のように利用できる、組み込み可能な評価SDKとして設計する。

    from retrieval_lab import EvaluationRunner

    result = EvaluationRunner.from_config(
        "retrieval-lab.yaml"
    ).run()

    print(result.summary())
    result.save_html("report.html")

検索評価を最初の完成範囲とし、将来的にはRAGの回答生成、引用、根拠性、回答品質まで一貫して評価できるライブラリへ拡張する。

# 2. プロダクトの定義

本ライブラリを以下のように定義する。

> 文書、評価クエリ、正解データを渡すだけで、複数の検索方式を同一条件で実行し、検索精度・速度・リソース消費を比較できるPython評価ライブラリ。

利用者は、検索方式ごとの評価処理を個別に実装する必要がない。

ライブラリ側で以下を実行する。

1.  文書の読み込み
2.  文書のチャンク分割
3.  各検索方式のインデックス作成
4.  評価クエリによる検索
5.  評価指標の計算
6.  レイテンシなどの計測
7.  結果オブジェクトの生成
8.  JSON、CSV、HTMLなどのレポート出力

# 3. インターフェースの優先順位

本ライブラリのインターフェースは、以下の優先順位で設計する。

1.  Python API
2.  Python APIを使用するCLI
3.  Jupyter Notebook向け表示
4.  外部フレームワーク向けAdapter
5.  公開Webデモ

Python APIを正式なApplication APIとする。

CLI専用の評価処理を別に実装してはならない。CLIはPython APIを呼び出す薄いAdapterとして実装する。

# 4. 想定利用者

主な対象は以下とする。

- RAGを開発している個人開発者
- 検索方式を選定したいAIエンジニア
- BM25、Dense、Hybridの違いを比較したい学習者
- RAGの改善前後を定量比較したい開発チーム
- 検索品質の回帰テストをCIへ組み込みたい開発者
- 既存の検索APIやVector Databaseを評価したい開発者
- Notebook上で検索実験を行いたい研究者・エンジニア

個人開発から小規模な実務PoCまでを中心対象とする。

# 5. 主要な利用方法

## 5.1 設定ファイルから実行

    from retrieval_lab import EvaluationRunner

    runner = EvaluationRunner.from_config(
        "retrieval-lab.yaml"
    )

    result = runner.run()

    print(result.summary())
    result.save_json("reports/result.json")
    result.save_html("reports/result.html")

## 5.2 Pythonコードだけで設定

    from retrieval_lab import (
        BM25RetrieverConfig,
        CorpusConfig,
        DatasetConfig,
        DenseRetrieverConfig,
        EvaluationConfig,
        EvaluationRunner,
        HybridRetrieverConfig,
    )

    config = EvaluationConfig(
        corpus=CorpusConfig(
            path="./documents",
            chunk_size=512,
            chunk_overlap=64,
        ),
        dataset=DatasetConfig(
            path="./evaluation.jsonl",
            relevance_level="document",
        ),
        retrievers=[
            BM25RetrieverConfig(
                name="bm25",
            ),
            DenseRetrieverConfig(
                name="dense",
                model="default-multilingual-model",
            ),
            HybridRetrieverConfig(
                name="hybrid",
                retrievers=["bm25", "dense"],
                fusion="rrf",
            ),
        ],
        top_k=[1, 3, 5, 10],
        seed=42,
    )

    result = EvaluationRunner(config).run()

## 5.3 Pythonオブジェクトを直接渡す

ファイルを作成しなくても評価できること。

    from retrieval_lab import (
        Document,
        EvaluationQuery,
        EvaluationRunner,
    )

    documents = [
        Document(
            id="doc-1",
            text="AWS Secrets Managerは機密情報を管理するサービスです。",
            metadata={"category": "aws"},
        ),
        Document(
            id="doc-2",
            text="Amazon S3はオブジェクトストレージサービスです。",
            metadata={"category": "aws"},
        ),
    ]

    queries = [
        EvaluationQuery(
            id="q-1",
            query="AWSで機密情報を管理するサービスは何ですか",
            relevant_document_ids={"doc-1"},
        ),
    ]

    result = EvaluationRunner.quick_evaluate(
        documents=documents,
        queries=queries,
        strategies=["keyword", "bm25"],
        top_k=[1, 3],
    )

    print(result.metrics["bm25"].recall_at(1))

この利用方法は、以下での組み込みを想定する。

- pytest
- Jupyter Notebook
- FastAPI
- Streamlit
- 既存のRAGアプリ
- バッチ処理
- CI
- 社内検索システム

## 5.4 CLIから実行

    retrieval-lab evaluate \
      --documents ./documents \
      --dataset ./evaluation.jsonl \
      --strategies bm25,dense,hybrid \
      --output ./reports

内部ではPython APIと同一の処理を使用する。

## 5.5 既存の検索結果だけを評価

すでに別システムで検索が完了している場合、インデックス作成や検索を行わず、検索結果のみを評価できること。

    from retrieval_lab import (
        EvaluationDataset,
        RetrievedQueryResult,
        evaluate_results,
    )

    dataset = EvaluationDataset.from_jsonl(
        "evaluation.jsonl"
    )

    retrieved_results = [
        RetrievedQueryResult(
            query_id="q-1",
            retrieved_document_ids=[
                "doc-3",
                "doc-1",
                "doc-2",
            ],
        ),
    ]

    result = evaluate_results(
        dataset=dataset,
        retrieved_results=retrieved_results,
        top_k=[1, 3, 5],
    )

これにより、v0.1の時点から以下の検索システムを評価可能にする。

- Elasticsearch
- OpenSearch
- Qdrant
- pgvector
- Pinecone
- LangChain
- LlamaIndex
- 独自検索API
- 社内検索システム

各サービス専用Adapterがなくても、標準結果形式へ変換すれば評価できることを重視する。

# 6. 設計原則

## 6.1 Python APIファースト

Python APIを中心にドメインモデル、評価エンジン、レポート処理を設計する。

CLIはPython APIを呼び出す。

以下のような構造は禁止する。

- CLIへビジネスロジックを直接実装する
- CLIとPython APIで別々の評価処理を作る
- CLI実装後にPython APIを後付けする
- CLIでしか利用できない機能を作る

## 6.2 ローカルファースト

基本機能は、有料APIを使用せずローカル環境で完結させる。

- 文書を外部へ自動送信しない
- テレメトリを送信しない
- LLM APIを必須にしない
- Dense検索をローカルモデルで実行可能にする
- ネットワークアクセスが発生する場合は明示する
- APIキーをコードや設定例へ埋め込まない

## 6.3 軽量なコア

コア機能をLangChainやLlamaIndexなどの大規模フレームワークへ依存させない。

外部フレームワークとの連携は、後続バージョンでAdapterまたはOptional Dependencyとして追加する。

## 6.4 拡張可能な構造

実際に複数実装が必要な以下の箇所は、Protocolまたは抽象基底クラスとして分離する。

- DocumentLoader
- Chunker
- Retriever
- Metric
- Reporter
- Generator
- Judge

ただし、将来使うかもしれないという理由だけで抽象化しない。

未使用のインターフェースや空実装を大量に作ることは禁止する。

## 6.5 再現可能性

評価結果には以下を記録する。

- データセット識別情報
- 文書数
- チャンク数
- チャンク設定
- 検索方式
- Embeddingモデル名
- モデルのリビジョン
- Top-K
- 乱数シード
- 実行環境
- Pythonバージョン
- ライブラリバージョン
- 実行日時
- 設定内容

同一条件で評価を再実行できることを重視する。

## 6.6 評価結果の正直な表示

正解データがない場合、検索精度を評価したとは表現しない。

正解データがないモードでは、以下のみを提供する。

- 検索結果の比較
- レイテンシ比較
- インデックス作成時間
- 人間による結果確認

Synthetic Datasetを利用した場合は、以下を明示する。

    Dataset Type: Synthetic
    Human Reviewed: No
    Evaluation Reliability: Experimental

# 7. v0.1のスコープ

v0.1では、検索部分の定量評価を完成させる。

## 7.1 対応する検索方式

### Keyword Baseline

- 完全一致
- 部分一致
- 正規表現
- 単純な文字列一致スコア

高度な検索方式ではなく、比較用ベースラインとして扱う。

### BM25

一般的なキーワード検索方式として実装する。

日本語を含む文書でも利用できるよう、Tokenizerを差し替え可能にする。

### Dense Retrieval

Sentence Transformers互換のEmbeddingモデルを使用する。

デフォルトモデルは以下の条件で選定する。

- CPUで実行可能
- 日本語または多言語に対応
- ライセンスが明確
- 過度に巨大ではない
- 設定で変更可能
- モデルリビジョンを記録可能

### Hybrid Retrieval

BM25とDense Retrievalの結果を統合する。

v0.1ではReciprocal Rank Fusionを標準方式とする。

重み付きスコア統合は後続バージョンの候補とする。

## 7.2 評価単位

以下の両方をサポートする。

- 文書単位
- チャンク単位

文書単位の正解データが指定された場合、その文書に属するチャンクが取得されれば正解として扱えるようにする。

評価単位は設定およびレポートへ明示する。

## 7.3 評価指標

最低限、以下を実装する。

- Hit Rate@k
- Recall@k
- Precision@k
- MRR
- nDCG@k
- MAP
- 平均検索レイテンシ
- P50検索レイテンシ
- P95検索レイテンシ
- インデックス作成時間
- インデックスサイズ

各指標には、小さな固定データを使った手計算との一致テストを作成する。

リソース計測はOSや実行環境による差が大きいため、取得できない項目があっても評価全体を失敗させない。

# 8. 公開データモデル

## 8.1 Document

    from retrieval_lab import Document

    document = Document(
        id="doc-1",
        text="検索対象となる本文",
        metadata={
            "category": "aws",
            "source": "manual",
        },
    )

最低限、以下を保持する。

    id
    text
    metadata
    source

## 8.2 Chunk

最低限、以下を保持する。

    id
    document_id
    text
    start_offset
    end_offset
    metadata

文書IDとチャンクIDは明確に区別する。

## 8.3 EvaluationQuery

    from retrieval_lab import EvaluationQuery

    query = EvaluationQuery(
        id="q-1",
        query="AWSで秘密情報を管理するサービスは何か",
        relevant_document_ids={"security-doc"},
        relevant_chunk_ids=set(),
    )

## 8.4 SearchResult

    document_id
    chunk_id
    text
    score
    rank
    metadata

検索方式によってスコア尺度が異なる可能性があるため、異なるRetrieverの生スコアを直接比較可能とは扱わない。

## 8.5 EvaluationResult

`run()`の戻り値は、辞書ではなく型付きの結果オブジェクトとする。

    result = runner.run()

    print(result.run_id)
    print(result.created_at)
    print(result.environment)
    print(result.metrics)
    print(result.query_results)
    print(result.failures)

以下の変換機能を提供する。

    result.to_dict()
    result.to_json()
    result.save_json("result.json")
    result.save_csv("result.csv")
    result.save_html("result.html")

`pandas`が利用可能な場合は、Optional Dependencyとして以下を提供してよい。

    dataframe = result.to_dataframe()

# 9. 入力形式

## 9.1 文書

v0.1では以下を対象とする。

- `.txt`
- `.md`
- `.jsonl`
- Pythonの`Document`オブジェクト

以下はv0.1の必須対象外とする。

- PDF
- Word
- OCR
- Webクローリング

## 9.2 評価データセット

JSONLを標準形式とする。

    {
      "query_id": "q001",
      "query": "AWSで秘密情報を管理するサービスは何か",
      "relevant_document_ids": ["security-doc"],
      "relevant_chunk_ids": [],
      "relevance_grades": {
        "security-doc": 3
      }
    }

必須項目は以下とする。

- `query_id`
- `query`
- `relevant_document_ids`または`relevant_chunk_ids`

段階的関連度は任意とする。

Pythonオブジェクトによる入力も正式にサポートする。

# 10. チャンク分割

v0.1では以下を提供する。

- 文字数ベース
- Token数ベース
- Markdown見出し単位

設定可能な項目は以下とする。

- Chunk Size
- Chunk Overlap
- Tokenizer
- Metadata引き継ぎ
- 空チャンク除外

検索方式を比較するときは、原則として同一チャンクを使用する。

チャンク方法の比較は、検索方式の比較とは別の実験軸として扱う。

# 11. Python公開API

利用者が使用してよいAPIを、パッケージ直下から明示的に公開する。

    from retrieval_lab import (
        BM25Retriever,
        Chunk,
        DenseRetriever,
        Document,
        EvaluationConfig,
        EvaluationQuery,
        EvaluationResult,
        EvaluationRunner,
        HybridRetriever,
        SearchResult,
    )

以下のような内部モジュールへの直接依存を、READMEやサンプルでは使用しない。

    # 使用禁止
    from retrieval_lab.internal.engine.runner_impl import RunnerImpl

公開APIは以下で管理する。

- `retrieval_lab/__init__.py`
- `retrieval_lab/public_api.py`
- `__all__`
- APIリファレンス
- Semantic Versioning
- Deprecation Warning

すべての公開APIへ型注釈を付ける。

パッケージには`py.typed`を含める。

# 12. Retrieverの個別利用

評価パイプライン全体だけでなく、Retrieverを単独でも利用できること。

    from retrieval_lab import BM25Retriever, Document

    documents = [
        Document(
            id="doc-1",
            text="RAGでは検索品質の評価が重要です。",
        ),
        Document(
            id="doc-2",
            text="Pythonではpytestを利用できます。",
        ),
    ]

    retriever = BM25Retriever()
    retriever.index(documents)

    results = retriever.search(
        query="RAGの評価",
        top_k=5,
    )

    for item in results:
        print(
            item.document_id,
            item.score,
            item.text,
        )

# 13. 評価指標の個別利用

    from retrieval_lab.metrics import (
        recall_at_k,
        reciprocal_rank,
    )

    retrieved_ids = [
        "doc-3",
        "doc-1",
        "doc-2",
    ]

    relevant_ids = {
        "doc-1",
        "doc-4",
    }

    recall = recall_at_k(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        k=3,
    )

    mrr = reciprocal_rank(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
    )

Metric関数の入力形式と、ゼロ件時の動作を明確に定義する。

# 14. カスタムRetriever

利用者が独自検索処理を標準評価エンジンへ接続できるようにする。

    from collections.abc import Sequence

    from retrieval_lab import (
        BaseRetriever,
        Chunk,
        SearchResult,
    )

    class CompanySearchRetriever(BaseRetriever):
        name = "company-search"

        def index(
            self,
            chunks: Sequence[Chunk],
        ) -> None:
            self._chunks = list(chunks)

        def search(
            self,
            query: str,
            top_k: int,
        ) -> list[SearchResult]:
            return [
                SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=1.0,
                    rank=index + 1,
                    metadata=chunk.metadata,
                )
                for index, chunk in enumerate(
                    self._chunks[:top_k]
                )
            ]

標準評価エンジンへ直接渡せること。

    runner = EvaluationRunner(
        documents=documents,
        queries=queries,
        retrievers=[
            CompanySearchRetriever(),
        ],
    )

    result = runner.run()

カスタムRetrieverを実装するために、非公開モジュールへ依存する必要がない構造にする。

# 15. 設定ファイル

YAMLを標準設定形式とする。

    corpus:
      path: ./documents
      format: auto

    chunking:
      strategy: token
      chunk_size: 512
      chunk_overlap: 64

    dataset:
      path: ./evaluation.jsonl
      relevance_level: document

    retrievers:
      - name: keyword
        type: keyword

      - name: bm25
        type: bm25

      - name: dense
        type: dense
        model: default-multilingual-model

      - name: hybrid
        type: hybrid
        retrievers:
          - bm25
          - dense
        fusion: rrf

    evaluation:
      top_k:
        - 1
        - 3
        - 5
        - 10
      seed: 42

    output:
      directory: ./reports
      formats:
        - json
        - csv
        - html

設定値は型付きモデルで検証する。

不正な設定には、修正方法が理解できるエラーメッセージを返す。

# 16. CLI設計

最低限、以下を実装する。

    retrieval-lab init
    retrieval-lab validate
    retrieval-lab evaluate
    retrieval-lab compare
    retrieval-lab inspect
    retrieval-lab dataset convert

## init

設定ファイルとサンプルデータを生成する。

    retrieval-lab init ./my-evaluation

## validate

文書、設定、評価データの整合性を確認する。

    retrieval-lab validate \
      --config retrieval-lab.yaml

以下を検出する。

- 存在しない正解文書ID
- 存在しない正解チャンクID
- query_idの重複
- 空クエリ
- 空文書
- 文書IDの重複
- 不正な設定値
- 評価不能なデータ

## evaluate

    retrieval-lab evaluate \
      --config retrieval-lab.yaml

## compare

    retrieval-lab compare \
      baseline.json \
      candidate.json

検索精度が指定値以上低下した場合、非ゼロ終了コードを返せるようにする。

## inspect

    retrieval-lab inspect \
      report.json \
      --query-id q001

# 17. CLIとPython APIの関係

CLIは以下のようにPython APIを呼び出す構造とする。

    def evaluate_command(
        config_path: str,
    ) -> int:
        try:
            result = EvaluationRunner.from_config(
                config_path
            ).run()

            result.print_summary()
            return 0

        except RetrievalLabError as exc:
            render_error(exc)
            return 1

以下はCLIとPython APIで完全に共通化する。

- 設定検証
- 文書読み込み
- チャンク分割
- インデックス作成
- 検索
- 指標計算
- 結果比較
- レポート生成

同一条件で実行した場合、CLIとPython APIの結果が一致することをテストする。

# 18. レポート

以下の形式を提供する。

- Pythonの`EvaluationResult`
- Terminal
- JSON
- CSV
- 静的HTML

HTMLレポートには最低限、以下を含める。

- 実行条件
- 検索方式ごとの指標比較
- レイテンシ比較
- クエリごとの検索結果
- 正解を取得できなかったクエリ
- 検索方式間で差が大きいクエリ
- 設定内容
- データセットの信頼性情報

推奨方式を表示する場合、単一指標だけで断定しない。

以下のように分ける。

- Recall重視の場合
- レイテンシ重視の場合
- 精度と速度のバランスを重視する場合

# 19. ログと進捗

ライブラリ内部では`print`を使用しない。

標準の`logging`を利用し、ログ設定は利用者が制御できるようにする。

長時間処理では、進捗Callbackを指定できる構造にする。

    def on_progress(event):
        print(
            event.stage,
            event.completed,
            event.total,
        )

    runner = EvaluationRunner.from_config(
        "retrieval-lab.yaml",
        progress_callback=on_progress,
    )

    result = runner.run()

進捗表示ライブラリへの依存を、コアへ固定しない。

# 20. 例外設計

ライブラリ独自の例外階層を提供する。

    from retrieval_lab.exceptions import (
        ConfigurationError,
        DatasetValidationError,
        EvaluationError,
        IndexingError,
        OptionalDependencyError,
        RetrievalLabError,
    )

利用例は以下とする。

    from retrieval_lab import EvaluationRunner
    from retrieval_lab.exceptions import (
        RetrievalLabError,
    )

    try:
        result = EvaluationRunner.from_config(
            "retrieval-lab.yaml"
        ).run()
    except RetrievalLabError as exc:
        print(f"評価に失敗しました: {exc}")

低レベルライブラリの例外を、そのまま公開APIから漏らさない。

原因調査ができるよう、例外チェーンは維持する。

# 21. Notebook対応

Jupyter Notebook上で自然に利用できること。

    result = EvaluationRunner.from_config(
        "retrieval-lab.yaml"
    ).run()

    result.display()

Notebook環境でない場合は、テキスト表現へフォールバックする。

Notebook専用処理を評価コアへ混在させない。

# 22. アーキテクチャ案

    src/retrieval_lab/
    ├── domain/
    │   ├── documents.py
    │   ├── queries.py
    │   ├── search_results.py
    │   └── evaluation_results.py
    ├── loaders/
    ├── chunkers/
    ├── retrievers/
    │   ├── base.py
    │   ├── keyword.py
    │   ├── bm25.py
    │   ├── dense.py
    │   └── hybrid.py
    ├── evaluation/
    │   ├── metrics.py
    │   ├── evaluator.py
    │   ├── runner.py
    │   └── comparison.py
    ├── reporting/
    │   ├── terminal.py
    │   ├── json_report.py
    │   ├── csv_report.py
    │   └── html_report.py
    ├── config/
    ├── application/
    │   ├── services.py
    │   └── progress.py
    ├── cli/
    ├── exceptions.py
    ├── public_api.py
    └── __init__.py

以下を維持する。

- Domainと外部ライブラリ依存を分離する
- CLIへビジネスロジックを書かない
- Retriever固有処理を評価処理へ混在させない
- レポート出力を評価計算から分離する
- Python APIとCLIが同じApplication層を使用する
- Optional Dependencyなしでも基本機能が動作する

内部構成は実装中に改善してよいが、責務分離は維持する。

# 23. 依存関係

依存関係は最小限にする。

Optional Dependencyを以下のように分割する。

    retrieval-lab
    retrieval-lab[dense]
    retrieval-lab[report]
    retrieval-lab[notebook]
    retrieval-lab[all]
    retrieval-lab[dev]

コアパッケージをインストールしただけで、巨大なMLライブラリやEmbeddingモデルを自動導入しない構成を優先する。

利便性とのトレードオフはADRへ記録する。

# 24. v0.1で実装しないもの

以下はv0.1の必須対象外とする。

- LLMによる最終回答生成
- Faithfulness評価
- Groundedness評価
- LLM-as-a-Judge
- 外部Vector Database固有Adapter
- 分散検索
- GPUクラスタ
- PDF OCR
- Webクローリング
- GraphRAG
- エージェント機能
- Streamlitアプリ
- 自動ハイパーパラメータ探索

将来の拡張を妨げない設計にはするが、未使用コードを先行実装しない。

# 25. 開発フェーズ

## Phase 0：要件と設計

作成するものは以下。

- READMEのプロダクト概要
- 用語定義
- 公開Python API案
- データモデル
- 評価単位の定義
- データセット仕様
- ADR
- v0.1の非対象範囲
- 実装Issue一覧

ADRとして最低限、以下を残す。

- Python APIを中心とする理由
- CLIを薄いAdapterとする理由
- LangChain等へ依存しない理由
- 文書単位とチャンク単位の扱い
- Hybrid方式の選定
- デフォルトEmbeddingモデルの選定
- Optional Dependencyの分け方
- 評価結果の再現性方針

## Phase 1：プロジェクト基盤

以下を構築する。

- `src`レイアウト
- `pyproject.toml`
- パッケージ直下の公開API
- `py.typed`
- pytest
- Ruff
- mypyまたはPyright
- Coverage
- pre-commit
- GitHub Actions
- ライセンス
- CONTRIBUTING
- SECURITY
- CHANGELOG
- Issueテンプレート
- Pull Requestテンプレート

## Phase 2：最小の縦切り実装

最初に以下だけを通す。

    PythonのDocument入力
    → 固定長チャンク
    → Keyword検索
    → Recall@k計算
    → EvaluationResult
    → JSON出力

以下のコードが実際に動作すること。

    from retrieval_lab import (
        Document,
        EvaluationQuery,
        EvaluationRunner,
    )

    result = EvaluationRunner.quick_evaluate(
        documents=[
            Document(
                id="doc-1",
                text="検索対象となる文章",
            ),
        ],
        queries=[
            EvaluationQuery(
                id="q-1",
                query="検索クエリ",
                relevant_document_ids={
                    "doc-1",
                },
            ),
        ],
        strategies=["keyword"],
        top_k=[1],
    )

    assert (
        result.metrics["keyword"].recall_at(1)
        == 1.0
    )

この縦切りがテスト付きで完成してから、次の機能へ進む。

## Phase 3：データ入力基盤

実装対象は以下。

- Document
- Chunk
- EvaluationQuery
- Relevance情報
- txt Loader
- Markdown Loader
- JSONL Loader
- Chunker
- Dataset Loader
- validate処理
- サンプルデータ

## Phase 4：検索方式

以下の順で実装する。

1.  Keyword Baseline
2.  BM25
3.  Dense Retrieval
4.  Hybrid Retrieval

すべてのRetrieverが共通のContract Testを通過すること。

個別利用とEvaluationRunner経由の両方をテストする。

## Phase 5：評価エンジン

以下を実装する。

- 指標計算
- Top-K別評価
- 文書単位評価
- チャンク単位評価
- クエリ単位結果
- 集約結果
- レイテンシ計測
- 既存検索結果の評価
- エラー処理
- 再現性情報の保存

## Phase 6：CLIとレポート

以下を実装する。

- init
- validate
- evaluate
- inspect
- compare
- Terminal出力
- JSON
- CSV
- HTML
- CI向け終了コード

CLIとPython APIが同一処理を使用していることを確認する。

## Phase 7：公開準備

以下を完了する。

- クリーン環境からのインストール確認
- Wheelからのインストール確認
- `import retrieval_lab`確認
- Python API Quick Start
- CLI Quick Start
- Notebookサンプル
- カスタムRetrieverサンプル
- 既存検索結果評価サンプル
- APIリファレンス
- ベンチマーク
- FAQ
- PyPI公開
- GitHub Release
- GitHub Pages公開

# 26. v0.1の完了条件

## 機能

- 4種類の検索方式を比較できる
- 文書単位とチャンク単位で評価できる
- Python APIのみで一連の評価を実行できる
- CLIから同じ評価を実行できる
- Pythonオブジェクトだけでも評価できる
- 既存の検索結果だけでも評価できる
- カスタムRetrieverを接続できる
- JSON、CSV、HTMLを出力できる
- 過去結果との比較ができる
- CIで回帰判定に利用できる

## Pythonパッケージ

- `pip install`後に`import retrieval_lab`が成功する
- Wheelから正常にインストールできる
- `py.typed`が含まれている
- 公開APIに型注釈がある
- パッケージ直下から主要APIをimportできる
- 内部モジュールを知らなくても利用できる
- READMEのPythonコード例が実際に動作する

## 品質

- 主要ロジックのテストカバレッジ90%以上
- 評価指標に手計算との一致テストがある
- 全RetrieverがContract Testを通過する
- CLIとPython APIの結果一致テストがある
- 型チェック、Lint、テストが成功する
- 未使用コードと重複実装がない
- 過剰な抽象化がない
- Optional Dependencyの不足時に適切な例外を返す

## 利用者体験

- READMEだけで最初の評価を実行できる
- Python APIとCLIの両方にQuick Startがある
- サンプルデータがある
- エラー内容から修正方法が分かる
- 有料APIなしで実行できる
- 外部通信の有無が明確である
- 初回利用者が設定ファイルを一から書かなくてもよい

# 27. v0.2：評価データ作成支援

v0.1公開後、評価データを作る負担を軽減する。

## 対象機能

- インタラクティブな関連文書選択
- 正解データの作成
- データセット検査
- クエリ重複検出
- 正解文書の偏り検出
- 難易度が低すぎるクエリの検出
- Synthetic Dataset生成
- 人間によるレビュー状態管理

Synthetic Datasetは正式なGround Truthとして扱わない。

# 28. v0.3：公開デモ

Streamlitなどを利用し、検索結果を横並びで比較できるデモを作る。

提供する機能は以下。

- サンプルデータセット選択
- クエリ入力
- 検索方式ごとの結果比較
- スコア表示
- レイテンシ表示
- チャンク内容表示
- 評価指標の説明
- 設定変更

デモはPython APIを呼び出して実装する。

Webデモ専用の検索・評価処理を作らない。

# 29. v0.4：回答生成評価

検索評価が安定した後、回答生成部分を追加する。

## 29.1 評価対象

    Query
      ↓
    Retrieval
      ↓
    Reranking
      ↓
    Context Construction
      ↓
    Answer Generation
      ↓
    Answer Evaluation

以下を比較可能にする。

- LLM
- プロンプト
- 検索方式
- Top-K
- Reranker
- Context構築方法
- 引用形式

## 29.2 Generatorインターフェース

段階的に以下を追加する。

- OpenAI互換API
- Ollama
- ローカルTransformers
- Mock Generator

外部APIはOptional Dependencyとする。

明示的に設定した場合だけ外部通信を行う。

## 29.3 回答評価指標

最初にルールベースで評価可能なものを実装する。

- Exact Match
- Token F1
- Citation Presence
- Citation Validity
- Citation Precision
- Citation Recall
- Citation Coverage
- Answer Latency
- Token Usage
- 推定コスト
- Empty Answer Rate

次にEmbeddingまたはLLMを利用する評価を追加する。

- Semantic Answer Similarity
- Answer Correctness
- Faithfulness
- Groundedness
- Context Relevance
- Context Utilization
- Completeness

## 29.4 回答評価データ

    {
      "query_id": "q001",
      "query": "AWSで秘密情報を管理するサービスは何か",
      "relevant_document_ids": [
        "security-doc"
      ],
      "reference_answer": "AWS Secrets Managerなどを使用する。",
      "required_facts": [
        "AWS Secrets Manager"
      ],
      "forbidden_claims": [],
      "answerable": true
    }

`reference_answer`がない場合でも、Faithfulnessや引用評価のみ実行できるようにする。

## 29.5 LLM-as-a-Judge

LLM-as-a-Judgeの結果を絶対的な正解として扱わない。

以下を記録する。

- Judgeモデル
- Judgeプロンプト
- 温度
- 評価基準
- 実行回数
- スコアの分散
- 人間評価との一致率
- APIコスト

Judgeによって結果が変わる可能性をレポートへ明記する。

## 29.6 エンドツーエンド比較

以下の比較を可能にする。

    Experiment A
    BM25 + Top-5 + Model A

    Experiment B
    Hybrid + Top-5 + Model A

    Experiment C
    Hybrid + Reranker + Top-3 + Model B

検索評価と回答評価を分けて表示する。

以下の失敗原因を区別する。

- 検索が失敗した
- 検索結果は正しいが回答生成が失敗した
- 回答内容は正しいが引用が不適切だった
- 正解データ自体が不十分だった

# 30. WorkとCodexの役割

## Work

Workは以下を担当する。

- 要件維持
- 公開Python API設計
- アーキテクチャ設計
- タスク分割
- Codexへの指示
- 実装レビュー
- テスト結果確認
- 重複コード排除
- フェーズ間整合性確認
- READMEとAPIドキュメントの品質確認
- リリース判断

Codexの出力を無条件で採用しない。

## Codex

Codexは以下を担当する。

- 実装
- テスト作成
- 型定義
- リファクタリング
- Lint修正
- CI修正
- ドキュメントコード例の検証
- PR単位の変更作成

# 31. Codexへの指示形式

各タスクには最低限、以下を含める。

    目的
    背景
    担当範囲
    公開APIへの影響
    変更可能なファイル
    変更禁止範囲
    実装要件
    非機能要件
    テスト要件
    完了条件
    確認コマンド
    成果物

「検索機能を実装する」のような曖昧な指示だけを出さない。

並列実装時は、担当ファイルの競合を避ける。

# 32. 推奨するCodex分担

## Agent 1：公開APIとデータモデル

- Document
- Chunk
- EvaluationQuery
- SearchResult
- EvaluationResult
- `__init__.py`
- `public_api.py`
- `py.typed`

## Agent 2：設定と入力

- 設定モデル
- YAML
- Loader
- Dataset
- Validation

## Agent 3：チャンク処理

- Character Chunker
- Token Chunker
- Markdown Chunker
- ID生成
- Metadata処理

## Agent 4：KeywordとBM25

- Keyword Retriever
- BM25 Retriever
- Contract Test

## Agent 5：DenseとHybrid

- Dense Retriever
- Embedding Cache
- Hybrid Retriever
- Optional Dependency

## Agent 6：評価指標

- Metrics
- 文書単位評価
- チャンク単位評価
- 既存検索結果評価
- 手計算テスト

## Agent 7：RunnerとApplication層

- EvaluationRunner
- quick_evaluate
- 進捗Callback
- 例外変換
- Python API統合

## Agent 8：CLIとレポート

- CLI
- Terminal
- JSON
- CSV
- HTML
- compare
- inspect

## Agent 9：品質と公開準備

- 統合テスト
- CLIとAPIの結果一致確認
- Packaging
- README
- Notebook
- CI
- Release手順

最終統合はWorkが担当する。

# 33. Git運用

以下を必須とする。

- `main`へ直接pushしない
- Force Pushしない
- 機能ごとにBranchを分ける
- Pull Request経由で統合する
- 1つのPRに無関係な変更を混在させない
- PR本文へ目的、変更内容、テスト結果、既知の制約を記載する
- Merge前にWorkが差分をレビューする
- 破壊的変更はCHANGELOGへ記録する

# 34. 品質管理

各PRで最低限、以下を実行する。

    pytest
    ruff check .
    ruff format --check .
    mypy src
    python -m build

リリース前に以下を確認する。

- 新しい仮想環境へのWheelインストール
- `import retrieval_lab`
- Python APIのサンプル実行
- CLIのサンプル実行
- JSONの保存と再読み込み
- HTMLレポート表示
- Optional Dependencyなしの動作
- Optional Dependencyありの動作
- Windows、Linux、macOSのパス処理
- README内コードの実行
- Notebookサンプルの実行

# 35. 実装時の禁止事項

以下を禁止する。

- CLI専用の評価ロジック
- Python APIの後付け
- mainへの直接実装
- テスト無効化
- 型エラーの無視
- 広範囲な`Any`
- 例外の握り潰し
- 巨大な単一クラス
- 不要なSingleton
- 未使用コード
- 過剰な抽象化
- APIキーのハードコード
- 文書の無断外部送信
- Synthetic Datasetを正式な正解として扱うこと
- ベンチマーク結果の誇張
- 内部モジュールを公開サンプルで利用すること

# 36. 公開成果物

最終的に以下を公開する。

1.  GitHubリポジトリ
2.  PyPIパッケージ
3.  Python APIリファレンス
4.  CLIリファレンス
5.  GitHub Pagesドキュメント
6.  Notebookサンプル
7.  サンプル評価データセット
8.  HTMLレポート例
9.  公開デモ
10. 技術記事

技術記事の候補は以下とする。

- RAGの検索にgrepは使えるのか
- BM25・Dense・Hybridを同一条件で比較する
- Recall、MRR、nDCGを使った検索評価
- Pythonから利用できるRAG検索評価ライブラリを作った
- PythonライブラリをPyPIへ公開する
- 検索精度が高くてもRAG回答が正しいとは限らない理由
- RAGの検索評価と回答評価を分離して実装する

# 37. 開発開始時の作業

以下の順で開始する。

1.  本計画を`docs/product-plan.md`へ保存する
2.  公開Python APIの最小仕様を決定する
3.  v0.1のIssue一覧を作成する
4.  データモデルと評価仕様を確定する
5.  ADRを作成する
6.  Phase 2の最小縦切りを実装する
7.  READMEのPythonコード例をテストへ組み込む
8.  Keyword評価が完成してからBM25へ進む

# 38. 最終方針

本プロジェクトは、単なる検索比較コマンドではなく、既存システムへ組み込める検索評価SDKとして完成させる。

成果物を通して、以下を示せる状態にする。

- Pythonライブラリを設計・公開できる
- 安定した公開APIを設計できる
- CLIとApplication層を分離できる
- 複数の検索方式を同一条件で比較できる
- 評価指標を正しく実装できる
- 既存検索システムの結果も評価できる
- 再現可能な実験を設計できる
- 型、テスト、CI、Packagingを整備できる
- 検索評価と回答評価を適切に分離できる
- ローカル実行とプライバシーを考慮できる

v0.1を検索評価SDKとして確実に公開した後、評価データ作成支援、公開デモ、回答生成評価の順で拡張する。

実装量を増やすことよりも、第三者が理解し、Pythonコードへ安全に組み込み、継続的に利用できる品質を優先する。
