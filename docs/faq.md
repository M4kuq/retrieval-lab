# Retrieval Lab FAQ

## Q. Recall と document/chunk relevance はどう違いますか？

Recall@k は、指定 cutoff までに gold の relevant ID が何割現れたかを測ります。
`relevance_level: document` なら親文書 ID、`chunk` なら chunk ID を比較します。
同じ検索結果でも評価単位が違えば値が変わるため、qrels と設定で単位を明示します。

## Q. MRR、nDCG、AP は何を見ていますか？

MRR は最初の relevant 結果の順位、nDCG は graded relevance と順位の両方、AP は
relevant 結果が現れる位置ごとの precision を要約します。Hit Rate、Precision、
Recall と合わせ、ひとつの指標だけで判断しないでください。

## Q. p95 warning が出るのは失敗ですか？

いいえ。search latency の sample_count が 20 未満のとき、p95 の推定が不安定である
ことを warning として記録します。評価の失敗ではありません。反復回数を増やせない
v0.1 の設定では、p95 を比較の主根拠にせず、同じ環境での目安として扱います。

## Q. tutorial の qrels は正式な正解データですか？

いいえ。小さなローカル例の qrels は API の動作を確かめるための教材です。合成した
ground truth や tutorial の ranking を、そのまま本番の品質基準・品質 gate の正式な
正解として扱わないでください。実験の目的に沿った文書単位の relevance レビューと
独立した検証セットを用意してください。

## Q. Dense を使うとモデルが自動でダウンロードされますか？

`DenseRetriever()` の既定 backend は最初の index/search まで optional dependency の
import とモデルロードを遅延します。Dense の実行には `retrieval-lab[dense]` とモデル
キャッシュが必要です。ネットワークを使いたくない場合は、
`examples/dense_hybrid_comparison.py` のように決定的なローカル `EmbeddingBackend` を
注入してください。この例の backend はモデルをダウンロードしません。

## Q. Dense のモデルキャッシュを共有してよいですか？

実験ごとにモデル ID、revision、prompt、normalize 設定を固定してください。Retrieval
Lab の評価 cache は chunk/index の再構築を減らしますが、cache hit/miss や runtime
latency は deterministic run_id を変えません。異なるモデルや revision の結果を、
同じ実験として比較しないでください。

## Q. baseline と candidate を比較できないのはなぜですか？

dataset hash、query IDs、relevance level、metric version、top_k、metric shape などの
blocking 条件が一致しない場合、比較は計算前に拒否されます。まず同じ qrels と評価設定
で再実行し、retriever の設定差は実験上の差として記録してください。

## Q. 評価対象の本文や秘密情報はレポートに出ますか？

CSV/HTML/summary は query 本文、retrieved IDs、document 本文を出力しません。入力の
Markdown、qrels、manifest、エラーメッセージにも API キー、password、token を入れないで
ください。既に秘密を含む入力を使った場合は、結果の共有前にファイルと生成物を破棄し、
認証情報を通常の運用手順でローテーションしてください。
