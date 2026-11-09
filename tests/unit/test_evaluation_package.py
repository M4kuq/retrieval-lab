"""Public surface of the ``retrieval_lab.evaluation`` package.

The engine and latency names are imported eagerly, while the precomputed,
callable, and asynchronous entry points resolve through a lazy module-level
``__getattr__`` to avoid import cycles. These tests pin every name declared
in ``__all__`` to the object provided by its defining module.
"""

import importlib

import pytest

import retrieval_lab.evaluation as evaluation

_LAZY_MODULES = {
    "RetrievedQueryResult": "retrieval_lab.evaluation.precomputed",
    "evaluate_results": "retrieval_lab.evaluation.precomputed",
    "evaluate_retrievers": "retrieval_lab.retrievers.callable",
    "AsyncRetriever": "retrieval_lab.retrievers.async_callable",
    "AsyncCallableRetriever": "retrieval_lab.retrievers.async_callable",
    "evaluate_async_retrievers": "retrieval_lab.retrievers.async_callable",
}

_EAGER_MODULES = {
    "average_precision_at_k": "retrieval_lab.evaluation.metrics",
    "hit_rate_at_k": "retrieval_lab.evaluation.metrics",
    "ndcg_at_k": "retrieval_lab.evaluation.metrics",
    "precision_at_k": "retrieval_lab.evaluation.metrics",
    "recall_at_k": "retrieval_lab.evaluation.metrics",
    "reciprocal_rank": "retrieval_lab.evaluation.metrics",
    "LatencyStats": "retrieval_lab.evaluation.latency",
    "nearest_rank_percentile": "retrieval_lab.evaluation.latency",
}


@pytest.mark.parametrize("name", evaluation.__all__)
def test_every_declared_export_resolves(name: str) -> None:
    assert getattr(evaluation, name) is not None


@pytest.mark.parametrize("name,module_name", sorted(_EAGER_MODULES.items()))
def test_eager_exports_are_the_defining_module_objects(
    name: str, module_name: str
) -> None:
    module = importlib.import_module(module_name)

    assert getattr(evaluation, name) is getattr(module, name)


@pytest.mark.parametrize("name,module_name", sorted(_LAZY_MODULES.items()))
def test_lazy_exports_are_the_defining_module_objects(
    name: str, module_name: str
) -> None:
    module = importlib.import_module(module_name)

    assert getattr(evaluation, name) is getattr(module, name)


def test_declared_exports_cover_eager_and_lazy_names() -> None:
    assert set(evaluation.__all__) == set(_EAGER_MODULES) | set(_LAZY_MODULES)


def test_unknown_attribute_raises_attribute_error() -> None:
    missing = "not_a_real_export"

    with pytest.raises(AttributeError):
        getattr(evaluation, missing)

    assert not hasattr(evaluation, missing)
