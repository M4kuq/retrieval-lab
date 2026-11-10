"""Dependency-free result presentation helpers."""

from retrieval_lab.reporting.csv import per_query_csv, summary_csv
from retrieval_lab.reporting.html import comparison_html, result_html
from retrieval_lab.reporting.summary import result_summary

__all__ = [
    "comparison_html",
    "per_query_csv",
    "result_html",
    "result_summary",
    "summary_csv",
]
