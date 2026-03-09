"""Custom Prometheus business metrics for the RAG pipeline."""
from prometheus_client import Counter, Histogram

documents_uploaded_total = Counter(
    "rag_documents_uploaded_total",
    "Total number of new documents accepted for processing",
)

document_processing_seconds = Histogram(
    "rag_document_processing_seconds",
    "Time spent in the process_document background pipeline (extract+NER+chunk+embed+store)",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

embedding_cache_hits_total = Counter(
    "rag_embedding_cache_hits_total",
    "Embedding cache hits — text found in Redis, model call skipped",
)

embedding_cache_misses_total = Counter(
    "rag_embedding_cache_misses_total",
    "Embedding cache misses — text not in Redis, model was called",
)

documents_processed_total = Counter(
    "rag_documents_processed_total",
    "Background processing outcomes — success or failed",
    ["status"],  # label values: "success" | "failed"
)
