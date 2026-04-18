"""
setup_vector_search.py
----------------------
One-time setup for the RAG retriever used by the chat agent.

What this script does (mirrors `chatbot/setup.ipynb`, but everything runs on
Databricks — no local sentence-transformers needed):

    1. Extracts + chunks the rulebook PDFs (G&SR, IRCTC cancellation, refund).
    2. Writes the chunks to a Delta table     workspace.rail_adhikar.rules_gold
    3. Enables Change Data Feed on that table (required for Delta Sync).
    4. Creates / reuses a Vector Search endpoint.
    5. Creates a Delta-sync Vector Search index that uses a **Databricks-managed
       embedding endpoint** (default: `databricks-gte-large-en`) so Databricks
       computes embeddings server-side. Queries can then use `query_text=...`
       and Databricks embeds on the fly.

Run once from your terminal or a Databricks notebook:

    python setup_vector_search.py                       # uses defaults
    python setup_vector_search.py --pdf-dir ./chatbot   # custom PDF folder

Requirements (in requirements.txt already):
    databricks-sdk, databricks-vectorsearch, pdfplumber
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setup_vector_search")

# ── Defaults (all overridable via env or CLI) ───────────────────────────────
DEFAULT_CATALOG  = os.environ.get("VS_CATALOG", "workspace")
DEFAULT_SCHEMA   = os.environ.get("VS_SCHEMA", "rail_adhikar")
DEFAULT_TABLE    = os.environ.get("VS_SOURCE_TABLE", "rules_gold")
DEFAULT_ENDPOINT = os.environ.get("VECTOR_SEARCH_ENDPOINT", "try1")
DEFAULT_INDEX    = os.environ.get(
    "VECTOR_SEARCH_INDEX", f"{DEFAULT_CATALOG}.{DEFAULT_SCHEMA}.rules_vs_index"
)
# Databricks-hosted Foundation Model API embedding endpoint.
# `databricks-gte-large-en` (1024 dim) is available by default on most workspaces.
# Other good picks: `databricks-bge-large-en` (1024 dim).
DEFAULT_EMBEDDING_ENDPOINT = os.environ.get(
    "VECTOR_SEARCH_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
)
DEFAULT_PDF_DIR = Path(os.environ.get("RULES_PDF_DIR", "./chatbot"))


# ── PDF → chunks ─────────────────────────────────────────────────────────────

def extract_and_chunk_pdf(pdf_path: Path, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Slide a window over each page's text, emit (source, page, chunk_id, text)."""
    import pdfplumber  # noqa: PLC0415

    chunks: list[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").replace("\n", " ").strip()
            if not text:
                continue
            step = max(chunk_size - overlap, 1)
            for i in range(0, len(text), step):
                snippet = text[i:i + chunk_size]
                if not snippet.strip():
                    continue
                chunks.append({
                    "source":   pdf_path.name,
                    "page":     page_num,
                    "chunk_id": len(chunks),
                    "text":     snippet,
                })
    return chunks


def build_chunks(pdf_dir: Path) -> list[dict]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {pdf_dir.resolve()}")
    all_chunks: list[dict] = []
    for pdf in pdfs:
        logger.info("Processing %s ...", pdf.name)
        chunks = extract_and_chunk_pdf(pdf)
        logger.info("  \u2192 %d chunks", len(chunks))
        all_chunks.extend(chunks)
    # Re-index chunk_id globally so it stays monotonic across files.
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i
    return all_chunks


# ── Delta table write ────────────────────────────────────────────────────────

def write_delta_table(chunks: list[dict], catalog: str, schema: str, table: str) -> str:
    """Write the chunks to `<catalog>.<schema>.<table>` and return the FQN."""
    from pyspark.sql import SparkSession                          # noqa: PLC0415
    from pyspark.sql import functions as F                        # noqa: PLC0415
    from pyspark.sql.types import (                               # noqa: PLC0415
        IntegerType, StringType, StructField, StructType,
    )

    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    schema_struct = StructType([
        StructField("source",   StringType(),  True),
        StructField("page",     IntegerType(), True),
        StructField("chunk_id", IntegerType(), True),
        StructField("text",     StringType(),  True),
    ])
    df = spark.createDataFrame(chunks, schema_struct)
    df = df.withColumn("id", F.monotonically_increasing_id())

    fqn = f"{catalog}.{schema}.{table}"
    df.write.mode("overwrite").option("mergeSchema", "true").format("delta").saveAsTable(fqn)
    logger.info("\u2705 Delta table written: %s (%d rows)", fqn, df.count())

    # Change Data Feed is required for Delta Sync
    spark.sql(f"ALTER TABLE {fqn} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    logger.info("\u2705 Change Data Feed enabled on %s", fqn)
    return fqn


# ── Vector Search endpoint + index ───────────────────────────────────────────

def create_vs_index(
    source_table: str,
    index_name: str,
    endpoint_name: str,
    embedding_endpoint: str,
) -> None:
    """
    Create a Delta-sync Vector Search index that uses Databricks-managed
    embeddings (Foundation Model API endpoint `embedding_endpoint`).
    """
    from databricks.vector_search.client import VectorSearchClient  # noqa: PLC0415

    vs_client = VectorSearchClient(disable_notice=True)

    # 1. Endpoint (Free edition allows 1)
    try:
        vs_client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        logger.info("\u2705 Vector Search endpoint created: %s", endpoint_name)
    except Exception as exc:  # noqa: BLE001
        if "already exists" in str(exc).lower():
            logger.info("\u2705 Endpoint already exists: %s", endpoint_name)
        else:
            logger.warning("\u26a0\ufe0f Endpoint issue (continuing): %s", exc)

    # 2. Delta-sync index with Databricks-managed embeddings.
    #    `embedding_source_column` + `embedding_model_endpoint_name` tells
    #    Databricks to compute embeddings itself \u2014 no local ML needed.
    try:
        vs_client.create_delta_sync_index(
            endpoint_name=endpoint_name,
            source_table_name=source_table,
            index_name=index_name,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_source_column="text",
            embedding_model_endpoint_name=embedding_endpoint,
        )
        logger.info(
            "\u2705 Vector Search index created: %s  (embeddings: %s)",
            index_name, embedding_endpoint,
        )
    except Exception as exc:  # noqa: BLE001
        if "already exists" in str(exc).lower():
            logger.info("\u2705 Index already exists: %s", index_name)
        else:
            logger.error("\u274c Index creation failed: %s", exc)
            raise


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir",   type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--catalog",   default=DEFAULT_CATALOG)
    parser.add_argument("--schema",    default=DEFAULT_SCHEMA)
    parser.add_argument("--table",     default=DEFAULT_TABLE)
    parser.add_argument("--endpoint",  default=DEFAULT_ENDPOINT)
    parser.add_argument("--index",     default=DEFAULT_INDEX)
    parser.add_argument("--embedding-endpoint", default=DEFAULT_EMBEDDING_ENDPOINT)
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Only create the VS endpoint + index; assume Delta table already exists.",
    )
    args = parser.parse_args()

    if not args.skip_ingest:
        chunks = build_chunks(args.pdf_dir)
        source_table = write_delta_table(chunks, args.catalog, args.schema, args.table)
    else:
        source_table = f"{args.catalog}.{args.schema}.{args.table}"
        logger.info("Skipping ingest; reusing %s", source_table)

    create_vs_index(
        source_table=source_table,
        index_name=args.index,
        endpoint_name=args.endpoint,
        embedding_endpoint=args.embedding_endpoint,
    )

    print(f"""
\u2705 Vector Search is ready.

  endpoint :  {args.endpoint}
  index    :  {args.index}
  source   :  {source_table}
  embedder :  {args.embedding_endpoint}  (Databricks-managed)

Point the chat agent / FastAPI at it with:

  export VECTOR_SEARCH_ENDPOINT={args.endpoint}
  export VECTOR_SEARCH_INDEX={args.index}
""")


if __name__ == "__main__":
    main()
