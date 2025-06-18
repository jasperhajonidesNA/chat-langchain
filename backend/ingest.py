"""Load html from files, clean up, split, ingest into Weaviate."""

import logging
import os
import re
from typing import Optional
from pathlib import Path
from typing import List

import weaviate
from bs4 import BeautifulSoup, SoupStrainer

# from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_weaviate import WeaviateVectorStore

from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.embeddings import get_embeddings_model
from backend.parser import langchain_docs_extractor


from langchain.schema import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader


import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Choose a directory that is in the search path, or add your own
nltk_data_dir = os.path.expanduser('~/nltk_data')
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Download to that directory
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)

print("NLTK data path:", nltk.data.path)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    print("averaged_perceptron_tagger_eng found!")
except LookupError:
    print("averaged_perceptron_tagger_eng NOT found!")


def metadata_extractor(
    meta: dict, soup: BeautifulSoup, title_suffix: Optional[str] = None
) -> dict:
    title_element = soup.find("title")
    description_element = soup.find("meta", attrs={"name": "description"})
    html_element = soup.find("html")
    title = title_element.get_text() if title_element else ""
    if title_suffix is not None:
        title += title_suffix

    return {
        "source": meta["loc"],
        "title": title,
        "description": description_element.get("content", "")
        if description_element
        else "",
        "language": html_element.get("lang", "") if html_element else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langgraph_docs():
    return SitemapLoader(
        "https://langchain-ai.github.io/langgraph/sitemap.xml",
        parsing_function=simple_extractor,
        default_parser="lxml",
        bs_kwargs={"parse_only": SoupStrainer(name=("article", "title"))},
        meta_function=lambda meta, soup: metadata_extractor(
            meta, soup, title_suffix=" | 🦜🕸️LangGraph"
        ),
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


# --------------------------------------------------------------------------- #
#  Helper – trim repeated blank lines and normalise whitespace
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)  # normalise line endings
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)  # collapse multiple blank lines
    return text.strip()


# --------------------------------------------------------------------------- #
#  Main loader
# --------------------------------------------------------------------------- #
def load_methodology_docs(
    root: str = "/Users/jasper.hajonides/Downloads/Module Methodologies/test",
) -> List[Document]:
    """
    Recursively scans *root* for .md files, loads raw markdown content, and returns a list
    of LangChain `Document` objects with metadata fields that mirror the ones
    you already use for web-sourced content.

    PRESERVES markdown formatting (headers, etc.) for downstream header-based splitting.

    Returns
    -------
    List[Document]
        Each document's `page_content` is the raw markdown text of the .md
        file (with headers preserved).  Metadata keys:

        * source      – absolute file path
        * loc         – same as source (keeps interface parity with sitemap loader)
        * title       – filename stem
        * description – first 200 characters of text (empty if none)
        * language    – "en"  (change if you detect others)
    """
    # 1) Load every .md under *root* (TextLoader preserves raw markdown)
    dir_loader = DirectoryLoader(
        root,
        glob="**/*[!~$]*.md",  # exclude files starting with ~$
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = dir_loader.load()  # synchronous

    # 2) Post-process: clean text & add the extra metadata keys you expect
    for doc in docs:
        raw_path = doc.metadata.get("source", "")
        doc.page_content = _clean(doc.page_content)

        title = Path(raw_path).stem
        snippet = doc.page_content[:200].replace("\n", " ").strip()

        doc.metadata.update(
            {
                "title": title,
                "description": snippet,
                "language": "en",
            }
        )

    return docs


def simple_extractor(html: str | BeautifulSoup) -> str:
    if isinstance(html, str):
        soup = BeautifulSoup(html, "lxml")
    elif isinstance(html, BeautifulSoup):
        soup = html
    else:
        raise ValueError(
            "Input should be either BeautifulSoup object or an HTML string"
        )
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def clean_metadata_keys(metadata: dict) -> dict:
    """Convert metadata keys to be GraphQL compatible (no spaces, valid names)."""
    clean_metadata = {}
    for key, value in metadata.items():
        # Convert spaces to underscores and make lowercase
        clean_key = key.replace(" ", "_").replace("-", "_").lower()
        # Only keep alphanumeric and underscores
        clean_key = "".join(c for c in clean_key if c.isalnum() or c == "_")
        # Ensure it starts with a letter or underscore
        if clean_key and not (clean_key[0].isalpha() or clean_key[0] == "_"):
            clean_key = "field_" + clean_key
        clean_metadata[clean_key] = value
    return clean_metadata


def process_documents(docs_from_methodology: List[Document], text_splitter) -> List[Document]:
    """Process documents by splitting on markdown headers and cleaning metadata."""
    docs_transformed = []
    
    for doc in docs_from_methodology:
        logger.info(f"Processing document: {doc.metadata.get('source', 'Unknown')}")
        
        # Split by markdown headers
        header_splits = text_splitter.split_text(doc.page_content)
        logger.info(f"split_text returned {len(header_splits)} items of type: {type(header_splits[0]) if header_splits else 'None'}")
        
        # Handle based on what's actually returned
        for split in header_splits:
            if isinstance(split, Document):
                # If it's already a Document, preserve original metadata and update
                split.metadata.update(doc.metadata)
                docs_transformed.append(split)
            else:
                # If it's a string, create Document with original metadata
                new_doc = Document(
                    page_content=str(split),
                    metadata=doc.metadata.copy()
                )
                docs_transformed.append(new_doc)
    
    # Filter out tiny chunks
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]
    
    # Clean metadata and ensure required fields
    for doc in docs_transformed:
        doc.metadata = clean_metadata_keys(doc.metadata)
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""
    
    return docs_transformed


def ingest_docs():
    """Main ingestion function - loads, processes, and indexes methodology documents."""
    # Environment setup
    WEAVIATE_URL = os.environ[
        "WEAVIATE_URL"
    ]  # os.environ["WEAVIATE_URL"]

    # 1. Read & strip any trailing slash
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "").rstrip("/").removesuffix("/v1")
    # 2. (Optional) also strip "/v1" if someone accidentally included it
    print(f"📡  Connecting to Weaviate at: {WEAVIATE_URL}")

    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    # Initialize components
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    embedding = get_embeddings_model()

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    ) as weaviate_client:
        
        # Setup vector store and record manager
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_INDEX_NAME,
            text_key="text",
            embedding=embedding,
            attributes=["source", "title"],
        )

        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
        record_manager.create_schema()

        # Load and process documents
        docs_from_methodology = load_methodology_docs()
        logger.info(f"Loaded {len(docs_from_methodology)} docs from Methodology")
        
        docs_transformed = process_documents(docs_from_methodology, text_splitter)
        logger.info(f"Final chunk count: {len(docs_transformed)}")

        # Index documents
        indexing_stats = index(
            docs_transformed,
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key="source",
            force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
        )

        # Report results
        logger.info(f"Indexing stats: {indexing_stats}")
        num_vecs = (
            weaviate_client.collections.get(WEAVIATE_DOCS_INDEX_NAME)
            .aggregate.over_all()
            .total_count
        )
        logger.info(f"LangChain now has this many vectors: {num_vecs}")

if __name__ == "__main__":
    ingest_docs()
