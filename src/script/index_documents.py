import os
import glob
import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

# Document Loaders and Splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Azure Vector Store & Embeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch

# -----------------------------
# 1. Setup Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("indexer")

def index_docs():
    """
    Reads PDFs from src/data, splits them into chunks,
    and uploads vectors to Azure AI Search.
    """

    # -----------------------------
    # 2. Define Paths
    # -----------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "../data")  # Updated path to src/data

    # -----------------------------
    # 3. Check Environment Variables
    # -----------------------------
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",  # Must match Azure deployment name
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file and ensure all variables are set.")
        return

    # Print environment config for debug
    logger.info("=" * 60)
    logger.info("✅ Environment Configuration Check:")
    for var in required_vars:
        logger.info(f"{var}: {os.getenv(var)}")
    logger.info("=" * 60)

    # -----------------------------
    # 4. Initialize Embedding Model
    # -----------------------------
    try:
        embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        logger.info("Initializing Azure OpenAI Embeddings...")
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        logger.info(f"✓ Embeddings model initialized successfully using deployment '{embedding_deployment}'")
    except Exception as e:
        logger.error(f"❌ Failed to initialize embeddings: {e}")
        logger.error("Please verify your Azure OpenAI deployment name and endpoint.")
        return

    # -----------------------------
    # 5. Initialize Azure Search
    # -----------------------------
    try:
        logger.info("Initializing Azure AI Search vector store...")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        vector_store = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
            index_name=index_name,
            embedding_function=embeddings.embed_query
        )
        logger.info(f"✓ Vector store initialized for index: {index_name}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Azure Search: {e}")
        logger.error("Please verify your Azure Search endpoint, API key, and index name.")
        return

    # -----------------------------
    # 6. Find PDF Files
    # -----------------------------
    pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {data_folder}. Please add files.")
        return

    logger.info(f"Found {len(pdf_files)} PDFs to process: {[os.path.basename(f) for f in pdf_files]}")
    all_splits = []

    # -----------------------------
    # 7. Process Each PDF
    # -----------------------------
    for pdf_path in pdf_files:
        try:
            logger.info(f"Loading: {os.path.basename(pdf_path)}...")
            loader = PyPDFLoader(pdf_path)
            raw_docs = loader.load()

            # Chunking strategy: 1000 chars with 200 overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(raw_docs)

            # Tag source for citation later
            for split in splits:
                split.metadata["source"] = os.path.basename(pdf_path)

            all_splits.extend(splits)
            logger.info(f" -> Split into {len(splits)} chunks.")

        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_path}: {e}")

    # -----------------------------
    # 8. Upload to Azure Search
    # -----------------------------
    if all_splits:
        logger.info(f"Uploading {len(all_splits)} chunks to Azure AI Search Index '{index_name}'...")
        try:
            vector_store.add_documents(documents=all_splits)
            logger.info("=" * 60)
            logger.info("✅ Indexing Complete! The Knowledge Base is ready.")
            logger.info(f"Total chunks indexed: {len(all_splits)}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"❌ Failed to upload documents to Azure Search: {e}")
            logger.error("Please check your Azure Search configuration and try again.")
    else:
        logger.warning("No documents were processed.")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    index_docs()
