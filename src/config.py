import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    # Gemini 3.8 Flash is the current Flash SKU for new AI Studio keys.
    # gemini-2.5-flash is listed in the catalog but returns 404 for new users.
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    GEMINI_FALLBACK_MODELS = (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
    )
    GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    USE_RAGAS = os.getenv("USE_RAGAS", "1") == "1"
    OKF_BUNDLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../okf_bundle"))
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    VECTOR_SIZE = 384
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6343"))
    COLLECTION_NAME = "saas_enterprise_catalog"
    RRF_K = 60
    TOP_K = 3
