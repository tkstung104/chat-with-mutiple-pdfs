import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# LangChain Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATOR = "\n"

# File Upload Configuration
ALLOWED_FILE_TYPES = ["pdf"]

# Memory Configuration
MEMORY_KEY = "chat_history"  # Key trong response object