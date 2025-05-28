import os

DEFAULT_API_KEYS = "bedrock"

API_ROUTE_PREFIX = os.environ.get("API_ROUTE_PREFIX", "/api/v1")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE")

TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for Amazon Bedrock models.
"""

DEBUG = os.environ.get("DEBUG", "false").lower() != "false"
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3")
ENABLE_CROSS_REGION_INFERENCE = os.environ.get("ENABLE_CROSS_REGION_INFERENCE", "true").lower() != "false"

# Vertex AI Settings
VERTEX_AI_PROJECT_ID = os.environ.get("VERTEX_AI_PROJECT_ID", "your-gcp-project-id")
VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
DEFAULT_VERTEX_AI_CHAT_MODEL = os.environ.get("DEFAULT_VERTEX_AI_CHAT_MODEL", "gemini-1.5-flash-001")
DEFAULT_VERTEX_AI_EMBEDDING_MODEL = os.environ.get("DEFAULT_VERTEX_AI_EMBEDDING_MODEL", "textembedding-gecko@003")
