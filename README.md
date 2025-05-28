# OpenAI-Compatible API Proxy

This project provides an OpenAI-compatible API proxy that allows you to interact with various Large Language Model (LLM) providers using the familiar OpenAI API schema.

## Supported Services

*   Amazon Bedrock
*   Azure OpenAI Service
*   **Google Cloud Vertex AI**

## Features

*   **OpenAI Compatibility:** Use your existing OpenAI SDKs and tools.
*   **Multi-Provider Support:** Access models from different cloud providers through a unified interface.
*   **Streaming Support:** Handle streaming responses for chat completions.
*   **Embeddings:** Generate text embeddings.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Access credentials for the desired LLM providers (e.g., AWS credentials for Bedrock, Azure API Key for Azure OpenAI, Google Cloud credentials for Vertex AI).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r src/requirements.txt
    ```

3.  **Set up environment variables:**
    Copy the `.env.example` file to `.env` and configure the necessary variables for the services you want to use.

    ```bash
    cp .env.example .env
    # Edit .env with your credentials and settings
    ```

### Running the Application

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## Configuration

### General Configuration

*   `API_ROUTE_PREFIX`: The prefix for API routes (default: `/api/v1`).
*   `DEFAULT_API_KEYS`: Comma-separated list of default API keys to use if no specific key is provided by the client (e.g., `your-secret-key`). For production, it's recommended to use unique, strong keys.

### Amazon Bedrock

*   `AWS_REGION`: The AWS region where Bedrock service is located (e.g., `us-west-2`).
*   `DEFAULT_MODEL`: The default Bedrock model ID for chat completions (e.g., `anthropic.claude-3-sonnet-20240229-v1:0`).
*   `DEFAULT_EMBEDDING_MODEL`: The default Bedrock model ID for embeddings (e.g., `cohere.embed-multilingual-v3`).
*   `ENABLE_CROSS_REGION_INFERENCE`: Set to `true` or `false` to enable/disable cross-region inference for Bedrock.

Authentication for Bedrock typically uses standard AWS credentials (e.g., via environment variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, or an IAM role).

### Azure OpenAI Service

*   `AZURE_API_KEY`: Your Azure OpenAI API key.
*   `AZURE_API_BASE`: Your Azure OpenAI API base URL (e.g., `https://your-resource-name.openai.azure.com`).
    *   Known Azure deployments are configured within `src/api/models/azure_openai.py`.

### Google Cloud Vertex AI

*   `VERTEX_AI_PROJECT_ID`: Your Google Cloud project ID where Vertex AI is enabled.
*   `VERTEX_AI_LOCATION`: The Google Cloud location for Vertex AI services (e.g., `us-central1`).
*   `DEFAULT_VERTEX_AI_CHAT_MODEL`: The default Vertex AI chat model ID (e.g., `gemini-1.5-flash-001`).
*   `DEFAULT_VERTEX_AI_EMBEDDING_MODEL`: The default Vertex AI embedding model ID (e.g., `textembedding-gecko@003`).

Authentication for Vertex AI typically uses Google Cloud Application Default Credentials (ADC). Ensure your environment is authenticated (e.g., by running `gcloud auth application-default login` or by setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to a service account key JSON file if running outside Google Cloud).

## API Usage Examples

The API follows the OpenAI specification. You can use the OpenAI Python client or other compatible libraries.

### List Models

```bash
curl http://localhost:8000/api/v1/models -H "Authorization: Bearer your-secret-key"
```

### Chat Completions

**Using a Bedrock Model (Claude 3 Sonnet):**

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "messages": [
      {"role": "user", "content": "Hello from Bedrock!"}
    ]
  }'
```

**Using an Azure OpenAI Model (e.g., GPT-4 deployed as "gpt-4"):**

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "gpt-4", 
    "messages": [
      {"role": "user", "content": "Hello from Azure OpenAI!"}
    ]
  }'
```

**Using a Google Cloud Vertex AI Model (Gemini 1.5 Flash):**

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "gemini-1.5-flash-001", 
    "messages": [
      {"role": "user", "content": "Hello from Vertex AI!"}
    ]
  }'
```

### Embeddings

**Using a Bedrock Model (Cohere Embed Multilingual):**

```bash
curl http://localhost:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "cohere.embed-multilingual-v3",
    "input": ["This is a test sentence for Bedrock embeddings."]
  }'
```

**Using a Google Cloud Vertex AI Model (Gecko):**

```bash
curl http://localhost:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "textembedding-gecko@003",
    "input": ["This is a test sentence for Vertex AI embeddings."]
  }'
```

## Development

(Details about setting up a development environment, running tests, etc.)

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
