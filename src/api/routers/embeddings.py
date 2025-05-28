from typing import Annotated

from fastapi import APIRouter, Body, Depends

from api.auth import api_key_auth
from api.models.bedrock import get_embeddings_model as get_bedrock_embeddings_model # Alias to avoid confusion
from api.models.vertex_ai import VertexAIEmbeddingsModel, KNOWN_VERTEX_AI_MODELS as KNOWN_VERTEX_MODELS
from api.schema import EmbeddingsRequest, EmbeddingsResponse
from api.setting import DEFAULT_EMBEDDING_MODEL # This is Bedrock's default
# It seems there is no Azure specific embedding model explicitly imported here yet.
# If Azure embeddings were handled, we'd import its model and known deployments too.

router = APIRouter(
    prefix="/embeddings",
    dependencies=[Depends(api_key_auth)],
)


@router.post("", response_model=EmbeddingsResponse)
async def embeddings(
    embeddings_request: Annotated[
        EmbeddingsRequest,
        Body(
            examples=[
                {
                    "model": "cohere.embed-multilingual-v3",
                    "input": ["Your text string goes here"],
                }
            ],
        ),
    ],
):
    # Current logic: if model starts with "text-embedding-", it defaults to Bedrock's DEFAULT_EMBEDDING_MODEL.
    # This might be legacy or specific to how Azure models were being handled, assuming they also use that prefix.
    # We need to ensure Vertex AI models are checked before this default behavior if they don't match the prefix.
    # Or, if Vertex models also use this prefix, the check needs to be more specific.
    # Given KNOWN_VERTEX_MODELS, we can check against it directly.

    if embeddings_request.model in KNOWN_VERTEX_MODELS and KNOWN_VERTEX_MODELS[embeddings_request.model].get("type") == "embedding":
        vertex_model = VertexAIEmbeddingsModel(model_id=embeddings_request.model)
        # The VertexAIEmbeddingsModel's __init__ and embed methods handle validation.
        print(f"Routing to Vertex AI Embeddings for model: {embeddings_request.model}")
        return await vertex_model.embed(embeddings_request)
    elif embeddings_request.model.lower().startswith("text-embedding-"): # OpenAI/Azure prefix?
        # This was originally setting to DEFAULT_EMBEDDING_MODEL (Bedrock's default).
        # This implies that any "text-embedding-*" model not explicitly handled elsewhere
        # should be routed to Bedrock using its default embedding model.
        # This could be problematic if an Azure model like "text-embedding-ada-002" is requested
        # and there's no explicit Azure handling block above this.
        # For now, preserving the original behavior for this specific prefix after Vertex check.
        print(f"Model '{embeddings_request.model}' matches 'text-embedding-' prefix, routing to Bedrock default: {DEFAULT_EMBEDDING_MODEL}")
        embeddings_request.model = DEFAULT_EMBEDDING_MODEL # Use Bedrock's default
        bedrock_model = get_bedrock_embeddings_model(embeddings_request.model)
        return await bedrock_model.embed(embeddings_request) # Assuming Bedrock's embed can be async or is handled
    else:
        # Default to Bedrock for any other model
        print(f"Routing to Bedrock Embeddings for model: {embeddings_request.model}")
        # Exception will be raised by get_bedrock_embeddings_model if model not supported by Bedrock.
        bedrock_model = get_bedrock_embeddings_model(embeddings_request.model)
        # Assuming BedrockModel's embed method is async or the TextEmbeddingModel's embed is.
        # Based on previous BedrockModel structure, its embed methods were async.
        return await bedrock_model.embed(embeddings_request)

