from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.models.azure_openai import AzureOpenAIChatModel, KNOWN_AZURE_DEPLOYMENTS
from api.models.vertex_ai import KNOWN_VERTEX_AI_MODELS as KNOWN_VERTEX_MODELS # Import Vertex AI models
from api.schema import Model, Models
from api.setting import DEFAULT_MODEL, AZURE_API_KEY, AZURE_API_BASE

router = APIRouter(
    prefix="/models",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)

chat_model = BedrockModel()
azure_model = AzureOpenAIChatModel()


async def validate_model_id(model_id: str):
    # Consider instantiating BedrockModel once if chat_model.list_models() makes external calls
    bedrock_model_ids = chat_model.list_models() # Assuming this returns list of strings
    azure_model_ids = list(KNOWN_AZURE_DEPLOYMENTS.keys())
    vertex_model_ids = [m for m, info in KNOWN_VERTEX_MODELS.items() if info.get("type") == "chat"] # Filter for chat models
    all_models = bedrock_model_ids + azure_model_ids + vertex_model_ids
    if model_id not in all_models:
        # Changed to 404 and more informative message
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}. Supported models are: {', '.join(sorted(list(set(all_models))))}")


@router.get("", response_model=Models)
async def list_models():
    bedrock_models = [Model(id=model_id) for model_id in chat_model.list_models()]
    azure_models = [Model(id=model_id) for model_id in KNOWN_AZURE_DEPLOYMENTS.keys()]
    # Filter KNOWN_VERTEX_MODELS for those of type 'chat'
    vertex_chat_models = [
        Model(id=model_id) for model_id, info in KNOWN_VERTEX_MODELS.items() if info.get("type") == "chat"
    ]
    # Combine all model lists and sort them for consistent output
    combined_models = azure_models + bedrock_models + vertex_chat_models
    # Deduplicate and sort by ID
    unique_models_dict = {m.id: m for m in combined_models}
    sorted_models = sorted(list(unique_models_dict.values()), key=lambda m: m.id)
    return Models(data=sorted_models)


@router.get(
    "/{model_id}",
    response_model=Model,
)
async def get_model(
    model_id: Annotated[
        str,
        Path(description="Model ID", example="anthropic.claude-3-sonnet-20240229-v1:0"),
    ],
):
    await validate_model_id(model_id)
    return Model(id=model_id)
