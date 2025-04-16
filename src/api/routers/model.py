from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.models.azure_openai import AzureOpenAIChatModel, KNOWN_AZURE_DEPLOYMENTS
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
    all_models = chat_model.list_models() + list(KNOWN_AZURE_DEPLOYMENTS.keys())
    if model_id not in all_models:
        raise HTTPException(status_code=500, detail="Unsupported Model Id")


@router.get("", response_model=Models)
async def list_models():
    # Combine Bedrock and Azure OpenAI models
    bedrock_models = [Model(id=model_id) for model_id in chat_model.list_models()]
    azure_models = [Model(id=model_id) for model_id in KNOWN_AZURE_DEPLOYMENTS.keys()]
    return Models(data=bedrock_models + azure_models)


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