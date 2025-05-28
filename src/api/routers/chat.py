from typing import Annotated

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.models.azure_openai import AzureOpenAIChatModel, KNOWN_AZURE_DEPLOYMENTS
from api.models.vertex_ai import VertexAIChatModel, KNOWN_VERTEX_AI_MODELS as KNOWN_VERTEX_MODELS
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error
from api.setting import DEFAULT_MODEL, AZURE_API_KEY, AZURE_API_BASE
# Removed redundant import of KNOWN_AZURE_DEPLOYMENTS

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


@router.post(
    "/completions", response_model=ChatResponse | ChatStreamResponse | Error, response_model_exclude_unset=True
)
async def chat_completions(
    chat_request: Annotated[
        ChatRequest,
        Body(
            examples=[
                {
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                },
                {
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                },
            ],
        ),
    ],
):
    if chat_request.model in KNOWN_AZURE_DEPLOYMENTS:
        azure_model = AzureOpenAIChatModel()
        # print(chat_request)
        if chat_request.stream:
            print("*********STREAM_CHAT_RESPONSE**********")
            print(StreamingResponse(content=azure_model.chat_stream(chat_request), media_type="text/event-stream"))
            return StreamingResponse(content=azure_model.chat_stream(chat_request), media_type="text/event-stream")
        print("*********CHAT_REQUEST**********")
        print(azure_model.chat(chat_request))
        return await azure_model.chat(chat_request)
    elif chat_request.model in KNOWN_VERTEX_MODELS and KNOWN_VERTEX_MODELS[chat_request.model].get("type") == "chat":
        vertex_model = VertexAIChatModel(model_id=chat_request.model)
        # The VertexAIChatModel's __init__ and chat/chat_stream methods already handle
        # validation via _validate_chat_request.
        if chat_request.stream:
            print("*********VERTEX_AI_STREAM_CHAT_RESPONSE**********")
            return StreamingResponse(content=vertex_model.chat_stream(chat_request), media_type="text/event-stream")
        print("*********VERTEX_AI_CHAT_REQUEST**********")
        return await vertex_model.chat(chat_request)
    else:
        # Default to Bedrock models
        # Add a print statement to indicate which model is being used for Bedrock
        print(f"Routing to Bedrock for model: {chat_request.model if chat_request.model else DEFAULT_MODEL}")
        model = BedrockModel() # BedrockModel uses DEFAULT_MODEL if chat_request.model is None
        model.validate(chat_request) # Validate the request against Bedrock's capabilities
        print("Chat_STREAM (Bedrock)", chat_request.stream)
        if chat_request.stream:
            return StreamingResponse(content=model.chat_stream(chat_request), media_type="text/event-stream")
        return await model.chat(chat_request)
