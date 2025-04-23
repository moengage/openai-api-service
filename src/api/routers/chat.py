from typing import Annotated

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.models.azure_openai import AzureOpenAIChatModel, KNOWN_AZURE_DEPLOYMENTS
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error
from api.setting import DEFAULT_MODEL, AZURE_API_KEY, AZURE_API_BASE
from api.models.azure_openai import KNOWN_AZURE_DEPLOYMENTS

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

    # Handle Bedrock models
    model = BedrockModel()
    model.validate(chat_request)
    print("Chat_STREAM", chat_request.stream)
    if chat_request.stream:
        # print("*********BEADROCK_STEAM_CHAT_RESPONSE**********")
        # print(StreamingResponse(content=model.chat_stream(chat_request), media_type="text/event-stream"))
        return StreamingResponse(content=model.chat_stream(chat_request), media_type="text/event-stream")
    # print("*********BEDROCK_CHAT_REQUEST**********")
    # print(model.chat(chat_request))
    return await model.chat(chat_request)