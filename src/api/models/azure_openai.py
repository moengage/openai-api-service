import base64
import json
import logging
import re
import time
from fastapi import HTTPException
import os
from abc import ABC
from typing import AsyncIterable, Iterable, Literal, List, Dict, Any, Optional, Union
from openai import AzureOpenAI, AsyncAzureOpenAI, BadRequestError, RateLimitError, APIError
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionChunk
from openai.types.chat.completion_create_params import Function
import tiktoken
import numpy as np
import requests


from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.schema import (
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    ChatStreamResponse,
    Choice,
    ChoiceDelta,
    Embedding, 
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    Error,
    ErrorMessage,
    ImageContent,
    ResponseFunction,
    TextContent,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)

logger = logging.getLogger(__name__)
DEBUG = os.environ.get("DEBUG", "false").lower() != "false"

try:
    azure_openai_client = AzureOpenAI() 
    async_azure_openai_client = AsyncAzureOpenAI() 
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI clients: {e}")
    azure_openai_client = None
    async_azure_openai_client = None


# --- Model Listing (Azure Deployments) ---
# Option 1: Hardcoded list (simple)
KNOWN_AZURE_DEPLOYMENTS = {
    "gpt-5-chat": {"type": "chat", "modalities": ["TEXT", "IMAGE"]},
    "gpt-5-mini": {"type": "chat", "modalities": ["TEXT","IMAGE"]},
}

azure_deployments_list = KNOWN_AZURE_DEPLOYMENTS

ENCODER = tiktoken.get_encoding("cl100k_base") 


class AzureOpenAIChatModel(BaseChatModel):

    def __init__(self):
        if not async_azure_openai_client:
            raise RuntimeError("AsyncAzureOpenAI client not initialized. Check configuration.")

    def list_models(self) -> list[str]:
        """Return known Azure OpenAI deployment names suitable for chat."""
        global azure_deployments_list
        azure_deployments_list = KNOWN_AZURE_DEPLOYMENTS() 
        return [name for name, info in azure_deployments_list.items() if info.get("type") == "chat"]

    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests for Azure OpenAI."""
        error = ""
        
        model_info = azure_deployments_list.get(chat_request.model)
        if not model_info:
            error = f"Unknown Azure OpenAI deployment name '{chat_request.model}'. Check configuration or use the models API."
        elif model_info.get("type") != "chat":
             error = f"Deployment '{chat_request.model}' is not configured as a chat model."

        if error:
            raise HTTPException(
                status_code=400,
                detail=error,
            )

    async def _invoke_azure_openai(self, azure_args: dict, stream: bool = False):
        """Common logic for invoking Azure OpenAI Chat Completions."""
        if DEBUG:
             # Be careful logging sensitive data like API keys if they somehow end up here
             log_args = azure_args.copy()
             logger.info("Azure OpenAI Request: " + json.dumps(log_args, cls=CustomEncoder))


        try:
            if stream:
                # Returns an AsyncStream
                response = await async_azure_openai_client.chat.completions.create(
                     stream=True,
                     **azure_args
                 )
            else:
                 # Returns a ChatCompletion object
                 response = await async_azure_openai_client.chat.completions.create(
                     stream=False,
                     **azure_args
                 )
            return response
        except BadRequestError as e:
            logger.error(f"Azure OpenAI Bad Request Error (400): {e}")
            e.response.json()
            detail = f"Azure OpenAI request failed: {e.message}"
            try: 
                err_body = e.response.json()
                detail = err_body.get("error", {}).get("message", detail)
            except: pass
            raise HTTPException(status_code=400, detail=detail)
        except RateLimitError as e:
            logger.error(f"Azure OpenAI Rate Limit Error (429): {e}")
            raise HTTPException(status_code=429, detail=f"Azure OpenAI rate limit exceeded: {e.message}")
        except APIError as e:
             logger.error(f"Azure OpenAI API Error ({e.status_code}): {e}")
             raise HTTPException(status_code=e.status_code or 500, detail=f"Azure OpenAI API error: {e.message}")
        except Exception as e:
             logger.error(f"Unexpected error calling Azure OpenAI: {e}")
             raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    async def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Implementation for Chat API using Azure OpenAI."""
        self.validate(chat_request)
        azure_args = self._parse_request_for_azure(chat_request)

        # Invoke Azure OpenAI (non-streaming)
        azure_response = await self._invoke_azure_openai(azure_args, stream=False)

        # Parse the Azure response into our internal ChatResponse format
        chat_response = self._create_response_from_azure(azure_response)

        if DEBUG:
            logger.info("Proxy response: " + chat_response.model_dump_json())
        return chat_response

    async def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Implementation for Chat Stream API using Azure OpenAI."""
        self.validate(chat_request)
        message_id = self.generate_message_id() 
        azure_args = self._parse_request_for_azure(chat_request)

        try:
            # Invoke Azure OpenAI (streaming)
            stream = await self._invoke_azure_openai(azure_args, stream=True)

            async for chunk in stream:
                 stream_response = self._create_response_stream_from_azure(
                     model_id=chat_request.model, 
                     message_id=message_id,
                     chunk=chunk
                 )
                 if not stream_response:
                     continue

                 if DEBUG:
                     logger.info("Proxy stream response chunk: " + stream_response.model_dump_json())

                 """
                 Yield based on content or usage request
                 OpenAI stream chunks can contain choices OR usage, but usually not both populated at once.
                 The last chunk *might* contain usage if requested.
                 """
                 if stream_response.choices and stream_response.choices[0].delta != {}: 
                     yield self.stream_response_to_bytes(stream_response)
                 elif stream_response.usage and chat_request.stream_options and chat_request.stream_options.include_usage:
                      yield self.stream_response_to_bytes(stream_response)
            yield self.stream_response_to_bytes()

        except HTTPException as e: 
             error_event = Error(error=ErrorMessage(message=e.detail, code=str(e.status_code)))
             yield self.stream_response_to_bytes(error_event)
             logger.error(f"Stream failed with HTTPException: {e.detail}")
        except Exception as e: 
            error_event = Error(error=ErrorMessage(message=str(e), code="500"))
            yield self.stream_response_to_bytes(error_event)
            logger.error(f"Stream failed with unexpected error: {e}", exc_info=True)


    def _parse_messages_for_azure(
        self,
        chat_request: ChatRequest
    ) -> List[Dict[str, Any]]:
        """
        Prepare the messages list for Azure OpenAI Chat Completions API.
        Handles system prompts, user/assistant roles, tool calls, and multimodal content.
        """
        messages = []
        system_content = ""
        for message in chat_request.messages:
            if message.role == "system":
                if isinstance(message.content, str):
                    system_content += message.content + "\n"

        if system_content:
            messages.append({"role": "system", "content": system_content.strip()})
        for message in chat_request.messages:
            if message.role == "user":
                content_parts = self._parse_content_parts_for_azure(message, chat_request.model)
                msg_content = content_parts[0]['text'] if len(content_parts) == 1 and 'text' in content_parts[0] else content_parts
                messages.append({"role": "user", "content": msg_content})

            elif message.role == "assistant":
                msg = {"role": "assistant"}
                if message.content:
                     if isinstance(message.content, str):
                         msg["content"] = message.content
                     else:
                           logger.warning("Non-string assistant content detected, using only text parts.")
                           msg["content"] = " ".join([part.text for part in message.content if isinstance(part, TextContent)])

                if message.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        } for tc in message.tool_calls
                    ]
                if "content" in msg or "tool_calls" in msg:
                    messages.append(msg)

            elif message.role == "tool":
                 messages.append(
                     {
                         "role": "tool",
                         "tool_call_id": message.tool_call_id,
                         "content": message.content, 
                     }
                 )
            elif message.role == "system":
                continue

            else:
                logger.warning(f"Unsupported message role encountered: {message.role}")

        print("*********** Azure OpenAI messages **********")
        print(json.dumps(messages, indent=2, cls=CustomEncoder))
        return messages

    def _parse_content_parts_for_azure(
        self,
        message: UserMessage, 
        model_deployment: str,
    ) -> list[dict]:
        """ Parses UserMessage content (text or image) into Azure format."""
        if isinstance(message.content, str):
            return [{"type": "text", "text": message.content}]

        content_parts = []
        has_image = False
        for part in message.content:
            if isinstance(part, TextContent):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContent):
                 has_image = True
                 if not self.is_supported_modality(model_deployment, modality="IMAGE"):
                      raise HTTPException(
                          status_code=400,
                          detail=f"Multimodal message (Image) is not supported by deployment '{model_deployment}' according to configuration."
                      )
                 image_url = part.image_url.url
                 if image_url.startswith("http://") or image_url.startswith("https://"):
                      content_parts.append({
                          "type": "image_url",
                          "image_url": {"url": image_url}
                      })
                 elif image_url.startswith("data:image"):
                       content_parts.append({
                           "type": "image_url",
                           "image_url": {"url": image_url}
                       })
                 else:
                      logger.warning("Image content is not a URL or data URI, attempting to treat as raw base64.")
                      try:
                          base64_data = base64.b64encode(base64.b64decode(image_url)).decode('utf-8') # Validate/re-encode
                          data_uri = f"data:image/jpeg;base64,{base64_data}"
                          content_parts.append({
                              "type": "image_url",
                              "image_url": {"url": data_uri}
                           })
                      except Exception as e:
                          logger.error(f"Failed to process image content: {e}")
                          raise HTTPException(status_code=400, detail="Invalid image content format provided.")

        if not content_parts:
             return [{"type": "text", "text": ""}]

        return content_parts


    @staticmethod
    def is_supported_modality(deployment_name: str, modality: str = "IMAGE") -> bool:
        """Checks if the deployment supports the given modality based on config."""
        model_info = azure_deployments_list.get(deployment_name, {})
        modalities = model_info.get("modalities", [])
        return modality in modalities


    def _convert_tool_spec_for_azure(self, func: Function) -> Dict[str, Any]:
         """ Converts internal Function spec to Azure/OpenAI tool spec format."""
         return {
             "type": "function",
             "function": {
                 "name": func.name,
                 "description": func.description,
                 "parameters": func.parameters, # Should be JSON schema dict
             }
         }

    def _parse_request_for_azure(self, chat_request: ChatRequest) -> dict:
        """Create the request body dictionary for Azure OpenAI Chat Completions API."""
        messages = self._parse_messages_for_azure(chat_request)
        azure_params = {
            "model": chat_request.model, 
            "messages": messages,
        }

        # --- Optional Inference Parameters ---
        if chat_request.temperature is not None:
            azure_params["temperature"] = chat_request.temperature
        if chat_request.max_completion_tokens is not None:
             # Ensure max_completion_tokens is not 0 or negative if provided
             if chat_request.max_completion_tokens > 0:
                 azure_params["max_completion_tokens"] = chat_request.max_completion_tokens
             else:
                 logger.warning(f"Ignoring invalid max_completion_tokens value: {chat_request.max_completion_tokens}")
        if chat_request.top_p is not None:
            azure_params["top_p"] = chat_request.top_p
        if chat_request.stop is not None:
            azure_params["stop"] = chat_request.stop # string or list[string]

        # Add other common OpenAI parameters if they exist on ChatRequest
        if hasattr(chat_request, 'frequency_penalty') and chat_request.frequency_penalty is not None:
            azure_params["frequency_penalty"] = chat_request.frequency_penalty
        if hasattr(chat_request, 'presence_penalty') and chat_request.presence_penalty is not None:
            azure_params["presence_penalty"] = chat_request.presence_penalty
        if hasattr(chat_request, 'seed') and chat_request.seed is not None:
            azure_params["seed"] = chat_request.seed
        if hasattr(chat_request, 'user') and chat_request.user is not None:
             azure_params["user"] = chat_request.user
        if chat_request.tools:
            azure_params["tools"] = [
                self._convert_tool_spec_for_azure(t.function) for t in chat_request.tools
            ]
            if chat_request.tool_choice:
                 azure_params["tool_choice"] = chat_request.tool_choice

        return azure_params

    def _convert_finish_reason_for_azure(self, azure_finish_reason: str | None) -> str | None:
        """
        Maps Azure OpenAI finish reasons to internal standard if necessary.
        Current mapping seems aligned with OpenAI standard, so direct pass-through is fine.
        If your internal standard differs, add mapping here.
        Example: if internal needs 'length' instead of 'max_completion_tokens'
        mapping = { "length": "length", "stop": "stop", "tool_calls": "tool_calls", ... }
        return mapping.get(azure_finish_reason)
        """
        if azure_finish_reason == "function_call": # Handle legacy if needed
             return "tool_calls"
        return azure_finish_reason


    def _create_response_from_azure(self, azure_response) -> ChatResponse:
        """ Creates a standardized ChatResponse object from an Azure OpenAI API response."""
        if not azure_response or not azure_response.choices:
            raise ValueError("Invalid or empty Azure OpenAI response received.")

        choice = azure_response.choices[0]
        message_data = choice.message 
        finish_reason = choice.finish_reason
        usage_data = azure_response.usage 

        # --- Create the core message object ---
        message = ChatResponseMessage(
            role=message_data.role or "assistant",
            content=message_data.content,
            tool_calls=None 
        )

        # --- Check for Tool Calls ---
        if message_data.tool_calls:
            internal_tool_calls = []
            for tc in message_data.tool_calls: 
                if tc.type == 'function':
                     internal_tool_calls.append(
                         ToolCall(
                             id=tc.id,
                             type=tc.type,
                             function=ResponseFunction( 
                                 name=tc.function.name,
                                 arguments=tc.function.arguments 
                             )
                         )
                     )
                else:
                     logger.warning(f"Unsupported tool call type received: {tc.type}")

            message.tool_calls = internal_tool_calls
            message.content = None 
        input_tokens = usage_data.prompt_tokens if usage_data else 0
        output_tokens = usage_data.completion_tokens if usage_data else 0
        total_tokens = usage_data.total_tokens if usage_data else input_tokens + output_tokens

        # --- Construct the final ChatResponse ---
        response = ChatResponse(
            id=azure_response.id,
            model=azure_response.model,
            choices=[
                Choice(
                    index=choice.index,
                    message=message,
                    finish_reason=self._convert_finish_reason_for_azure(finish_reason),
                    logprobs=choice.logprobs, 
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
            system_fingerprint=azure_response.system_fingerprint,
            object=azure_response.object, 
            created=azure_response.created,
        )
        return response


    def _create_response_stream_from_azure(
        self, model_id: str, message_id: str, chunk
    ) -> ChatStreamResponse | None:
        """ Parses an Azure OpenAI stream chunk into ChatStreamResponse. """
        if not chunk or not chunk.choices:
             if chunk and chunk.usage: 
                  return ChatStreamResponse(
                      id=message_id,
                      model=model_id,
                      choices=[], 
                      usage=Usage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens
                        ),
                    system_fingerprint="", 
                    object=chunk.object,
                    created=chunk.created,
                  )
             return None 

        # We are interested in the delta
        delta = chunk.choices[0].delta 
        finish_reason = chunk.choices[0].finish_reason
        logprobs = chunk.choices[0].logprobs 
        delta_message = ChatResponseMessage(
             role=delta.role,
             content=delta.content,
             tool_calls=None
        )

        if delta.tool_calls:
            internal_tool_calls_delta = []
            for tc_delta in delta.tool_calls: 
                 internal_tool_calls_delta.append(
                      ToolCall( 
                          index=tc_delta.index,
                          id=tc_delta.id,
                          type=tc_delta.type,
                          function=ResponseFunction(
                              name=tc_delta.function.name if tc_delta.function else None,
                              arguments=tc_delta.function.arguments if tc_delta.function else None
                          )
                      )
                 )
            delta_message.tool_calls = internal_tool_calls_delta
            delta_message.content = None 
        if delta_message.content is None and delta_message.tool_calls is None and delta_message.role is None:
             if finish_reason is None:
                  return None


        # --- Construct the stream response ---
        stream_response = ChatStreamResponse(
            id=message_id,
            model=model_id,
            choices=[
                ChoiceDelta(
                    index=chunk.choices[0].index,
                    delta=delta_message,
                    logprobs=logprobs,
                    finish_reason=self._convert_finish_reason_for_azure(finish_reason),
                )
            ],
            usage=None,
            system_fingerprint=chunk.system_fingerprint or "",  # Default to empty string
            object=chunk.object,
            created=chunk.created,
        )
        return stream_response
# --- Embeddings ---

class AzureOpenAIEmbeddingsModel(BaseEmbeddingsModel, ABC):

    def __init__(self):
        if not azure_openai_client:
            raise RuntimeError("AzureOpenAI sync client not initialized. Check configuration.")
        self.client = azure_openai_client

    def _parse_args_for_azure(self, embeddings_request: EmbeddingsRequest) -> dict:
        """Parses input for Azure OpenAI embeddings."""
        input_data = embeddings_request.input
        if isinstance(input_data, Iterable) and not isinstance(input_data, (str, list)):
             logger.warning("Iterable input detected for embeddings. Decoding using tiktoken.")
             texts = []
             encodings = []
             is_nested_int = False
             try:
                  first_item = next(iter(input_data), None)
                  is_nested_int = isinstance(first_item, Iterable) and not isinstance(first_item, str)

                  if is_nested_int:
                      for inner in input_data:
                           if isinstance(inner, Iterable) and not isinstance(inner, str):
                                texts.append(ENCODER.decode(list(inner)))
                           else:
                                raise ValueError("Mixed int and Iterable[int] in input.")
                  else: 
                      encodings = list(input_data)
                      if encodings:
                           texts.append(ENCODER.decode(encodings))
                  input_data = texts
             except Exception as e:
                 logger.error(f"Failed to decode token input for embeddings: {e}")
                 raise HTTPException(status_code=400, detail="Invalid tokenized input format for embeddings.")

        if not isinstance(input_data, (str, list)):
            raise HTTPException(status_code=400, detail="Embeddings input must be a string or list of strings.")
        if isinstance(input_data, list) and not all(isinstance(item, str) for item in input_data):
             raise HTTPException(status_code=400, detail="If embeddings input is a list, all items must be strings.")


        args = {
            "model": embeddings_request.model,
            "input": input_data,
        }

        if hasattr(embeddings_request, 'dimensions') and embeddings_request.dimensions:
             args["dimensions"] = embeddings_request.dimensions

        return args

    def _create_response_from_azure_embeddings(
        self,
        azure_response, 
        model_deployment: str,
        encoding_format: Literal["float", "base64"] = "float",
    ) -> EmbeddingsResponse:
        """Creates EmbeddingsResponse from Azure OpenAI embeddings result."""

        data = []
        for i, embedding_data in enumerate(azure_response.data):
            embedding_value = embedding_data.embedding 

            if encoding_format == "base64":
                arr = np.array(embedding_value, dtype=np.float32)
                arr_bytes = arr.tobytes()
                encoded_embedding = base64.b64encode(arr_bytes).decode('utf-8')
                data.append(Embedding(index=embedding_data.index, embedding=encoded_embedding))
            else: 
                data.append(Embedding(index=embedding_data.index, embedding=embedding_value))

        usage_data = azure_response.usage 

        response = EmbeddingsResponse(
            data=data,
            model=azure_response.model, 
            usage=EmbeddingsUsage(
                prompt_tokens=usage_data.prompt_tokens,
                total_tokens=usage_data.total_tokens, 
            ),
            object=azure_response.object 
        )
        if DEBUG:
             log_data = response.model_dump()
             log_data['data'] = f"<{len(log_data.get('data',[]))} embeddings>"
             logger.info("Embeddings Proxy response :" + json.dumps(log_data))
        return response

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Generates embeddings using Azure OpenAI."""
        if embeddings_request.model not in azure_deployments_list or azure_deployments_list[embeddings_request.model].get("type") != "embedding":
             raise HTTPException(status_code=400, detail=f"Deployment '{embeddings_request.model}' is not a known or valid embedding deployment.")

        args = self._parse_args_for_azure(embeddings_request)
        if DEBUG:
             log_args = args.copy()
             log_args['input'] = "<input data>" # Redact input for logs
             logger.info("Azure Embeddings Request: " + json.dumps(log_args))

        try:
        
            azure_response = self.client.embeddings.create(**args)
        except BadRequestError as e:
            logger.error(f"Azure OpenAI Embeddings Bad Request Error (400): {e}")
            detail = f"Azure OpenAI Embeddings request failed: {e.message}"
            try:
                err_body = e.response.json()
                detail = err_body.get("error", {}).get("message", detail)
            except: pass
            raise HTTPException(status_code=400, detail=detail)
        except RateLimitError as e:
            logger.error(f"Azure OpenAI Embeddings Rate Limit Error (429): {e}")
            raise HTTPException(status_code=429, detail=f"Azure OpenAI Embeddings rate limit exceeded: {e.message}")
        except APIError as e:
             logger.error(f"Azure OpenAI Embeddings API Error ({e.status_code}): {e}")
             raise HTTPException(status_code=e.status_code or 500, detail=f"Azure OpenAI Embeddings API error: {e.message}")
        except Exception as e:
             logger.error(f"Unexpected error calling Azure OpenAI Embeddings: {e}")
             raise HTTPException(status_code=500, detail=f"An unexpected error occurred during embeddings: {str(e)}")


        return self._create_response_from_azure_embeddings(
            azure_response,
            model_deployment=embeddings_request.model,
            encoding_format=embeddings_request.encoding_format
        )

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
             return base64.b64encode(obj).decode('utf-8')
        return json.JSONEncoder.default(self, obj)


def get_chat_model(model_id: str) -> AzureOpenAIChatModel:
     if model_id in [name for name, info in azure_deployments_list.items() if info.get("type") == "chat"]:
         return AzureOpenAIChatModel()
     else:
          raise HTTPException(status_code=404, detail=f"Chat deployment not found: {model_id}")

def get_embeddings_model(model_id: str) -> AzureOpenAIEmbeddingsModel:
     if model_id in [name for name, info in azure_deployments_list.items() if info.get("type") == "embedding"]:
         return AzureOpenAIEmbeddingsModel()
     else:
         raise HTTPException(status_code=404, detail=f"Embeddings deployment not found: {model_id}")