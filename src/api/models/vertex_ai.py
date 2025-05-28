import json
import logging
import os
import base64
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import AsyncIterable, List, Dict, Any, Optional, Union, Literal

# Imports from api.models.base and api.schema
from api.setting import VERTEX_AI_PROJECT_ID, VERTEX_AI_LOCATION, DEFAULT_VERTEX_AI_CHAT_MODEL, DEFAULT_VERTEX_AI_EMBEDDING_MODEL
from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.schema import (
    ChatRequest,
    ChatResponse,
    ChatStreamResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    SystemMessage, # Assuming SystemMessage might be needed
    TextContent,
    ImageContent,
    ToolCall,
    ResponseFunction, # Assuming this maps to a tool's function declaration
    Usage,
    Error,
    ErrorMessage,
    ModelTypeEnum, # Changed from ModelType
    ModelCard,
    # ModelsResponse is removed
)

# Google Cloud and Vertex AI imports
import google.cloud.aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content, Tool, FunctionDeclaration
from vertexai.language_models import TextEmbeddingModel

# Initialize logger
logger = logging.getLogger(__name__)

# Updated known Vertex AI models using imported defaults
KNOWN_VERTEX_AI_MODELS: Dict[str, Dict[str, Any]] = {
    DEFAULT_VERTEX_AI_CHAT_MODEL: {'type': 'chat', 'modalities': ['TEXT', 'IMAGE'], 'context_length': 8192}, # Default chat
    'gemini-1.0-pro-001': {'type': 'chat', 'modalities': ['TEXT'], 'context_length': 8192}, # Example specific chat
    'gemini-1.5-pro-001': {"type": "chat", "modalities": ["TEXT", "IMAGE"], 'context_length': 30720}, # Example specific chat with larger context
    DEFAULT_VERTEX_AI_EMBEDDING_MODEL: {'type': 'embedding', 'output_dimensions': 768}, # Default embedding
    'textembedding-gecko-multilingual@001': {'type': 'embedding', 'output_dimensions': 768}, # Example specific embedding
    # Add other commonly used Vertex AI models here, e.g.:
    # 'claude-3-sonnet@20240229': {'type': 'chat', 'modalities': ['TEXT', 'IMAGE'], 'context_length': 200000}, # Not a Google model, example of structure
}

# --- Helper Functions (if any, or can be part of classes) ---

def _convert_finish_reason_from_vertex(reason: Optional[str]) -> Optional[str]:
    """Converts Vertex AI finish reason to our schema's finish reason."""
    if not reason:
        return None
    # This is a placeholder, actual mapping will depend on Vertex AI SDK
    mapping = {
        "MAX_TOKENS": "length",
        "STOP": "stop",
        "SAFETY": "content_filter",
        "RECITATION": "recitation", # Example, might not be direct
        "OTHER": "unknown", # Example
        # Add more mappings as needed
    }
    return mapping.get(str(reason).upper(), "unknown")

def _parse_content_parts_from_vertex(parts: List[Part]) -> List[Union[TextContent, ImageContent]]:
    """Parses content parts from Vertex AI response to our schema."""
    # Placeholder implementation
    content_list = []
    for part in parts:
        if part.text:
            content_list.append(TextContent(text=part.text))
        # Add handling for other part types like images if necessary
        # elif part.inline_data and part.inline_data.mime_type.startswith("image/"):
        #     content_list.append(ImageContent(url=f"data:{part.inline_data.mime_type};base64,{base64.b64encode(part.inline_data.data).decode()}"))
    return content_list

# --- Chat Model Implementation ---
class VertexAIChatModel(BaseChatModel):
    def __init__(self, model_id: str):
        self.model_id = model_id
        if not model_id in KNOWN_VERTEX_AI_MODELS or KNOWN_VERTEX_AI_MODELS[model_id]['type'] != 'chat':
            # This check could also be in a factory or registry
            raise ValueError(f"Invalid or unknown chat model_id for Vertex AI: {model_id}")
        try:
            # Ensure Vertex AI is initialized
            vertexai.init(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_LOCATION)
            # Initialize the Vertex AI GenerativeModel client
            self.client = GenerativeModel(model_id)
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI GenerativeModel for {model_id}: {e}")
            raise

    async def list_models(self) -> List[ModelCard]:
        """Lists available Vertex AI chat models."""
        return [
            ModelCard(id=model_id, type=ModelTypeEnum.CHAT, **details)
            for model_id, details in KNOWN_VERTEX_AI_MODELS.items()
            if details['type'] == 'chat'
        ]

    def _validate_chat_request(self, chat_request: ChatRequest):
        """Validates the chat request against known models."""
        if chat_request.model not in KNOWN_VERTEX_AI_MODELS:
            raise ValueError(f"Unknown model: {chat_request.model}")
        if KNOWN_VERTEX_AI_MODELS[chat_request.model]['type'] != 'chat':
            raise ValueError(f"Model {chat_request.model} is not a chat model.")
        # Further validation for modalities can be added here if needed
        # e.g., check if ImageContent is provided for a model not supporting images.

    def _parse_messages_for_vertex(self, chat_request: ChatRequest) -> List[Content]:
        """Converts API messages to Vertex AI Content objects."""
        vertex_messages: List[Content] = []
        system_prompt_parts: List[Part] = []

        for message in chat_request.messages:
            role = None
            parts: List[Part] = []

            if isinstance(message, SystemMessage): # Assuming SystemMessage exists
                # Vertex AI combines system instructions with the first user message or as a separate instruction set.
                # For simplicity, let's assume system prompts are handled by the GenerativeModel's system_instruction parameter.
                # Or, they can be prepended to the parts of the first user message.
                # Here, we'll collect them to be passed to `system_instruction` if the model supports it.
                # This part needs to align with how `GenerativeModel` handles system prompts.
                # For now, let's assume they are added as parts.
                if isinstance(message.content, str):
                    system_prompt_parts.append(Part.from_text(message.content))
                elif isinstance(message.content, List): # List of TextContent/ImageContent
                     for content_item in message.content:
                        if isinstance(content_item, TextContent):
                            system_prompt_parts.append(Part.from_text(content_item.text))
                        # Image handling for system prompt might be unusual, but included for completeness
                        elif isinstance(content_item, ImageContent):
                            # Assuming ImageContent.url is a base64 encoded image or a GCS URI
                            match = re.match(r"data:(image/\w+);base64,(.*)", content_item.url)
                            if match:
                                mime_type, b64_data = match.groups()
                                image_data = base64.b64decode(b64_data)
                                system_prompt_parts.append(Part.from_data(data=image_data, mime_type=mime_type))
                            elif content_item.url.startswith("gs://"):
                                system_prompt_parts.append(Part.from_uri(uri=content_item.url, mime_type="image/png")) # Assuming png, adjust as needed
                            else:
                                logger.warning(f"Unsupported image URL format in SystemMessage: {content_item.url}")
                continue # System messages are handled differently, not as a separate Content object in the list

            elif isinstance(message, UserMessage):
                role = "user"
                if isinstance(message.content, str):
                    parts.append(Part.from_text(message.content))
                elif isinstance(message.content, List):
                    for content_item in message.content:
                        if isinstance(content_item, TextContent):
                            parts.append(Part.from_text(content_item.text))
                        elif isinstance(content_item, ImageContent):
                            # Example: "data:image/png;base64,iVBORw0KGgo..." or "gs://bucket/object"
                            match = re.match(r"data:(image/\w+);base64,(.*)", content_item.url)
                            if match:
                                mime_type, b64_data = match.groups()
                                image_data = base64.b64decode(b64_data)
                                parts.append(Part.from_data(data=image_data, mime_type=mime_type))
                            elif content_item.url.startswith("gs://"):
                                # Extract GCS URI and attempt to infer mime_type or require it
                                # For simplicity, assuming a common image type if not specified
                                # Mime type might need to be provided or inferred more robustly
                                parts.append(Part.from_uri(uri=content_item.url, mime_type="image/png")) # Placeholder mime_type
                            else:
                                logger.warning(f"Unsupported image URL format: {content_item.url}")
                                parts.append(Part.from_text(f"[Unsupported image: {content_item.url}]"))


            elif isinstance(message, AssistantMessage):
                role = "model" # Vertex AI uses "model" for assistant
                if message.content: # Assistant message can be simple text
                     parts.append(Part.from_text(message.content))
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        # This is for when the *assistant* calls a tool.
                        # The Part should represent the function call itself.
                        # Placeholder: actual representation depends on Vertex AI SDK for tool calls by assistant
                        parts.append(Part.from_function_call(
                            name=tool_call.function.name,
                            args=json.loads(tool_call.function.arguments) # Assuming arguments is a JSON string
                        ))

            elif isinstance(message, ToolMessage):
                # This is the result of a tool execution, provided by the user.
                # Vertex AI expects this as a Part with a FunctionResponse.
                # The role should be "user" or "function" depending on SDK version.
                # Let's assume it's a part of a "user" message or a dedicated "tool" role Content.
                # For now, adding as a special part. The `role` for Content containing tool responses
                # is typically `user` or a dedicated role if the SDK supports it.
                # Let's assume for now it's part of a "user" turn, or we use specific role if available.
                # The Vertex SDK expects tool results as `Part.from_function_response`.
                # These are then added to a `Content` object, typically with `role="user"` or `role="function"`.
                # Let's create a new content object for each tool message for clarity.
                vertex_messages.append(Content(
                    role="tool", # Or "user" - consult Vertex AI documentation for `Content(role=...)` with tool responses
                    parts=[Part.from_function_response(
                        name=message.tool_call_id, # This should map to the function name that was called
                        response={"content": message.content} # The response structure for Vertex AI
                    )]
                ))
                continue # Skip creating a generic Content object below for ToolMessage

            if role and parts:
                vertex_messages.append(Content(role=role, parts=parts))
            elif not role and not parts and not isinstance(message, SystemMessage): # Ensure not to add empty content unless it's a tool message handled above
                 logger.warning(f"Skipping message of type {type(message)} as it resulted in no role or parts.")


        # Prepend system prompts if any, Vertex AI's `GenerativeModel` takes `system_instruction`
        # So this list might not be directly used here but passed to `generate_content_async`
        # For now, if system_prompt_parts exist, we assume they are handled by the `system_instruction` param of the model.
        # If the model doesn't have `system_instruction`, they might need to be the first `Content` object or part of the first user message.
        # This method returns List[Content] which is the history. System instructions are separate.
        return vertex_messages

    def _get_system_instructions(self, chat_request: ChatRequest) -> Optional[Content]:
        """Extracts system instructions from the chat request."""
        system_parts = []
        for message in chat_request.messages:
            if isinstance(message, SystemMessage):
                if isinstance(message.content, str):
                    system_parts.append(Part.from_text(message.content))
                elif isinstance(message.content, list):
                    for item in message.content:
                        if isinstance(item, TextContent):
                            system_parts.append(Part.from_text(item.text))
                        # Potentially handle ImageContent in system prompts if supported/needed
                        # For now, focusing on text for system instructions.
        return Content(parts=system_parts, role="system") if system_parts else None


    def _parse_tools_for_vertex(self, chat_request: ChatRequest) -> Optional[List[Tool]]:
        """Converts API tools to Vertex AI Tool objects."""
        if not chat_request.tools:
            return None
        
        vertex_tools: List[Tool] = []
        for tool_config in chat_request.tools:
            if tool_config.type == "function" and tool_config.function:
                # Assuming tool_config.function.parameters is a JSON schema dict
                function_decl = FunctionDeclaration(
                    name=tool_config.function.name,
                    description=tool_config.function.description,
                    parameters=tool_config.function.parameters # This should be a dict representing JSON schema
                )
                vertex_tools.append(Tool(function_declarations=[function_decl]))
            else:
                logger.warning(f"Unsupported tool type: {tool_config.type}")
        return vertex_tools if vertex_tools else None

    async def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Handles a non-streaming chat request."""
        self._validate_chat_request(chat_request)
        
        vertex_messages = self._parse_messages_for_vertex(chat_request)
        vertex_tools = self._parse_tools_for_vertex(chat_request)
        system_instructions = self._get_system_instructions(chat_request)

        generation_config = {}
        if chat_request.temperature is not None:
            generation_config["temperature"] = chat_request.temperature
        if chat_request.max_tokens is not None:
            generation_config["max_output_tokens"] = chat_request.max_tokens
        if chat_request.top_p is not None:
            generation_config["top_p"] = chat_request.top_p
        # top_k, stop_sequences might also be configurable

        try:
            # Placeholder for actual SDK call
            # response = await self.client.generate_content_async(
            #     contents=vertex_messages,
            #     tools=vertex_tools,
            #     system_instruction=system_instructions,
            #     generation_config=generation_config,
            #     stream=False
            # )
            
            # --- Placeholder Response ---
            logger.info(f"Simulating Vertex AI call for model {self.model_id} with messages: {vertex_messages}, tools: {vertex_tools}, system_instructions: {system_instructions}")
            time.sleep(0.1) # Simulate network latency
            
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created_timestamp = int(time.time())
            
            # Simulate a response that could involve a tool call or simple text
            simulated_content_parts = [Part.from_text("This is a simulated response from Vertex AI.")]
            simulated_finish_reason = "STOP"
            
            # Example: Simulate a tool call response
            # if vertex_tools and chat_request.tool_choice == "auto" or isinstance(chat_request.tool_choice, dict):
            #     first_tool_name = vertex_tools[0].function_declarations[0].name
            #     simulated_content_parts = [
            #         Part.from_function_call(name=first_tool_name, args={"param1": "value1"})
            #     ]
            #     simulated_finish_reason = "TOOL_CALL" # Or equivalent in Vertex AI

            # Mocked raw response object structure (highly simplified)
            class MockVertexResponse:
                def __init__(self, parts, finish_reason_str, usage_metadata=None):
                    self.candidates = [MockCandidate(parts, finish_reason_str)]
                    self.usage_metadata = usage_metadata if usage_metadata else {"prompt_token_count": 10, "candidates_token_count": 20, "total_token_count": 30}

                @property
                def text(self) -> Optional[str]: # For simple text responses (older models)
                    if self.candidates and self.candidates[0].content.parts:
                        return self.candidates[0].content.parts[0].text
                    return None


            class MockCandidate:
                def __init__(self, parts, finish_reason_str):
                    self.content = Content(role="model", parts=parts)
                    self.finish_reason = finish_reason_str # e.g., "STOP", "MAX_TOKENS"

            response = MockVertexResponse(simulated_content_parts, simulated_finish_reason)
            # --- End Placeholder Response ---

            first_candidate = response.candidates[0]
            
            message_content: Optional[str] = None
            tool_calls: Optional[List[ToolCall]] = None

            parsed_parts = _parse_content_parts_from_vertex(first_candidate.content.parts)
            # We need to reconstruct the message content based on the parts.
            # For now, let's assume simple text or a single tool call.
            
            text_parts_collected = []
            for part in first_candidate.content.parts:
                if part.text:
                    text_parts_collected.append(part.text)
                elif part.function_call:
                    if not tool_calls: tool_calls = []
                    tool_calls.append(ToolCall(
                        id=f"call_{uuid.uuid4().hex}", # Vertex might provide an ID, or we generate one
                        type="function",
                        function=ResponseFunction(
                            name=part.function_call.name,
                            arguments=json.dumps(part.function_call.args)
                        )
                    ))
            
            if text_parts_collected:
                message_content = "".join(text_parts_collected)

            api_message = AssistantMessage(
                id=f"msg_{uuid.uuid4().hex}",
                role="assistant",
                content=message_content,
                tool_calls=tool_calls
            )
            
            finish_reason = _convert_finish_reason_from_vertex(str(first_candidate.finish_reason))

            # Placeholder for usage, actual fields depend on Vertex AI response
            usage = Usage(
                prompt_tokens=response.usage_metadata.get("prompt_token_count", 0),
                completion_tokens=response.usage_metadata.get("candidates_token_count", 0), # Vertex uses "candidates_token_count"
                total_tokens=response.usage_metadata.get("total_token_count", 0)
            )

            return ChatResponse(
                id=response_id,
                object="chat.completion",
                created=created_timestamp,
                model=self.model_id,
                choices=[{
                    "index": 0,
                    "message": api_message,
                    "finish_reason": finish_reason,
                }],
                usage=usage,
            )

        except Exception as e:
            logger.error(f"Error during Vertex AI chat: {e}")
            # Consider re-raising or returning a ChatResponse with an error message
            # For now, let's create an error response
            return ChatResponse(
                id=f"err-{uuid.uuid4().hex}",
                object="chat.completion",
                created=int(time.time()),
                model=self.model_id,
                choices=[], # No choices on error
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                error=Error(
                    message=str(e),
                    type=type(e).__name__,
                    code=None # No standard HTTP-like code here, unless we map Vertex errors
                )
            )

    async def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Handles a streaming chat request."""
        self._validate_chat_request(chat_request)

        vertex_messages = self._parse_messages_for_vertex(chat_request)
        vertex_tools = self._parse_tools_for_vertex(chat_request)
        system_instructions = self._get_system_instructions(chat_request)

        generation_config = {}
        if chat_request.temperature is not None:
            generation_config["temperature"] = chat_request.temperature
        if chat_request.max_tokens is not None:
            generation_config["max_output_tokens"] = chat_request.max_tokens
        # Add other params as needed

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_timestamp = int(time.time())

        try:
            # Placeholder for actual SDK call
            # stream_response = await self.client.generate_content_async(
            #     contents=vertex_messages,
            #     tools=vertex_tools,
            #     system_instruction=system_instructions,
            #     generation_config=generation_config,
            #     stream=True
            # )
            
            # --- Placeholder Stream ---
            logger.info(f"Simulating Vertex AI stream for model {self.model_id} with messages: {vertex_messages}, tools: {vertex_tools}")
            
            class MockVertexStreamChunk:
                def __init__(self, text_delta=None, tool_call_chunks=None, finish_reason_str=None, usage_metadata=None, is_last_chunk=False):
                    self.delta_text = text_delta # Simplified: Vertex might stream parts, not just text delta
                    self.delta_tool_calls = tool_call_chunks # Simplified
                    self.finish_reason_str = finish_reason_str
                    self.usage_metadata = usage_metadata
                    self.is_last_chunk = is_last_chunk

                    # Mimic structure of real response if possible
                    self.candidates = []
                    if text_delta or tool_call_chunks:
                         parts = []
                         if text_delta: parts.append(Part.from_text(text_delta))
                         # if tool_call_chunks: ... (more complex to model tool call streaming accurately)
                         self.candidates.append(MockCandidate(parts, finish_reason_str if is_last_chunk else None))


            async def mock_stream_generator():
                yield MockVertexStreamChunk(text_delta="Hello, ")
                await asyncio.sleep(0.05)
                yield MockVertexStreamChunk(text_delta="this is ")
                await asyncio.sleep(0.05)
                yield MockVertexStreamChunk(text_delta="Vertex AI ", finish_reason_str="STOP", usage_metadata={"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}, is_last_chunk=True)

            # stream_response = mock_stream_generator() # Use this if you make the generator async
            # For now, a simple list to iterate over for the placeholder
            simulated_stream_chunks = [
                MockVertexStreamChunk(text_delta="Hello, "),
                MockVertexStreamChunk(text_delta="this is "),
                MockVertexStreamChunk(text_delta="Vertex AI streaming."),
                # Example of tool call chunk (conceptual)
                # MockVertexStreamChunk(tool_call_chunks=[{"index": 0, "id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments_delta": '{"loc'}}]),
                # MockVertexStreamChunk(tool_call_chunks=[{"index": 0, "id": "call_abc", "type": "function", "function": {"arguments_delta": 'ation":"SF"}'}}]),
                MockVertexStreamChunk(finish_reason_str="STOP", usage_metadata={"prompt_token_count": 10, "candidates_token_count": 15, "total_token_count": 25}, is_last_chunk=True) # Last chunk with reason and usage
            ]
            # --- End Placeholder Stream ---
            
            # async for chunk in stream_response: # This would be the actual loop
            for i, chunk in enumerate(simulated_stream_chunks): # Placeholder loop
                # This mapping logic is highly dependent on the actual structure of Vertex AI stream chunks
                delta_content: Optional[str] = None
                delta_tool_calls: Optional[List[ToolCall]] = None # Or ToolCallChunk for streaming
                finish_reason: Optional[str] = None
                usage: Optional[Usage] = None

                # Assuming chunk has a structure similar to the non-streaming response but with deltas
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    first_candidate = chunk.candidates[0]
                    if first_candidate.content and first_candidate.content.parts:
                        # Simplified: assume first part is text delta
                        delta_content = first_candidate.content.parts[0].text
                    
                    if chunk.is_last_chunk: # Check if it's the last conceptual chunk
                        finish_reason = _convert_finish_reason_from_vertex(str(chunk.finish_reason_str))
                        if chunk.usage_metadata:
                             usage = Usage(
                                prompt_tokens=chunk.usage_metadata.get("prompt_token_count", 0),
                                completion_tokens=chunk.usage_metadata.get("candidates_token_count", 0),
                                total_tokens=chunk.usage_metadata.get("total_token_count", 0)
                            )

                # Fallback if structure is simpler (like just delta_text on chunk)
                elif hasattr(chunk, 'delta_text') and chunk.delta_text:
                    delta_content = chunk.delta_text
                
                if hasattr(chunk, 'is_last_chunk') and chunk.is_last_chunk:
                    finish_reason = _convert_finish_reason_from_vertex(str(chunk.finish_reason_str))
                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        usage = Usage(
                            prompt_tokens=chunk.usage_metadata.get("prompt_token_count", 0),
                            completion_tokens=chunk.usage_metadata.get("candidates_token_count", 0),
                            total_tokens=chunk.usage_metadata.get("total_token_count", 0)
                        )
                
                # TODO: Map tool call deltas if Vertex AI streams them
                # if hasattr(chunk, 'delta_tool_calls') and chunk.delta_tool_calls:
                #   delta_tool_calls = ...

                stream_chunk_response = ChatStreamResponse(
                    id=response_id,
                    object="chat.completion.chunk",
                    created=created_timestamp,
                    model=self.model_id,
                    choices=[{
                        "index": 0,
                        "delta": AssistantMessage(role="assistant", content=delta_content, tool_calls=delta_tool_calls), # tool_calls might be partial
                        "finish_reason": finish_reason,
                        # Potentially logprobs, etc.
                    }],
                    usage=usage # Typically sent only in the last chunk or not at all in some APIs for stream
                )
                yield self.stream_response_to_bytes(stream_chunk_response)
                await asyncio.sleep(0.01) # simulate async nature if using sync list

            yield self.stream_response_to_bytes("[DONE]")

        except Exception as e:
            logger.error(f"Error during Vertex AI chat stream: {e}")
            error_response = ChatStreamResponse(
                id=response_id,
                object="chat.completion.chunk",
                created=created_timestamp,
                model=self.model_id,
                choices=[],
                error=Error(message=str(e), type=type(e).__name__, code=None)
            )
            yield self.stream_response_to_bytes(error_response)
            yield self.stream_response_to_bytes("[DONE]")


# --- Embeddings Model Implementation ---
class VertexAIEmbeddingsModel(BaseEmbeddingsModel):
    def __init__(self, model_id: str):
        self.model_id = model_id
        if not model_id in KNOWN_VERTEX_AI_MODELS or KNOWN_VERTEX_AI_MODELS[model_id]['type'] != 'embedding':
            raise ValueError(f"Invalid or unknown embedding model_id for Vertex AI: {model_id}")
        try:
            # Ensure Vertex AI is initialized
            vertexai.init(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_LOCATION)
            # Initialize the Vertex AI TextEmbeddingModel client
            self.client = TextEmbeddingModel.from_pretrained(model_id)
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI TextEmbeddingModel for {model_id}: {e}")
            raise

    async def list_models(self) -> List[ModelCard]:
        """Lists available Vertex AI embedding models."""
        return [
            ModelCard(id=model_id, type=ModelTypeEnum.EMBEDDING, **details)
            for model_id, details in KNOWN_VERTEX_AI_MODELS.items()
            if details['type'] == 'embedding'
        ]

    def _validate_embedding_request(self, embeddings_request: EmbeddingsRequest):
        """Validates the embedding request."""
        if embeddings_request.model not in KNOWN_VERTEX_AI_MODELS:
            raise ValueError(f"Unknown model: {embeddings_request.model}")
        if KNOWN_VERTEX_AI_MODELS[embeddings_request.model]['type'] != 'embedding':
            raise ValueError(f"Model {embeddings_request.model} is not an embedding model.")

    async def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Handles an embedding request."""
        self._validate_embedding_request(embeddings_request)
        
        texts_to_embed: List[str] = []
        if isinstance(embeddings_request.input, str):
            texts_to_embed.append(embeddings_request.input)
        elif isinstance(embeddings_request.input, list):
            texts_to_embed = embeddings_request.input
        else:
            # Should be caught by Pydantic validation, but good to have a safeguard
            raise ValueError("Invalid input type for embeddings. Must be str or List[str].")

        try:
            # Placeholder for actual SDK call
            # embeddings = await self.client.get_embeddings_async(texts_to_embed)
            # The actual call might be self.client.get_embeddings(texts_to_embed, auto_truncate=True)
            # The async version might be slightly different or might need to run in executor.
            # For now, using a sync call in an executor or assuming an async method exists.
            
            # --- Placeholder Response ---
            logger.info(f"Simulating Vertex AI embedding call for model {self.model_id} with {len(texts_to_embed)} texts.")
            await asyncio.sleep(0.05) # Simulate network latency
            
            class MockVertexEmbedding:
                def __init__(self, values, statistics):
                    self.values = values
                    self.statistics = statistics
            
            class MockVertexEmbeddingResponse:
                def __init__(self, text_embedding, statistics):
                    self.text_embedding = text_embedding # This is not how it's usually structured
                    self.values = text_embedding # Actual embeddings
                    self.statistics = statistics


            mock_embeddings_list = []
            output_dims = KNOWN_VERTEX_AI_MODELS[self.model_id].get('output_dimensions', 768)
            for i, text in enumerate(texts_to_embed):
                # Simulate embeddings based on text length or content hash for deterministic testing
                embedding_vector = [hash(text + str(j)) % 10000 / 10000.0 for j in range(output_dims)]
                mock_embeddings_list.append(MockVertexEmbedding(values=embedding_vector, statistics={"token_count": len(text.split()), "truncated": False}))
            
            # The actual response from `get_embeddings` is a list of `TextEmbedding` objects.
            # Each `TextEmbedding` has `values` (the vector) and `statistics`.
            # embeddings_response_sdk = [MockVertexEmbedding(values=[0.1, 0.2] * (output_dims // 2), statistics={"token_count": 10, "truncated": False}) for _ in texts_to_embed]
            embeddings_response_sdk = mock_embeddings_list
            # --- End Placeholder Response ---

            data = []
            total_tokens = 0
            for i, emb_result in enumerate(embeddings_response_sdk):
                data.append({
                    "object": "embedding",
                    "embedding": emb_result.values,
                    "index": i,
                })
                total_tokens += emb_result.statistics.get("token_count", 0) # Summing up token counts from each embedding result

            usage = Usage(
                prompt_tokens=total_tokens, # For embeddings, prompt_tokens often means input tokens
                total_tokens=total_tokens,
                completion_tokens=0 # Not applicable for embeddings
            )

            return EmbeddingsResponse(
                object="list",
                data=data,
                model=self.model_id,
                usage=usage,
            )

        except Exception as e:
            logger.error(f"Error during Vertex AI embedding: {e}")
            # Return an EmbeddingsResponse with an error
            return EmbeddingsResponse(
                object="list",
                data=[],
                model=self.model_id,
                usage=Usage(prompt_tokens=0, total_tokens=0, completion_tokens=0),
                error=Error(
                    message=str(e),
                    type=type(e).__name__,
                    code=None
                )
            )

# --- Factory/Getter Functions ---
def get_chat_model(model_id: str) -> VertexAIChatModel:
    """
    Returns a VertexAIChatModel instance for the given model_id.
    Initializes Vertex AI if not already initialized.
    """
    # One-time initialization
    try:
        vertexai.init(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_LOCATION)
        logger.info(f"Vertex AI SDK initialized with project {VERTEX_AI_PROJECT_ID} and location {VERTEX_AI_LOCATION} from get_chat_model.")
    except Exception as e:
        # It might have been initialized already by a constructor or another call.
        # Or there could be a genuine initialization issue (e.g. missing credentials if not using ADC, or wrong project/location).
        logger.warning(f"Vertex AI SDK initialization issue in get_chat_model: {e}. Assuming it might be already initialized or will use Application Default Credentials.")

    if model_id in KNOWN_VERTEX_AI_MODELS and KNOWN_VERTEX_AI_MODELS[model_id]['type'] == 'chat':
        # The VertexAIChatModel constructor will also call vertexai.init, which is generally safe.
        return VertexAIChatModel(model_id=model_id)
    else:
        raise ValueError(f"Model ID '{model_id}' is not a known Vertex AI chat model.")

def get_embeddings_model(model_id: str) -> VertexAIEmbeddingsModel:
    """
    Returns a VertexAIEmbeddingsModel instance for the given model_id.
    Initializes Vertex AI if not already initialized.
    """
    try:
        vertexai.init(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_LOCATION)
        logger.info(f"Vertex AI SDK initialized with project {VERTEX_AI_PROJECT_ID} and location {VERTEX_AI_LOCATION} from get_embeddings_model.")
    except Exception as e:
        logger.warning(f"Vertex AI SDK initialization issue in get_embeddings_model: {e}. Assuming it might be already initialized or will use Application Default Credentials.")
        
    if model_id in KNOWN_VERTEX_AI_MODELS and KNOWN_VERTEX_AI_MODELS[model_id]['type'] == 'embedding':
        # The VertexAIEmbeddingsModel constructor will also call vertexai.init.
        return VertexAIEmbeddingsModel(model_id=model_id)
    else:
        raise ValueError(f"Model ID '{model_id}' is not a known Vertex AI embedding model.")

# Example of how to ensure asyncio is available for the mock stream.
# This is more for testing the mock; real SDK calls would handle their own async mechanisms.
import asyncio

if __name__ == '__main__':
    # Example Usage (for testing purposes, typically not in the module itself)
    async def main_test():
        # Configure logging for testing
        logging.basicConfig(level=logging.INFO)
        
        # Mock environment variables if not set - now these will be drawn from api.setting effectively
        # os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "your-gcp-project") # No longer needed here if settings are imported
        # os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

        # ---- Test Chat Model ----
        try:
            # Use the default from settings for testing, or a specific one
            chat_model_id = DEFAULT_VERTEX_AI_CHAT_MODEL 
            logger.info(f"Testing with chat model ID: {chat_model_id}")
            chat_model = get_chat_model(chat_model_id)
            
            # Simple chat
            simple_chat_request = ChatRequest(
                model=chat_model_id,
                messages=[UserMessage(content="Hello, Vertex AI!")],
                temperature=0.7
            )
            chat_response = await chat_model.chat(simple_chat_request)
            logger.info(f"Chat Response: {chat_response.model_dump_json(indent=2)}")

            # Streaming chat
            logger.info(f"\n--- Streaming Chat Test ---")
            async for chunk_bytes in chat_model.chat_stream(simple_chat_request):
                if chunk_bytes == b"data: [DONE]\n\n":
                    logger.info("Stream finished.")
                    break
                # Assuming stream_response_to_bytes adds "data: " prefix and "\n\n" suffix
                logger.info(f"Stream Chunk: {chunk_bytes.decode()}")
            
            # Chat with tools (placeholder)
            tool_chat_request = ChatRequest(
                model=chat_model_id,
                messages=[UserMessage(content="What's the weather like in Boston?")],
                tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}]
            )
            tool_response = await chat_model.chat(tool_chat_request)
            logger.info(f"\nTool Chat Response: {tool_response.model_dump_json(indent=2)}")


        except ValueError as ve:
            logger.error(f"Configuration error: {ve}")
        except Exception as e:
            logger.error(f"An error occurred during chat model testing: {e}", exc_info=True)

        # ---- Test Embeddings Model ----
        try:
            embedding_model_id = DEFAULT_VERTEX_AI_EMBEDDING_MODEL
            logger.info(f"Testing with embedding model ID: {embedding_model_id}")
            embed_model = get_embeddings_model(embedding_model_id)
            
            embedding_request = EmbeddingsRequest(
                model=embedding_model_id,
                input=["Hello world", "This is a test."]
            )
            embedding_response = await embed_model.embed(embedding_request)
            logger.info(f"\nEmbedding Response: {embedding_response.model_dump_json(indent=2)}")

        except ValueError as ve:
            logger.error(f"Configuration error: {ve}")
        except Exception as e:
            logger.error(f"An error occurred during embedding model testing: {e}", exc_info=True)

    # Python 3.7+
    # asyncio.run(main_test()) # This line should be commented out for the actual file content.
    pass # End of __main__ example block
