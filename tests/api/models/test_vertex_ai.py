import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock, call # Added call for checking multiple calls
import os
import json # For stream response parsing

# Classes and functions to be tested
from api.models.vertex_ai import (
    VertexAIChatModel,
    VertexAIEmbeddingsModel,
    KNOWN_VERTEX_AI_MODELS,
    get_chat_model,
    get_embeddings_model
)
# Schemas for requests
from api.schema import (
    ChatRequest,
    ChatResponse,
    ChatStreamResponse,
    UserMessage,
    # SystemMessage, # Import if specific tests for system messages are added
    # AssistantMessage, # Import if needed for more complex chat flows
    # ToolMessage, # Import if tool usage is tested
    EmbeddingsRequest,
    EmbeddingsResponse,
    ModelTypeEnum, # Changed from ModelType
    ModelCard,
    # TextContent, # Not directly used in these tests, but good for reference
    # ImageContent, # Not directly used in these tests
    # ToolCall, 
    # ResponseFunction, 
    Usage,
    Error # For error response testing
)
# Settings
from api.setting import VERTEX_AI_PROJECT_ID, VERTEX_AI_LOCATION

# --- Mocked Vertex AI SDK classes (simplified representations) ---
# These help simulate the structure of actual Vertex AI SDK objects.

class MockVertexSDKPart:
    def __init__(self, text=None, function_call=None, inline_data=None, file_data=None, function_response=None):
        self.text = text
        self.function_call = function_call
        self.inline_data = inline_data
        self.file_data = file_data
        self.function_response = function_response

    @classmethod
    def from_text(cls, text: str): return cls(text=text)
    @classmethod
    def from_function_call(cls, name:str, args: dict): return cls(function_call=MagicMock(name=name, args=args))
    @classmethod
    def from_uri(cls, uri: str, mime_type: str): return cls(file_data=MagicMock(file_uri=uri, mime_type=mime_type))
    @classmethod
    def from_data(cls, data: bytes, mime_type: str): return cls(inline_data=MagicMock(data=data, mime_type=mime_type))
    @classmethod
    def from_function_response(cls, name: str, response: dict): return cls(function_response=MagicMock(name=name, response=response))

class MockVertexSDKContent:
    def __init__(self, role: str, parts: list):
        self.role = role
        self.parts = parts

class MockVertexSDKCandidate:
    def __init__(self, content: MockVertexSDKContent, finish_reason: str = "STOP", safety_ratings=None, citation_metadata=None):
        self.content = content
        self.finish_reason = finish_reason
        self.safety_ratings = safety_ratings if safety_ratings is not None else []
        self.citation_metadata = citation_metadata

class MockVertexSDKUsageMetadata:
    def __init__(self, prompt_token_count=10, candidates_token_count=20, total_token_count=30):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count

class MockVertexSDKGenerativeModelResponse:
    def __init__(self, candidates: list, usage_metadata: MockVertexSDKUsageMetadata = None):
        self.candidates = candidates
        self.usage_metadata = usage_metadata if usage_metadata else MockVertexSDKUsageMetadata()
    @property
    def text(self):
        if self.candidates and self.candidates[0].content.parts and self.candidates[0].content.parts[0].text:
            return self.candidates[0].content.parts[0].text
        return None

class MockVertexSDKTextEmbedding:
    def __init__(self, values, statistics):
        self.values = values
        self.statistics = statistics
# --- End of Mocked Vertex AI SDK classes ---

# Determine test model IDs safely
DEFAULT_TEST_CHAT_MODEL_ID = "gemini-1.5-flash-001" # Fallback
DEFAULT_TEST_EMBEDDING_MODEL_ID = "textembedding-gecko@003" # Fallback

try:
    TEST_CHAT_MODEL_ID = next(iter(k for k, v in KNOWN_VERTEX_AI_MODELS.items() if v['type'] == 'chat'))
except StopIteration:
    TEST_CHAT_MODEL_ID = DEFAULT_TEST_CHAT_MODEL_ID
    # Add this default to KNOWN_VERTEX_AI_MODELS for test consistency if it's missing
    if TEST_CHAT_MODEL_ID not in KNOWN_VERTEX_AI_MODELS:
         KNOWN_VERTEX_AI_MODELS[TEST_CHAT_MODEL_ID] = {'type': 'chat', 'modalities': ['TEXT', 'IMAGE'], 'context_length': 8192}


try:
    TEST_EMBEDDING_MODEL_ID = next(iter(k for k, v in KNOWN_VERTEX_AI_MODELS.items() if v['type'] == 'embedding'))
except StopIteration:
    TEST_EMBEDDING_MODEL_ID = DEFAULT_TEST_EMBEDDING_MODEL_ID
    if TEST_EMBEDDING_MODEL_ID not in KNOWN_VERTEX_AI_MODELS:
        KNOWN_VERTEX_AI_MODELS[TEST_EMBEDDING_MODEL_ID] = {'type': 'embedding', 'output_dimensions': 768}

INVALID_MODEL_ID = "this-model-does-not-exist"


# Use new_callable to pass our mock classes directly to be used by the patched targets
@patch('vertexai.generative_models.Part', new_callable=lambda: MockVertexSDKPart)
@patch('vertexai.generative_models.Content', new_callable=lambda: MockVertexSDKContent)
class TestVertexAIChatModel(unittest.IsolatedAsyncioTestCase): # Use IsolatedAsyncioTestCase for async tests

    def _assert_init_called_with_settings(self, mock_vertex_init_func):
        mock_vertex_init_func.assert_any_call(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_LOCATION)

    @patch('vertexai.init')
    @patch('vertexai.generative_models.GenerativeModel')
    def setUp(self, MockSDKGenerativeModel, mock_vertex_ai_init, MockSDKContentUnused, MockSDKPartUnused):
        self.mock_vertex_ai_init_global = mock_vertex_ai_init
        
        # Configure the class-level mocks for Part and Content (done by @patch new_callable)
        # MockSDKPart.from_text = classmethod(MockVertexSDKPart.from_text) ... and so on.
        # This is implicitly handled by new_callable=lambda: MockVertexSDKPart

        self.sdk_client_mock = MockSDKGenerativeModel.return_value
        self.sdk_client_mock.generate_content_async = AsyncMock()

        self.chat_model = VertexAIChatModel(model_id=TEST_CHAT_MODEL_ID)
        self._assert_init_called_with_settings(mock_vertex_ai_init)

    async def test_list_models(self, MockSDKContentUnused, MockSDKPartUnused):
        # This test is already present, needs to be updated for ModelTypeEnum
        # and ensure all fields are checked.
        models = await self.chat_model.list_models()
        self.assertIsInstance(models, list)
        
        expected_chat_models_details = {
            mid: details for mid, details in KNOWN_VERTEX_AI_MODELS.items() if details['type'] == 'chat'
        }
        self.assertEqual(len(models), len(expected_chat_models_details))

        for model_card in models:
            self.assertIsInstance(model_card, ModelCard)
            self.assertEqual(model_card.type, ModelTypeEnum.CHAT)
            self.assertIn(model_card.id, expected_chat_models_details)
            
            # Verify other details from KNOWN_VERTEX_AI_MODELS
            details = expected_chat_models_details[model_card.id]
            self.assertEqual(model_card.modalities, details.get('modalities'))
            self.assertEqual(model_card.context_length, details.get('context_length'))
            # Check common ModelCard fields
            self.assertIsInstance(model_card.created, int)
            self.assertEqual(model_card.object, "model_card")
            self.assertEqual(model_card.owned_by, "general")


    def test_validate_chat_request_valid(self, MockSDKContentUnused, MockSDKPartUnused):
        request = ChatRequest(model=TEST_CHAT_MODEL_ID, messages=[UserMessage(content="Hi")])
        self.chat_model._validate_chat_request(request) # No exception expected

    def test_validate_chat_request_invalid_model_id(self, MockSDKContentUnused, MockSDKPartUnused):
        request = ChatRequest(model=INVALID_MODEL_ID, messages=[UserMessage(content="Hi")])
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            self.chat_model._validate_chat_request(request)

    def test_validate_chat_request_embedding_model_for_chat(self, MockSDKContentUnused, MockSDKPartUnused):
        request = ChatRequest(model=TEST_EMBEDDING_MODEL_ID, messages=[UserMessage(content="Hi")])
        with self.assertRaisesRegex(ValueError, "is not a chat model"):
            self.chat_model._validate_chat_request(request)

    async def test_chat_simple_text_response(self, MockSDKContentUnused, MockSDKPartUnused):
        mock_resp_content = MockVertexSDKContent(role="model", parts=[MockVertexSDKPart(text="Hello from Vertex")])
        mock_candidate = MockVertexSDKCandidate(content=mock_resp_content, finish_reason="STOP")
        mock_sdk_api_response = MockVertexSDKGenerativeModelResponse(
            candidates=[mock_candidate],
            usage_metadata=MockVertexSDKUsageMetadata(5, 10, 15)
        )
        self.sdk_client_mock.generate_content_async.return_value = mock_sdk_api_response

        chat_request = ChatRequest(model=TEST_CHAT_MODEL_ID, messages=[UserMessage(content="Hello")])
        response = await self.chat_model.chat(chat_request)

        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.choices[0].message.content, "Hello from Vertex")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(response.usage.prompt_tokens, 5)
        self.sdk_client_mock.generate_content_async.assert_called_once()

    async def test_chat_stream_response(self, MockSDKContentUnused, MockSDKPartUnused):
        async def mock_sdk_stream_generator():
            yield MockVertexSDKGenerativeModelResponse(candidates=[MockVertexSDKCandidate(content=MockVertexSDKContent(role="model", parts=[MockVertexSDKPart(text="Chunk1 ")]))])
            yield MockVertexSDKGenerativeModelResponse(candidates=[MockVertexSDKCandidate(content=MockVertexSDKContent(role="model", parts=[MockVertexSDKPart(text="Chunk2")]))])
            yield MockVertexSDKGenerativeModelResponse(candidates=[MockVertexSDKCandidate(content=MockVertexSDKContent(role="model", parts=[]), finish_reason="MAX_TOKENS")], usage_metadata=MockVertexSDKUsageMetadata(4, 8, 12))
        
        self.sdk_client_mock.generate_content_async.return_value = mock_sdk_stream_generator()
        chat_request = ChatRequest(model=TEST_CHAT_MODEL_ID, messages=[UserMessage(content="Stream this")], stream=True)
        
        raw_chunks, parsed_chunks = [], []
        async for chunk_bytes in self.chat_model.chat_stream(chat_request):
            raw_chunks.append(chunk_bytes)
            if chunk_bytes == b"data: [DONE]\n\n": break
            json_str = chunk_bytes.decode('utf-8').replace("data: ", "").strip()
            parsed_chunks.append(ChatStreamResponse(**json.loads(json_str)))

        self.assertEqual(len(parsed_chunks), 3)
        self.assertEqual(parsed_chunks[0].choices[0].delta.content, "Chunk1 ")
        self.assertEqual(parsed_chunks[1].choices[0].delta.content, "Chunk2")
        self.assertEqual(parsed_chunks[2].choices[0].finish_reason, "length")
        self.assertEqual(parsed_chunks[2].usage.total_tokens, 12)
        self.assertEqual(raw_chunks[-1], b"data: [DONE]\n\n")

    @patch('vertexai.init')
    @patch('vertexai.generative_models.GenerativeModel')
    async def test_chat_sdk_error_handling(self, MockSDKGenModelError, mock_vertex_init_err, MockSDKContentUnused, MockSDKPartUnused):
        sdk_client_err_mock = MockSDKGenModelError.return_value
        # Ensure the side_effect is an Exception instance
        mocked_exception = Exception("Vertex SDK Internal Error")
        sdk_client_err_mock.generate_content_async = AsyncMock(side_effect=mocked_exception)
        
        with patch('vertexai.generative_models.GenerativeModel', return_value=sdk_client_err_mock):
            error_chat_model = VertexAIChatModel(model_id=TEST_CHAT_MODEL_ID)
        
        response = await error_chat_model.chat(ChatRequest(model=TEST_CHAT_MODEL_ID, messages=[UserMessage(content="Error test")]))
        
        self.assertIsNotNone(response.error)
        self.assertIsInstance(response.error, Error)
        self.assertIsNotNone(response.error.error)
        self.assertIsInstance(response.error.error, ErrorMessage)
        self.assertEqual(response.error.error.message, "Vertex SDK Internal Error")
        self.assertEqual(response.error.error.type, "Exception")
        self.assertIsNone(response.error.error.code)

    @patch('vertexai.init')
    @patch('vertexai.generative_models.GenerativeModel')
    async def test_chat_stream_sdk_error_handling(self, MockSDKGenModelError, mock_vertex_init_err, MockSDKContentUnused, MockSDKPartUnused):
        sdk_client_err_mock = MockSDKGenModelError.return_value
        
        mocked_stream_exception = Exception("Vertex SDK Stream Error")
        async def error_stream_generator():
            raise mocked_stream_exception
            yield # Unreachable, but makes it a generator

        sdk_client_err_mock.generate_content_async.return_value = error_stream_generator()
        
        with patch('vertexai.generative_models.GenerativeModel', return_value=sdk_client_err_mock):
             error_chat_model = VertexAIChatModel(model_id=TEST_CHAT_MODEL_ID)

        error_response_chunk = None
        try:
            async for chunk_bytes in error_chat_model.chat_stream(ChatRequest(model=TEST_CHAT_MODEL_ID, messages=[UserMessage(content="Stream error test")], stream=True)):
                if chunk_bytes == b"data: [DONE]\n\n":
                    continue
                json_str = chunk_bytes.decode('utf-8').replace("data: ", "").strip()
                parsed_chunk = ChatStreamResponse(**json.loads(json_str))
                if parsed_chunk.error:
                    error_response_chunk = parsed_chunk
                    break 
        except Exception as e:
            self.fail(f"Stream processing raised an unexpected exception: {e}")

        self.assertIsNotNone(error_response_chunk, "Error chunk not found in stream")
        self.assertIsInstance(error_response_chunk, ChatStreamResponse)
        self.assertIsNotNone(error_response_chunk.error)
        self.assertIsInstance(error_response_chunk.error, Error)
        self.assertIsNotNone(error_response_chunk.error.error)
        self.assertIsInstance(error_response_chunk.error.error, ErrorMessage)
        self.assertEqual(error_response_chunk.error.error.message, "Vertex SDK Stream Error")
        self.assertEqual(error_response_chunk.error.error.type, "Exception")
        self.assertIsNone(error_response_chunk.error.error.code)
        self.assertEqual(len(error_response_chunk.choices), 0)


@patch('vertexai.init')
@patch('vertexai.language_models.TextEmbeddingModel')
class TestVertexAIEmbeddingsModel(unittest.IsolatedAsyncioTestCase):

    def _assert_init_called_with_settings(self, mock_vertex_init_func):
        mock_vertex_init_func.assert_any_call(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_LOCATION)

    def setUp(self, MockSDKTextEmbeddingModel, mock_vertex_ai_init):
        self.mock_vertex_ai_init_global = mock_vertex_ai_init
        self.sdk_embedding_client_mock = MockSDKTextEmbeddingModel.from_pretrained.return_value
        self.sdk_embedding_client_mock.get_embeddings_async = AsyncMock()
        self.embedding_model = VertexAIEmbeddingsModel(model_id=TEST_EMBEDDING_MODEL_ID)
        self._assert_init_called_with_settings(mock_vertex_ai_init)

    async def test_list_models(self, MockSDKTextEmbeddingModelUnused, mock_vertex_ai_init_unused):
        """Tests the list_models method for VertexAIEmbeddingsModel."""
        models = await self.embedding_model.list_models()
        self.assertIsInstance(models, list)

        expected_embedding_models_details = {
            mid: details for mid, details in KNOWN_VERTEX_AI_MODELS.items() if details['type'] == 'embedding'
        }
        self.assertEqual(len(models), len(expected_embedding_models_details))

        for model_card in models:
            self.assertIsInstance(model_card, ModelCard)
            self.assertEqual(model_card.type, ModelTypeEnum.EMBEDDING)
            self.assertIn(model_card.id, expected_embedding_models_details)
            
            # Verify other details from KNOWN_VERTEX_AI_MODELS
            details = expected_embedding_models_details[model_card.id]
            self.assertEqual(model_card.output_dimensions, details.get('output_dimensions'))
            # Check common ModelCard fields
            self.assertIsInstance(model_card.created, int)
            self.assertEqual(model_card.object, "model_card")
            self.assertEqual(model_card.owned_by, "general")

    async def test_embed_single_input(self, MockSDKTextEmbeddingModel, mock_vertex_ai_init):
        mock_sdk_emb_response = [MockVertexSDKTextEmbedding([0.1, 0.2, 0.3], {"token_count": 3, "truncated": False})]
        self.sdk_embedding_client_mock.get_embeddings_async.return_value = mock_sdk_emb_response
        response = await self.embedding_model.embed(EmbeddingsRequest(model=TEST_EMBEDDING_MODEL_ID, input="Hello"))
        self.assertEqual(response.data[0]['embedding'], [0.1, 0.2, 0.3])
        self.assertEqual(response.usage.prompt_tokens, 3)

    async def test_embed_list_input(self, MockSDKTextEmbeddingModel, mock_vertex_ai_init):
        mock_sdk_emb_response = [
            MockVertexSDKTextEmbedding([0.1], {"token_count": 1, "truncated": False}),
            MockVertexSDKTextEmbedding([0.2], {"token_count": 1, "truncated": False})
        ]
        self.sdk_embedding_client_mock.get_embeddings_async.return_value = mock_sdk_emb_response
        response = await self.embedding_model.embed(EmbeddingsRequest(model=TEST_EMBEDDING_MODEL_ID, input=["Hi", "Lo"]))
        self.assertEqual(len(response.data), 2)
        self.assertEqual(response.usage.prompt_tokens, 2)

    @patch('vertexai.init')
    @patch('vertexai.language_models.TextEmbeddingModel')
    async def test_embed_sdk_error_handling(self, MockSDKEmbModelError, mock_vertex_init_err, MockSDKTEModelOuter, mockVAEInitOuter):
        sdk_emb_client_err_mock = MockSDKEmbModelError.from_pretrained.return_value
        # Ensure the side_effect is an Exception instance
        mocked_exception = Exception("Vertex SDK Embedding Error")
        sdk_emb_client_err_mock.get_embeddings_async = AsyncMock(side_effect=mocked_exception)
        
        with patch('vertexai.language_models.TextEmbeddingModel.from_pretrained', return_value=sdk_emb_client_err_mock):
            error_embedding_model = VertexAIEmbeddingsModel(model_id=TEST_EMBEDDING_MODEL_ID)
        
        response = await error_embedding_model.embed(EmbeddingsRequest(model=TEST_EMBEDDING_MODEL_ID, input="Err"))
        
        self.assertIsNotNone(response.error)
        self.assertIsInstance(response.error, Error)
        self.assertIsNotNone(response.error.error)
        self.assertIsInstance(response.error.error, ErrorMessage)
        self.assertEqual(response.error.error.message, "Vertex SDK Embedding Error")
        self.assertEqual(response.error.error.type, "Exception")
        self.assertIsNone(response.error.error.code)

class TestVertexAIFactoryFunctions(unittest.IsolatedAsyncioTestCase): # Use IsolatedAsyncioTestCase

    @patch('vertexai.init')
    @patch('api.models.vertex_ai.VertexAIChatModel')
    async def test_get_chat_model_success(self, MockLocalVertexAIChatModel, mock_vertex_ai_init_factory):
        MockLocalVertexAIChatModel.return_value = MagicMock(spec=VertexAIChatModel)
        get_chat_model(TEST_CHAT_MODEL_ID)
        mock_vertex_ai_init_factory.assert_called_with(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_LOCATION)
        MockLocalVertexAIChatModel.assert_called_once_with(model_id=TEST_CHAT_MODEL_ID)

    @patch('vertexai.init')
    async def test_get_chat_model_invalid_id_raises_valueerror(self, mock_vertex_ai_init_factory_invalid):
        with self.assertRaisesRegex(ValueError, "is not a known Vertex AI chat model"):
            get_chat_model(INVALID_MODEL_ID)

    @patch('vertexai.init')
    @patch('api.models.vertex_ai.VertexAIEmbeddingsModel')
    async def test_get_embeddings_model_success(self, MockLocalVertexAIEmbeddingsModel, mock_vertex_ai_init_factory_emb):
        MockLocalVertexAIEmbeddingsModel.return_value = MagicMock(spec=VertexAIEmbeddingsModel)
        get_embeddings_model(TEST_EMBEDDING_MODEL_ID)
        mock_vertex_ai_init_factory_emb.assert_called_with(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_LOCATION)
        MockLocalVertexAIEmbeddingsModel.assert_called_once_with(model_id=TEST_EMBEDDING_MODEL_ID)

    @patch('vertexai.init')
    async def test_get_embeddings_model_invalid_id_raises_valueerror(self, mock_vertex_ai_init_factory_emb_invalid):
        with self.assertRaisesRegex(ValueError, "is not a known Vertex AI embedding model"):
            get_embeddings_model(INVALID_MODEL_ID)

if __name__ == '__main__':
    unittest.main()
