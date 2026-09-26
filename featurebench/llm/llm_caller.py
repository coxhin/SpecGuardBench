"""LLM caller base class with unified init/call interface."""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import requests
from openai import OpenAI, AzureOpenAI, APIStatusError
from tqdm import tqdm

from featurebench.llm.llm_exceptions import LLMException, LLMAPIException, LLMInitException


@dataclass
class StreamedCompletion:
    """Lightweight container that mimics the OpenAI ChatCompletion interface."""

    content: str = ""
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    id: Optional[str] = None
    system_fingerprint: Optional[str] = None
    usage: Any = None

    @property
    def choices(self):
        """Return a list with a single pseudo-choice matching the OpenAI API shape."""
        return [_StreamedChoice(self.content, self.finish_reason)]


@dataclass
class _StreamedChoice:
    """Pseudo-choice for StreamedCompletion.choices[0]."""
    _content: str
    finish_reason: Optional[str] = None

    @property
    def message(self):
        return _StreamedMessage(self._content)


@dataclass
class _StreamedMessage:
    content: str


@dataclass
class _ResponsesUsage:
    """Normalize Responses-API usage onto the chat/completions field names."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @classmethod
    def from_raw(cls, usage: Any) -> Optional["_ResponsesUsage"]:
        if usage is None:
            return None
        return cls(
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )


class LLMCaller:
    """LLM caller base class for LLM operations."""

    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize LLM caller."""
        self.config = config
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.llm_config = config.llm_config or {}
        self.llm_clients: List[Any] = []
        self.llm_client = None
        self.llm_model = None
        self.is_gpt5 = False
        self.is_vllm = False
        self.base_urls: List[str] = []
        self.wire_api = "chat"
        self._unsupported_responses_params: set = set()
        self._usage_lock = threading.Lock()
        self.usage_totals: Dict[str, int] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        }

        if self.llm_config:
            self._setup_llm_client()

    def _setup_llm_client(self) -> None:
        """Initialize LLM client(s), supporting multiple base_url values."""
        if not self.llm_config:
            raise LLMInitException("LLM config not provided")

        if OpenAI is None:
            raise LLMInitException("Please install openai: pip install openai")

        try:
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("openai._base_client").setLevel(logging.WARNING)

            api_key = self.llm_config.get('api_key')
            base_url_config = self.llm_config.get('base_url')
            self.llm_model = self.llm_config.get('model')
            api_version = self.llm_config.get('api_version')
            backend = self.llm_config.get('backend', '').lower()

            wire_api = str(self.llm_config.get('wire_api', 'chat')).strip().lower()
            if wire_api in ('chat', 'chat_completions', 'chat/completions', ''):
                self.wire_api = 'chat'
            elif wire_api == 'responses':
                self.wire_api = 'responses'
            else:
                raise LLMInitException(
                    f"Unknown wire_api '{wire_api}'; expected 'chat' or 'responses'"
                )

            self.is_vllm = backend == 'vllm'

            if isinstance(base_url_config, str):
                self.base_urls = [base_url_config]
            elif isinstance(base_url_config, list):
                self.base_urls = base_url_config
            else:
                self.base_urls = []

            if self.is_vllm:
                if not self.base_urls:
                    raise LLMInitException("vLLM config incomplete: missing base_url")
                if not api_key:
                    api_key = "EMPTY"
                if not self.llm_model:
                    self.llm_model = self._fetch_vllm_model_name(self.base_urls[0], api_key)
                self.logger.debug(f"Detected vLLM config: backend={backend}, model={self.llm_model}")
            else:
                if not api_key or not self.base_urls or not self.llm_model:
                    raise LLMInitException("LLM config incomplete: missing api_key/base_url/model")

            model_for_check = self.llm_model
            if model_for_check.startswith('azure/'):
                model_for_check = model_for_check[6:]
            self.is_gpt5 = not self.is_vllm and ('gpt-5' in model_for_check.lower() or 'gpt5' in model_for_check.lower())

            for base_url in self.base_urls:
                client = self._create_client(api_key, base_url, api_version)
                self.llm_clients.append(client)

            if self.llm_clients:
                self.llm_client = self.llm_clients[0]

            if api_version and self.llm_model.startswith('azure/'):
                self.llm_model = self.llm_model[6:]
                self.logger.debug(f"Azure prefix detected; normalized model: {self.llm_model}")

            if self.is_gpt5:
                self.logger.debug("Detected GPT-5 model; using special parameter config")

            self.logger.debug(f"Using base_url list: {self.base_urls}")
            self.logger.debug(f"Using wire API: {self.wire_api}")

        except LLMInitException:
            raise
        except Exception as e:
            raise LLMInitException(f"Failed to initialize LLM client: {e}")

    def _create_client(self, api_key: str, base_url: str, api_version: Optional[str]) -> Any:
        """Create a single LLM client."""
        timeout = self.llm_config.get('timeout', 180)

        if api_version:
            if AzureOpenAI is None:
                self.logger.warning("AzureOpenAI not found; using standard OpenAI client")
                return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            else:
                return AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=base_url,
                    api_version=api_version,
                    timeout=timeout
                )
        else:
            return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _fetch_vllm_model_name(self, base_url: str, api_key: Optional[str] = None) -> str:
        """Auto-fetch model name from a vLLM service."""
        try:
            models_url = base_url.rstrip('/') + '/models'

            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            response = requests.get(models_url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            models = data.get('data', [])

            if not models:
                raise LLMInitException(f"vLLM service ({base_url}) returned no models")

            model_name = models[0].get('id')
            if not model_name:
                raise LLMInitException(f"vLLM service ({base_url}) returned invalid model data")

            self.logger.debug(f"Auto-detected model name from vLLM service: {model_name}")
            return model_name

        except requests.RequestException as e:
            raise LLMInitException(f"Failed to connect to vLLM service ({base_url}): {e}")
        except (KeyError, IndexError, ValueError) as e:
            raise LLMInitException(f"Failed to parse vLLM model list: {e}")

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        **kwargs
    ) -> Any:
        """Call the LLM API."""
        if not self.llm_client or not self.llm_model:
            raise LLMInitException("LLM client is not initialized")

        if self.wire_api == 'responses':
            return self._call_responses(messages, temperature, max_tokens, stream, **kwargs)

        api_params = {
            'model': self.llm_model,
            'messages': messages
        }

        api_params.update(kwargs)

        if self.is_vllm:
            if temperature is not None:
                api_params['temperature'] = temperature
            elif 'temperature' not in api_params:
                api_params['temperature'] = self.llm_config.get('llm_temperature', 0.0)

            if max_tokens is not None and max_tokens > 0:
                api_params['max_tokens'] = max_tokens
            elif 'max_tokens' not in api_params:
                config_max_tokens = self.llm_config.get('llm_max_tokens', -1)
                if config_max_tokens > 0:
                    api_params['max_tokens'] = config_max_tokens

        elif self.is_gpt5:
            if max_tokens is not None and max_tokens > 0:
                api_params['max_completion_tokens'] = max_tokens
            elif 'max_completion_tokens' not in api_params:
                config_max_tokens = self.llm_config.get('llm_max_tokens', -1)
                if config_max_tokens > 0:
                    api_params['max_completion_tokens'] = config_max_tokens
        else:
            if temperature is not None:
                api_params['temperature'] = temperature
            elif 'temperature' not in api_params:
                api_params['temperature'] = self.llm_config.get('llm_temperature', 0.0)

            if max_tokens is not None and max_tokens > 0:
                api_params['max_tokens'] = max_tokens
            elif 'max_tokens' not in api_params:
                config_max_tokens = self.llm_config.get('llm_max_tokens', -1)
                if config_max_tokens > 0:
                    api_params['max_tokens'] = config_max_tokens

        if stream:
            api_params['stream'] = True
            api_params['stream_options'] = {"include_usage": True}

        max_rounds = 3
        retry_delay = 1
        last_exception = None

        for round_idx in range(max_rounds):
            for client_idx, client in enumerate(self.llm_clients):
                try:
                    response = client.chat.completions.create(**api_params)
                    if stream:
                        return self._collect_stream(response)
                    self._record_usage(getattr(response, "usage", None))
                    return response
                except Exception as e:
                    last_exception = e
                    base_url = self.base_urls[client_idx] if client_idx < len(self.base_urls) else "unknown"
                    error_details = {
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                    if isinstance(e, APIStatusError):
                        error_details["status_code"] = e.status_code
                        error_details["body"] = e.body
                        error_details["request_id"] = getattr(e, "request_id", None)
                        resp = getattr(e, "response", None)
                        if resp is not None:
                            error_details["response_headers"] = dict(resp.headers)
                    self.logger.warning(
                        f"LLM call failed (round {round_idx + 1}/{max_rounds}, "
                        f"base_url[{client_idx}] {base_url}): {error_details}"
                    )
                    time.sleep(retry_delay)

        self.logger.error(f"LLM API call failed after {max_rounds} rounds: {last_exception}")
        raise LLMAPIException(f"LLM API call failed: {last_exception}")

    def _call_responses(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
        **kwargs
    ) -> Any:
        """Call the Responses API, returning the same StreamedCompletion shape."""
        max_rounds = 3
        retry_delay = 1
        last_exception = None

        for round_idx in range(max_rounds):
            for client_idx, client in enumerate(self.llm_clients):
                api_params = self._build_responses_params(
                    messages, temperature, max_tokens, kwargs
                )
                if stream:
                    api_params['stream'] = True
                try:
                    response = client.responses.create(**api_params)
                    if stream:
                        return self._collect_responses_stream(response)
                    return self._responses_to_completion(response)
                except Exception as e:
                    last_exception = e
                    base_url = (
                        self.base_urls[client_idx]
                        if client_idx < len(self.base_urls) else "unknown"
                    )

                    dropped = self._note_unsupported_param(e)
                    if dropped:
                        self.logger.warning(
                            f"{base_url} rejects Responses parameter '{dropped}'; "
                            f"retrying without it (it will be omitted from now on)"
                        )
                        continue

                    error_details = {"type": type(e).__name__, "message": str(e)}
                    if isinstance(e, APIStatusError):
                        error_details["status_code"] = e.status_code
                        error_details["body"] = e.body
                        error_details["request_id"] = getattr(e, "request_id", None)
                    self.logger.warning(
                        f"LLM call failed (responses API, round {round_idx + 1}/"
                        f"{max_rounds}, base_url[{client_idx}] {base_url}): {error_details}"
                    )
                    time.sleep(retry_delay)

        self.logger.error(
            f"LLM API call failed after {max_rounds} rounds: {last_exception}"
        )
        raise LLMAPIException(f"LLM API call failed: {last_exception}")

    _UNSUPPORTED_PARAM_RE = re.compile(
        r"[Uu]nsupported parameter:?\s*'?\"?([A-Za-z0-9_]+)"
    )

    def _note_unsupported_param(self, error: Exception) -> Optional[str]:
        """Record a parameter the endpoint rejected; return its name if new."""
        match = self._UNSUPPORTED_PARAM_RE.search(str(error))
        if not match:
            return None
        name = match.group(1)
        if name in self._unsupported_responses_params:
            return None
        self._unsupported_responses_params.add(name)
        return name

    def _build_responses_params(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Translate chat-style messages into Responses-API parameters."""
        instructions: List[str] = []
        input_items: List[Dict[str, Any]] = []

        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '') or ''
            if role == 'system':
                instructions.append(content)
                continue
            content_type = 'output_text' if role == 'assistant' else 'input_text'
            input_items.append({
                'role': role,
                'content': [{'type': content_type, 'text': content}],
            })

        api_params: Dict[str, Any] = {
            'model': self.llm_model,
            'input': input_items,
        }
        if instructions:
            api_params['instructions'] = "\n\n".join(instructions)

        api_params.update(kwargs)

        if not self.is_gpt5:
            if temperature is not None:
                api_params['temperature'] = temperature
            elif 'temperature' not in api_params:
                api_params['temperature'] = self.llm_config.get('llm_temperature', 0.0)

        if max_tokens is not None and max_tokens > 0:
            api_params['max_output_tokens'] = max_tokens
        elif 'max_output_tokens' not in api_params:
            config_max_tokens = self.llm_config.get('llm_max_tokens', -1)
            if config_max_tokens > 0:
                api_params['max_output_tokens'] = config_max_tokens

        for name in self._unsupported_responses_params:
            api_params.pop(name, None)

        return api_params

    @staticmethod
    def _responses_finish_reason(response: Any) -> str:
        """Map a Responses-API status onto a chat-style finish_reason."""
        status = getattr(response, 'status', None)
        if status == 'completed':
            return 'stop'
        details = getattr(response, 'incomplete_details', None)
        reason = getattr(details, 'reason', None) if details else None
        if reason in ('max_output_tokens', 'max_tokens'):
            return 'length'
        if reason == 'content_filter':
            return 'content_filter'
        return reason or status or 'unknown'

    def _responses_to_completion(self, response: Any) -> "StreamedCompletion":
        """Wrap a non-streamed Response so callers see the chat-completion shape."""
        self._record_usage(getattr(response, "usage", None))
        return StreamedCompletion(
            content=getattr(response, 'output_text', '') or '',
            finish_reason=self._responses_finish_reason(response),
            model=getattr(response, 'model', None),
            id=getattr(response, 'id', None),
            usage=_ResponsesUsage.from_raw(getattr(response, 'usage', None)),
        )

    def _collect_responses_stream(self, stream) -> "StreamedCompletion":
        """Consume a Responses-API event stream into a StreamedCompletion."""
        content_parts: List[str] = []
        final_response = None

        for event in stream:
            event_type = getattr(event, 'type', '')
            if event_type == 'response.output_text.delta':
                delta = getattr(event, 'delta', '') or ''
                if delta:
                    content_parts.append(delta)
            elif event_type in (
                'response.completed', 'response.incomplete', 'response.failed'
            ):
                final_response = getattr(event, 'response', None)
            elif event_type == 'error':
                raise LLMAPIException(
                    f"Responses stream returned an error event: "
                    f"{getattr(event, 'message', event)}"
                )

        if final_response is None:
            raise LLMAPIException(
                "Responses stream ended without a terminal event (truncated response)"
            )

        error = getattr(final_response, 'error', None)
        if error:
            raise LLMAPIException(f"Responses API returned an error: {error}")

        content = "".join(content_parts)
        if not content:
            content = getattr(final_response, 'output_text', '') or ''

        self._record_usage(getattr(final_response, 'usage', None))
        return StreamedCompletion(
            content=content,
            finish_reason=self._responses_finish_reason(final_response),
            model=getattr(final_response, 'model', None),
            id=getattr(final_response, 'id', None),
            usage=_ResponsesUsage.from_raw(getattr(final_response, 'usage', None)),
        )

    def _collect_stream(self, stream) -> "StreamedCompletion":
        """Consume an OpenAI streaming response and return a StreamedCompletion."""
        content_parts: List[str] = []
        finish_reason = None
        model = None
        completion_id = None
        system_fingerprint = None
        usage = None

        for chunk in stream:
            if not chunk.choices and hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
                continue

            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content_parts.append(delta.content)
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            if chunk.model:
                model = chunk.model
            if chunk.id:
                completion_id = chunk.id
            if getattr(chunk, "system_fingerprint", None):
                system_fingerprint = chunk.system_fingerprint

        if finish_reason is None:
            raise LLMAPIException(
                "Stream ended without finish_reason (truncated response)"
            )

        self._record_usage(usage)
        return StreamedCompletion(
            content="".join(content_parts),
            finish_reason=finish_reason,
            model=model,
            id=completion_id,
            system_fingerprint=system_fingerprint,
            usage=usage,
        )

    @staticmethod
    def _usage_field(usage: Any, *names: str) -> int:
        """First present integer among ``names``, else 0."""
        for name in names:
            value = getattr(usage, name, None)
            if isinstance(value, int):
                return value
        return 0

    def _record_usage(self, usage: Any) -> None:
        """Add one response's token usage to this caller's running total."""
        with self._usage_lock:
            self.usage_totals["calls"] += 1
            if usage is None:
                return

            prompt = self._usage_field(usage, "prompt_tokens", "input_tokens")
            completion = self._usage_field(usage, "completion_tokens", "output_tokens")
            self.usage_totals["prompt_tokens"] += prompt
            self.usage_totals["completion_tokens"] += completion
            self.usage_totals["total_tokens"] += self._usage_field(
                usage, "total_tokens"
            ) or (prompt + completion)

            prompt_details = getattr(usage, "prompt_tokens_details", None) or getattr(
                usage, "input_tokens_details", None
            )
            if prompt_details is not None:
                self.usage_totals["cached_prompt_tokens"] += self._usage_field(
                    prompt_details, "cached_tokens"
                )
                self.usage_totals["cache_write_tokens"] += self._usage_field(
                    prompt_details, "cache_write_tokens", "cache_creation_tokens"
                )

            completion_details = getattr(
                usage, "completion_tokens_details", None
            ) or getattr(usage, "output_tokens_details", None)
            if completion_details is not None:
                self.usage_totals["reasoning_tokens"] += self._usage_field(
                    completion_details, "reasoning_tokens"
                )

    def snapshot_usage(self) -> Dict[str, int]:
        """Return a copy of the running account, taken between calls."""
        with self._usage_lock:
            return dict(self.usage_totals)

    def get_llm_response_text(self, completion) -> str:
        """Extract text content from an LLM response."""
        choice = completion.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)

        usage = getattr(completion, "usage", None)
        if usage:
            self.logger.debug(
                f"LLM usage: prompt_tokens={getattr(usage, 'prompt_tokens', '?')}, "
                f"completion_tokens={getattr(usage, 'completion_tokens', '?')}, "
                f"finish_reason={finish_reason}"
            )

        if finish_reason != "stop":
            response_details = {
                "finish_reason": finish_reason,
                "content_length": len(content),
                "content_preview": content[:500] if content else "",
            }
            if usage:
                response_details["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            response_details["model"] = getattr(completion, "model", None)
            response_details["id"] = getattr(completion, "id", None)
            response_details["system_fingerprint"] = getattr(completion, "system_fingerprint", None)
            error = getattr(completion, "error", None)
            if error:
                response_details["error"] = error if isinstance(error, (str, dict)) else str(error)
            self.logger.warning(
                f"LLM response with non-stop finish_reason: {response_details}"
            )

            if finish_reason == "length":
                self.logger.warning(
                    f"LLM response truncated (finish_reason=length). "
                    f"Response length: {len(content)} chars"
                )
            elif finish_reason == "content_filter":
                self.logger.warning("LLM response blocked by content filter")
            elif not content:
                self.logger.warning(
                    f"LLM returned empty response (finish_reason={finish_reason})"
                )

        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        return content
