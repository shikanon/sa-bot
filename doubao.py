
"""
doubao chat wrapper.
author: shikanon
create: 2023/10/2
"""
from __future__ import annotations
from langchain.pydantic_v1 import Field, root_validator
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
)

from retrying import retry
from volcengine.maas import MaasService, MaasException, ChatRole
from volcengine.maas.models.api.api_pb2 import ChatResp
from langchain.adapters.openai import convert_dict_to_message, convert_message_to_dict

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import create_base_retry_decorator
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema import AIMessage, HumanMessage, ChatMessage, SystemMessage
from langchain.schema.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
)
from langchain.schema.output import ChatGenerationChunk
from langchain.utils import get_from_dict_or_env, get_pydantic_field_names

class ChatSkylark(BaseChatModel):
    model_name: str = Field(default="skylark-chat", alias="model")
    """VOLC_ACCESSKEY"""
    model_ak: Optional[str] = None
    """VOLC_SECRETKEY"""
    model_sk: Optional[str] = None
    # 输出文本的最大tokens限制
    max_tokens: Optional[int] = None
    """temperature:用于控制生成文本的随机性和创造性，Temperature值越大随机性越大，取值范围0~1"""
    temperature: float = 0.7
    """top_p:用于控制输出tokens的多样性，TopP值越大输出的tokens类型越丰富，取值范围0~1"""
    top_p: float = 0.9
    """top_k:选择预测值最大的k个token进行采样，取值范围0-1000，0表示不生效"""
    top_k: Optional[int] = None
    maas: MaasService = MaasService('maas-api.ml-platform-cn-beijing.volces.com', 'cn-beijing')
    streaming: bool = False

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "skylark-chat"
    
    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> Dict[str, Any]:
        messages_list = []
        for m in messages:
            if isinstance(m, SystemMessage):
                message_dict = {"role": ChatRole.SYSTEM, "content": m.content}
            elif isinstance(m, HumanMessage):
                message_dict = {"role": ChatRole.USER, "content": m.content}
            elif isinstance(m, AIMessage):
                message_dict = {"role": ChatRole.ASSISTANT, "content": m.content}
            else:
                message_dict = {"role": ChatRole.USER, "content": m.content}
            messages_list.append(message_dict)
        req = {
            "model": {
                "name": self.model_name,
            },
            "parameters": {
                # 输出文本的最大tokens限制
                "max_new_tokens": self.max_tokens,
                # 用于控制生成文本的随机性和创造性，Temperature值越大随机性越大，取值范围0~1
                "temperature": self.temperature,
                # 用于控制输出tokens的多样性，TopP值越大输出的tokens类型越丰富，取值范围0~1
                "top_p": self.top_p,
                # 选择预测值最大的k个token进行采样，取值范围0-1000，0表示不生效
                "top_k": self.top_k,
            },
            "messages": messages_list
        }
        return req

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        res = response["choice"]
        message = convert_dict_to_message(res["message"])
        gen = ChatGeneration(
            message=message,
            generation_info=dict(finish_reason=res.get("finish_reason")),
        )
        generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    def completion_with_retry(
        self, **kwargs: Any
    ) -> Any:

        @retry(stop_max_attempt_number=3)
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.maas.chat(**kwargs)

        return _completion_with_retry(**kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # todo: write stream chunk function
        pass

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            generation: Optional[ChatGenerationChunk] = None
            for chunk in self._stream(
                messages=messages, **kwargs
            ):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])
        # 将统一的message转换为maas处理的参数
        params = self._create_message_dicts(messages)
        response = self.completion_with_retry(req=params)
        # convert ChatResp to dict
        assert isinstance(response, ChatResp)
        response_dict = {
            "choice": {
                "message": {
                    "role": response.choice.message.role,
                    "content": response.choice.message.content
                },
                "finish_reason": response.choice.finish_reason,
            },
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        return self._create_chat_result(response_dict)